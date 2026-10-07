import EmbedANECore
import Foundation
import Hummingbird

private final class ListenerState: @unchecked Sendable {
    private let lock = NSLock()
    private var running = false
    private var port = 0
    func begin() throws {
        try lock.withLock {
            guard !running else { throw EmbedANEError.conflict("This server instance is already running.") }
            running = true
        }
    }
    func end() { lock.withLock { running = false; port = 0 } }
    func setPort(_ value: Int) { lock.withLock { port = value } }
    func getPort() -> Int { lock.withLock { port } }
}

/// Transport facade for App/CLI embedding. No CoreML adapter or mock is selected
/// implicitly: the composition root must inject its predictor and preparer.
public struct EmbeddingHTTPServer: Sendable {
    public static let bindHost = "127.0.0.1"
    public static let maximumBodyBytes = 1_048_576
    public let lifecycle: LifecycleActor
    public let settings: SettingsController
    private let preparer: any EmbeddingPreparer
    private let configuration: ServiceConfiguration
    private let store: ConfigurationStore
    private let tokenDirectory: URL
    private let listenPort: Int
    private let listener = ListenerState()
    private let clock: any RuntimeClock

    /// portOverride=0 asks the kernel for an ephemeral test listener. No hostname
    /// override exists, and a bind failure is never retried on another port.
    public init(configuration: ServiceConfiguration, predictor: any EmbeddingPredictor,
                preparer: any EmbeddingPreparer, store: ConfigurationStore = .init(),
                tokenDirectory: URL? = nil, portOverride: Int? = nil,
                clock: any RuntimeClock = MonotonicRuntimeClock(),
                residentBytes: @escaping @Sendable () -> UInt64 = { ProcessMemory.residentBytes() }) throws {
        let configuration = store.expandingPaths(configuration)
        try configuration.validate()
        let port = portOverride ?? configuration.port
        guard (0...65535).contains(port) else { throw EmbedANEError.invalidRequest("Invalid listen port.", param: "port") }
        self.configuration = configuration; self.preparer = preparer; self.store = store
        self.tokenDirectory = tokenDirectory ?? store.home.appendingPathComponent(".embed-ane")
        listenPort = port; self.clock = clock
        let lifecycle = try LifecycleActor(predictor: predictor, maxQueueDepth: configuration.maxQueueDepth,
                                           maxBatch: configuration.maxBatch, idleTimeoutS: configuration.idleTimeoutS,
                                           autoLoad: configuration.autoLoad,
                                           clock: clock, residentBytes: residentBytes)
        self.lifecycle = lifecycle
        settings = SettingsController(configuration: configuration, lifecycle: lifecycle, store: store)
    }

    /// Caller owns task cancellation / process signal handling. Hummingbird's
    /// structured service group shuts down the actual listener on cancellation.
    public func run(onServerRunning: @escaping @Sendable (Int) async -> Void = { _ in }) async throws {
        try listener.begin()
        defer { listener.end() }
        let token = try ControlToken.loadOrCreate(in: tokenDirectory)
        try store.saveIfAbsent(configuration)
        let responder = EmbeddingResponder(lifecycle: lifecycle, settings: settings, preparer: preparer,
                                            policy: ControlAccessPolicy(token: token), listener: listener,
                                            startedNS: clock.nowNS(), clock: clock)
        let application = Application(
            responder: responder,
            configuration: .init(address: .hostname(Self.bindHost, port: listenPort),
                                 serverName: "embed-ane", reuseAddress: false),
            onServerRunning: { channel in
                if let port = channel.localAddress?.port {
                    listener.setPort(port)
                    await onServerRunning(port)
                }
            })
        do { try await application.run() }
        catch is CancellationError { throw CancellationError() }
        catch {
            // Preserve the underlying EADDRINUSE diagnostic and the attempted
            // address. There is deliberately no next-port fallback loop.
            throw EmbedANEError.io(path: "\(Self.bindHost):\(listenPort)", reason: "HTTP server failed: \(error)")
        }
    }
}

private struct EmbeddingResponder: HTTPResponder {
    typealias Context = BasicRequestContext
    let lifecycle: LifecycleActor
    let settings: SettingsController
    let preparer: any EmbeddingPreparer
    let policy: ControlAccessPolicy
    let listener: ListenerState
    let startedNS: UInt64
    let clock: any RuntimeClock
    private let imageReader = ImageInputReader()

    func respond(to request: Request, context: BasicRequestContext) async throws -> Response {
        do {
            let port = listener.getPort()
            let host = try ControlAccessPolicy.validateHost(request.head.authority, boundPort: port)
            let path = String(request.uri.path)
            if path == "/control" || path.hasPrefix("/control/") {
                try policy.authorize(host: host, boundPort: port,
                                     origins: request.headers.filter { $0.name == .origin }.map(\.value),
                                     authorization: request.headers.filter { $0.name == .authorization }.map(\.value))
            }
            let body = try await readBody(request)
            switch (request.method, path) {
            case (.post, "/v1/embeddings"):
                let input = try HTTPWire.decode(EmbeddingsRequest.self, data: body)
                let effective = await settings.current().configuration
                try input.validate(maxTexts: min(effective.maxBatch, 8))
                let execution: EmbeddingExecution
                switch input.input {
                case let .texts(texts):
                    execution = try await lifecycle.embed(texts, using: preparer)
                case let .tokens(tokens):
                    // Validate the entire batch before enqueue; no retokenization.
                    // Token arrays are content; they get the same prompt wrapper as text.
                    let inputs = try tokens.enumerated().map { try PromptTemplate.textInput($0.element, index: $0.offset) }
                    execution = try await lifecycle.submit(PredictRequest(inputs: inputs))
                case .parts:
                    let joint = input.input.jointContent
                    if let url = joint.videos.first {
                        let source = try VideoInputSource(url)
                        guard await lifecycle.supportsImages else { throw EmbedANEError.visionNotConfigured }
                        let file = try await imageReader.validate(source)
                        execution = try await lifecycle.submitImage(ImageEmbeddingRequest(videoURL: file, text: joint.text))
                    } else if let url = joint.images.first {
                        let source = try ImageInputSource(url)
                        guard await lifecycle.supportsImages else { throw EmbedANEError.visionNotConfigured }
                        let bytes = try await imageReader.read(source)
                        execution = try await lifecycle.submitImage(ImageEmbeddingRequest(imageData: bytes, text: joint.text))
                    } else {
                        execution = try await lifecycle.embed([joint.text], using: preparer)
                    }
                }
                guard execution.promptTokens <= 4096 else {
                    throw EmbedANEError.abiMismatch(chunk: 0, reason: "Usage exceeded the frozen batch budget.")
                }
                return try HTTPWire.json(EmbeddingsResponse(model: input.model, execution: execution,
                                                            dimensions: input.dimensions, encodingFormat: input.encodingFormat))
            case (.get, "/health"):
                let effective = await settings.current().configuration
                let snapshot = await lifecycle.snapshot()
                let now = clock.nowNS()
                return try HTTPWire.json(HealthResponse(
                    model: .init(id: effective.modelID, state: snapshot.state, queueDepth: snapshot.queueDepth,
                                 residentBytes: snapshot.residentBytes),
                    uptimeS: Double(now >= startedNS ? now - startedNS : 0) / 1_000_000_000))
            case (.get, "/v1/models"):
                let effective = await settings.current().configuration
                return try HTTPWire.json(ModelsResponse(data: [.init(id: effective.modelID)]))
            case (.get, "/control/settings"):
                return try HTTPWire.json(await settings.current())
            case (.put, "/control/settings"):
                let patch = try HTTPWire.decode(ConfigurationOverrides.self, data: body)
                return try HTTPWire.json(await settings.apply(patch))
            case (.post, "/control/load"):
                try requireEmptyCommand(body)
                let report = try await lifecycle.load()
                return try HTTPWire.json(LoadResponse(state: await lifecycle.snapshot().state, report: report))
            case (.post, "/control/unload"):
                try requireEmptyCommand(body)
                let report = try await lifecycle.unload()
                return try HTTPWire.json(UnloadResponse(state: await lifecycle.snapshot().state, report: report))
            case (.get, "/control/stats"):
                return try HTTPWire.json(await lifecycle.statistics())
            default:
                throw EmbedANEError.invalidRequest("Unknown route or unsupported method.", param: nil)
            }
        } catch { return HTTPWire.error(error) }
    }

    private func readBody(_ request: Request) async throws -> Data {
        let lengths = request.headers.filter { $0.name == .contentLength }.map(\.value)
        guard lengths.count <= 1 else { throw EmbedANEError.invalidRequest("Multiple Content-Length fields.", param: nil) }
        if let length = lengths.first {
            guard !length.isEmpty, length.utf8.allSatisfy({ (48...57).contains($0) }) else {
                throw EmbedANEError.invalidRequest("Invalid Content-Length.", param: nil)
            }
            guard let size = UInt64(length), size <= EmbeddingHTTPServer.maximumBodyBytes else { throw EmbedANEError.bodyTooLarge }
        }
        var data = Data()
        // The limit is cumulative and enforced on streamed/chunked bodies too;
        // trusting Content-Length alone would leave a memory-exhaustion path.
        for try await buffer in request.body {
            guard buffer.readableBytes <= EmbeddingHTTPServer.maximumBodyBytes - data.count else {
                throw EmbedANEError.bodyTooLarge
            }
            data.append(contentsOf: buffer.readableBytesView)
        }
        return data
    }
    private func requireEmptyCommand(_ data: Data) throws {
        if data.isEmpty { return }
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any], object.isEmpty else {
            throw EmbedANEError.invalidRequest("Control command body must be absent or {}.", param: nil)
        }
    }
}
