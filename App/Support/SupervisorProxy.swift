import EmbedANECore
import EmbedANEHTTP
import Foundation
import HTTPTypes
import Hummingbird
import NIOCore

public protocol SupervisorSessionControlling: AnyObject, Sendable {
    func getWorkerState() async -> WorkerSessionState
    func ensureWorkerStarted() async
    func loadWorker() async throws -> LoadReport
    func unloadWorker() async throws -> UnloadReport
    func getEffectiveConfiguration() async -> ServiceConfiguration
    func applyConfigurationOverrides(_ patch: ConfigurationOverrides) async throws -> EffectiveSettings
    func recordInferenceActivity() async
    func finishInferenceActivity() async
    func getLastError() async -> String?
}

extension SupervisorSessionControlling {
    public func getLastError() async -> String? { nil }
}

private final class SupervisorListenerState: @unchecked Sendable {
    private let lock = NSLock()
    private var running = false
    private var port = 0
    func begin() throws {
        try lock.withLock {
            guard !running else { throw EmbedANEError.conflict("This supervisor proxy is already running.") }
            running = true
        }
    }
    func end() { lock.withLock { running = false; port = 0 } }
    func setPort(_ value: Int) { lock.withLock { port = value } }
    func getPort() -> Int { lock.withLock { port } }
}

public final class SupervisorProxy: @unchecked Sendable {
    public static let bindHost = "127.0.0.1"
    private let listenPort: Int
    public let internalPort: Int
    private let configuration: ServiceConfiguration
    private let store: ConfigurationStore
    private let tokenDirectory: URL
    private let session: any SupervisorSessionControlling
    private let listener = SupervisorListenerState()
    private let clock: any RuntimeClock

    public init(
        listenPort: Int,
        internalPort: Int,
        configuration: ServiceConfiguration,
        store: ConfigurationStore,
        session: any SupervisorSessionControlling,
        tokenDirectory: URL? = nil,
        clock: any RuntimeClock = MonotonicRuntimeClock()
    ) {
        self.listenPort = listenPort
        self.internalPort = internalPort
        self.configuration = configuration
        self.store = store
        self.session = session
        self.tokenDirectory = tokenDirectory ?? store.home.appendingPathComponent(".embed-ane")
        self.clock = clock
    }

    public func run(onServerRunning: @escaping @Sendable (Int) async -> Void = { _ in }) async throws {
        try listener.begin()
        defer { listener.end() }
        let token = try ControlToken.loadOrCreate(in: tokenDirectory)
        try store.saveIfAbsent(configuration)
        let responder = SupervisorProxyResponder(
            listener: listener,
            internalPort: internalPort,
            configuration: configuration,
            store: store,
            session: session,
            policy: ControlAccessPolicy(token: token),
            startedNS: clock.nowNS(),
            clock: clock
        )
        let application = Application(
            responder: responder,
            configuration: .init(
                address: .hostname(Self.bindHost, port: listenPort),
                serverName: "embed-ane-supervisor",
                reuseAddress: false
            ),
            onServerRunning: { channel in
                if let port = channel.localAddress?.port {
                    self.listener.setPort(port)
                    await onServerRunning(port)
                }
            }
        )
        do { try await application.run() }
        catch is CancellationError { throw CancellationError() }
        catch {
            throw EmbedANEError.io(path: "\(Self.bindHost):\(listenPort)", reason: "Supervisor proxy failed: \(error)")
        }
    }
}

private struct SupervisorProxyResponder: HTTPResponder {
    typealias Context = BasicRequestContext
    let listener: SupervisorListenerState
    let internalPort: Int
    let configuration: ServiceConfiguration
    let store: ConfigurationStore
    let session: any SupervisorSessionControlling
    let policy: ControlAccessPolicy
    let startedNS: UInt64
    let clock: any RuntimeClock
    private let urlSession: URLSession

    init(
        listener: SupervisorListenerState,
        internalPort: Int,
        configuration: ServiceConfiguration,
        store: ConfigurationStore,
        session: any SupervisorSessionControlling,
        policy: ControlAccessPolicy,
        startedNS: UInt64,
        clock: any RuntimeClock
    ) {
        self.listener = listener
        self.internalPort = internalPort
        self.configuration = configuration
        self.store = store
        self.session = session
        self.policy = policy
        self.startedNS = startedNS
        self.clock = clock
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 180
        config.timeoutIntervalForResource = 300
        config.connectionProxyDictionary = [:]
        self.urlSession = URLSession(configuration: config)
    }

    func respond(to request: Request, context: BasicRequestContext) async throws -> Response {
        do {
            let port = listener.getPort()
            let host = try ControlAccessPolicy.validateHost(request.head.authority, boundPort: port)
            let path = String(request.uri.path)

            if path == "/control" || path.hasPrefix("/control/") {
                try policy.authorize(
                    host: host,
                    boundPort: port,
                    origins: request.headers.filter { $0.name == .origin }.map(\.value),
                    authorization: request.headers.filter { $0.name == .authorization }.map(\.value)
                )
            }

            let body = try await readBody(request)

            switch (request.method, path) {
            case (.post, "/control/load"):
                try requireEmptyCommand(body)
                let report = try await session.loadWorker()
                return try HTTPWire.json(LoadResponse(state: .ready, report: report))

            case (.post, "/control/unload"):
                try requireEmptyCommand(body)
                let report = try await session.unloadWorker()
                return try HTTPWire.json(UnloadResponse(state: .unloaded, report: report))

            case (.get, "/control/stats"):
                let state = await session.getWorkerState()
                if state == .ready {
                    if let forwarded = try? await forward(request: request, body: body, path: path, method: .get) {
                        return forwarded
                    }
                }
                let emptyStats = RuntimeStatistics(
                    runtime: RuntimeSnapshot(
                        state: (state == .starting) ? LifecycleState.loading : (state == .failed ? LifecycleState.failed : LifecycleState.unloaded),
                        queueDepth: 0,
                        inFlight: 0,
                        preparing: 0,
                        residentBytes: 0,
                        retryAtNS: (nil as UInt64?)
                    ),
                    admittedRequests: 0,
                    completedRequests: 0,
                    failedRequests: 0,
                    windowCount: 0,
                    p50NS: 0,
                    p95NS: 0,
                    queueWait: TimingPercentiles(samples: []),
                    tokenization: TimingPercentiles(samples: []),
                    tableLookup: TimingPercentiles(samples: []),
                    perChunk: (0..<6).map { _ in TimingPercentiles(samples: []) },
                    residentBytesScope: "process",
                    lastError: await session.getLastError()
                )
                return try HTTPWire.json(emptyStats)

            case (.get, "/control/settings"):
                let state = await session.getWorkerState()
                if state == .ready {
                    if let forwarded = try? await forward(request: request, body: body, path: path, method: .get) {
                        return forwarded
                    }
                }
                let effective = await session.getEffectiveConfiguration()
                return try HTTPWire.json(EffectiveSettings(configuration: effective, restartRequired: []))

            case (.put, "/control/settings"):
                let patch = try HTTPWire.decode(ConfigurationOverrides.self, data: body)
                let updated = try await session.applyConfigurationOverrides(patch)
                return try HTTPWire.json(updated)

            case (.get, "/health"):
                let state = await session.getWorkerState()
                if state == .ready {
                    if let forwarded = try? await forward(request: request, body: body, path: path, method: .get) {
                        return forwarded
                    }
                }
                let workerStatus = (state == .starting) ? "starting" : (state == .failed ? "failed" : "down")
                let modelState: LifecycleState = (state == .starting) ? .loading : (state == .failed ? .failed : .unloaded)
                let now = clock.nowNS()
                let uptime = Double(now >= startedNS ? now - startedNS : 0) / 1_000_000_000
                let effective = await session.getEffectiveConfiguration()
                let health = SupervisorHealthResponse(
                    worker: workerStatus,
                    model: .init(id: effective.modelID, state: modelState, queueDepth: 0, residentBytes: 0),
                    uptimeS: uptime
                )
                return try HTTPWire.json(health)

            case (.get, "/v1/models"):
                let state = await session.getWorkerState()
                if state == .ready {
                    if let forwarded = try? await forward(request: request, body: body, path: path, method: .get) {
                        return forwarded
                    }
                }
                let effective = await session.getEffectiveConfiguration()
                return try HTTPWire.json(ModelsResponse(data: [.init(id: effective.modelID)]))

            default:
                if path.hasPrefix("/v1/") || path == "/v1/embeddings" {
                    await session.recordInferenceActivity()
                    defer { Task { await session.finishInferenceActivity() } }
                    let state = await session.getWorkerState()
                    if state == .ready {
                        do {
                            return try await forward(request: request, body: body, path: path, method: request.method)
                        } catch {
                            // Forwarding failure indicates worker is down or crashed
                            await session.ensureWorkerStarted()
                            return makeOverloadedResponse()
                        }
                    } else {
                        await session.ensureWorkerStarted()
                        return makeOverloadedResponse()
                    }
                }
                throw EmbedANEError.invalidRequest("Unknown route or unsupported method.", param: nil)
            }
        } catch {
            return HTTPWire.error(error)
        }
    }

    private func makeOverloadedResponse() -> Response {
        HTTPWire.error(
            message: "The model is currently loading or offline. Please retry after a brief pause.",
            type: "overloaded_error",
            code: "overloaded",
            param: nil,
            status: 503,
            retryAfter: "20"
        )
    }

    private func forward(
        request: Request,
        body: Data,
        path: String,
        method: HTTPRequest.Method
    ) async throws -> Response {
        guard var components = URLComponents(string: "http://127.0.0.1:\(internalPort)") else {
            throw EmbedANEError.invalidRequest("Invalid internal worker URL.", param: nil)
        }
        components.percentEncodedPath = request.uri.path
        components.percentEncodedQuery = request.uri.query
        guard let url = components.url else {
            throw EmbedANEError.invalidRequest("Invalid forwarding URL.", param: nil)
        }
        var forwardRequest = URLRequest(url: url)
        forwardRequest.httpMethod = method.rawValue
        forwardRequest.timeoutInterval = 180
        for header in request.headers {
            if header.name.rawName.lowercased() != "host" {
                forwardRequest.addValue(header.value, forHTTPHeaderField: header.name.rawName)
            }
        }
        forwardRequest.setValue("127.0.0.1:\(internalPort)", forHTTPHeaderField: "Host")
        if !body.isEmpty {
            forwardRequest.httpBody = body
        }
        let (data, urlResponse) = try await urlSession.data(for: forwardRequest)
        guard let httpResponse = urlResponse as? HTTPURLResponse else {
            throw EmbedANEError.transport(path: path, reason: "Target worker returned invalid HTTP response.")
        }
        var responseHeaders = HTTPFields()
        for (key, value) in httpResponse.allHeaderFields {
            if let keyStr = key as? String, let valStr = value as? String {
                let lower = keyStr.lowercased()
                if lower == "content-length" || lower == "transfer-encoding" || lower == "connection" {
                    continue
                }
                if let fieldName = HTTPField.Name(keyStr) {
                    responseHeaders[fieldName] = valStr
                }
            }
        }
        return Response(
            status: .init(code: httpResponse.statusCode),
            headers: responseHeaders,
            body: .init(byteBuffer: ByteBuffer(bytes: data))
        )
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
