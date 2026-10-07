import Dispatch
import Foundation

public protocol EmbeddingPredictor: Sendable {
    func load() async throws -> LoadReport
    func predict(_ request: PredictRequest) async throws -> PredictResult
    func unload() async throws -> UnloadReport
}

public struct LoadReport: Codable, Sendable, Equatable {
    public let perChunkNS: [UInt64]
    public let residentBytes: UInt64
    public let computePlanChecked: Bool
    public let visionLoaded: Bool?
    enum CodingKeys: String, CodingKey {
        case perChunkNS = "per_chunk_ns", residentBytes = "resident_bytes"
        case computePlanChecked = "compute_plan_checked"
        case visionLoaded = "vision_loaded"
    }
    public init(perChunkNS: [UInt64], residentBytes: UInt64, computePlanChecked: Bool, visionLoaded: Bool? = nil) {
        self.perChunkNS = perChunkNS
        self.residentBytes = residentBytes
        self.computePlanChecked = computePlanChecked
        self.visionLoaded = visionLoaded
    }
    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        perChunkNS = try c.decode([UInt64].self, forKey: .perChunkNS)
        residentBytes = try c.decode(UInt64.self, forKey: .residentBytes)
        computePlanChecked = try c.decode(Bool.self, forKey: .computePlanChecked)
        visionLoaded = try c.decodeIfPresent(Bool.self, forKey: .visionLoaded)
    }
    public func encode(to encoder: any Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(perChunkNS, forKey: .perChunkNS)
        try c.encode(residentBytes, forKey: .residentBytes)
        try c.encode(computePlanChecked, forKey: .computePlanChecked)
        try c.encodeIfPresent(visionLoaded, forKey: .visionLoaded)
    }
}

public struct UnloadReport: Codable, Sendable, Equatable {
    public let residentBytes: UInt64
    enum CodingKeys: String, CodingKey { case residentBytes = "resident_bytes" }
    public init(residentBytes: UInt64) { self.residentBytes = residentBytes }
}

/// Immutable, whole-batch tokenization result. An adapter must consume these ids
/// as-is, in order, with one B=1 cascade per element. It must not retokenize.
public struct PredictRequest: Sendable {
    public let inputs: [TokenizedInput]
    public let tokenizeNS: UInt64
    public var promptTokens: Int { inputs.reduce(0) { $0 + $1.nTokens } }
    public init(inputs: [TokenizedInput], tokenizeNS: UInt64 = 0) throws {
        guard !inputs.isEmpty else { throw EmbedANEError.emptyInput(index: nil) }
        guard inputs.count <= 8 else { throw EmbedANEError.batchTooLarge(inputs.count) }
        self.inputs = inputs
        self.tokenizeNS = tokenizeNS
    }
}

public struct PredictResult: Sendable {
    public let embeddings: [[Float]]
    public let tableLookupNS: UInt64
    public let perChunkNS: [UInt64]
    public init(embeddings: [[Float]], tableLookupNS: UInt64 = 0,
                perChunkNS: [UInt64] = Array(repeating: 0, count: 6)) {
        self.embeddings = embeddings
        self.tableLookupNS = tableLookupNS
        self.perChunkNS = perChunkNS
    }
    public func validate(for request: PredictRequest) throws {
        guard embeddings.count == request.inputs.count,
              embeddings.allSatisfy({ $0.count == ModelABI.dimension && $0.allSatisfy(\.isFinite) }),
              perChunkNS.count == 6 else {
            throw EmbedANEError.abiMismatch(chunk: 5, reason: "Predictor returned invalid batch, dimensions, floats, or timings.")
        }
    }
}

public protocol EmbeddingPreparer: Sendable {
    func prepare(_ texts: [String]) async throws -> PredictRequest
}

/// Tokenization is blocking CPU work, so it does not run on an HTTP event loop
/// or on the cooperative executor. This also serializes a dependency tokenizer.
public final class SerialTokenizationPreparer: EmbeddingPreparer, Sendable {
    private let tokenizer: any ContentTokenizer
    private let queue = DispatchQueue(label: "embed-ane.tokenization", qos: .userInitiated)
    public init(tokenizer: any ContentTokenizer) { self.tokenizer = tokenizer }
    public func prepare(_ texts: [String]) async throws -> PredictRequest {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [tokenizer] in
                let start = DispatchTime.now().uptimeNanoseconds
                do {
                    let inputs = try TokenizedInput.prepare(texts, tokenizer: tokenizer)
                    continuation.resume(returning: try PredictRequest(
                        inputs: inputs, tokenizeNS: DispatchTime.now().uptimeNanoseconds - start))
                } catch { continuation.resume(throwing: error) }
            }
        }
    }
}

/// Non-Sendable CoreML objects belong behind this synchronous interface.
/// Factory construction and every invocation occur on ONE dedicated queue.
public protocol BlockingEmbeddingBackend: AnyObject {
    func load() throws -> LoadReport
    func predict(_ request: PredictRequest) throws -> PredictResult
    func unload() throws -> UnloadReport
}

/// Optional synchronous preparation on the SAME backend/queue as prediction.
public protocol BlockingEmbeddingPreparer: AnyObject {
    func prepare(_ texts: [String]) throws -> PredictRequest
}

/// CoreML's async metadata API also uses this queue through executor preference.
/// No semaphore blocks the queue while the framework is doing asynchronous I/O.
final class QueueTaskExecutor: TaskExecutor {
    let queue: DispatchQueue
    init(queue: DispatchQueue) { self.queue = queue }
    func enqueue(_ job: UnownedJob) {
        queue.async { [self] in job.runSynchronously(on: asUnownedTaskExecutor()) }
    }
}

public final class SerialEmbeddingWorker: EmbeddingPredictor, EmbeddingPreparer, @unchecked Sendable {
    private let queue: DispatchQueue
    private let executor: QueueTaskExecutor
    private let factory: @Sendable () throws -> any BlockingEmbeddingBackend
    // The box is accessed only on queue, including the final backend release.
    private final class Storage: @unchecked Sendable {
        var backend: (any BlockingEmbeddingBackend)?
    }
    private let storage = Storage()
    public init(factory: @escaping @Sendable () throws -> any BlockingEmbeddingBackend) {
        self.factory = factory
        let queue = DispatchQueue(label: "embed-ane.cascade", qos: .userInitiated, autoreleaseFrequency: .workItem)
        self.queue = queue
        executor = QueueTaskExecutor(queue: queue)
    }
    deinit {
        let storage = storage
        queue.async { storage.backend = nil }
    }
    // Package-internal access also lets the multimodal facade use this SAME
    // queue/resource ownership discipline without exporting CoreML objects.
    func perform<T: Sendable>(
        _ operation: @escaping @Sendable (any BlockingEmbeddingBackend) throws -> T
    ) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [self] in
                do {
                    let instance: any BlockingEmbeddingBackend
                    if let backend = storage.backend { instance = backend }
                    else { instance = try factory(); storage.backend = instance }
                    continuation.resume(returning: try operation(instance))
                } catch { continuation.resume(throwing: error) }
            }
        }
    }
    /// For async framework metadata only; model objects never leave perform().
    func withExecutor<T: Sendable>(_ operation: @escaping @Sendable () async throws -> T) async throws -> T {
        try await Task(executorPreference: executor, operation: operation).value
    }
    public func prepare(_ texts: [String]) async throws -> PredictRequest {
        try await perform { backend in
            guard let preparer = backend as? any BlockingEmbeddingPreparer else {
                throw EmbedANEError.invalidRequest("Backend has no tokenizer.", param: nil)
            }
            return try preparer.prepare(texts)
        }
    }
    public func load() async throws -> LoadReport { try await perform { try $0.load() } }
    public func predict(_ request: PredictRequest) async throws -> PredictResult {
        try await perform { try $0.predict(request) }
    }
    public func unload() async throws -> UnloadReport { try await perform { try $0.unload() } }
}
