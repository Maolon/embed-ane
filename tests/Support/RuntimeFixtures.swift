import Dispatch
import EmbedANECore
import Foundation

public enum FixtureError: Error, Sendable { case timeout, deliberateFailure, malformedResponse }

public actor AsyncGate {
    private var opened: Bool
    private var waiters: [CheckedContinuation<Void, Never>] = []
    public init(open: Bool = true) { opened = open }
    public func close() { opened = false }
    public func open() {
        opened = true
        let pending = waiters; waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }
    public func wait() async {
        if opened { return }
        await withCheckedContinuation { waiters.append($0) }
    }
}

public final class ManualRuntimeClock: RuntimeClock, @unchecked Sendable {
    private struct Waiter {
        let deadline: UInt64
        let continuation: CheckedContinuation<Void, any Error>
    }
    private let lock = NSLock()
    private var current: UInt64 = 0
    private var waiters: [UUID: Waiter] = [:]
    public init() {}
    public func nowNS() -> UInt64 { lock.withLock { current } }
    public var pendingSleeps: Int { lock.withLock { waiters.count } }
    public func advance(nanoseconds: UInt64) {
        let ready: [Waiter] = lock.withLock {
            current += nanoseconds
            let ids = waiters.filter { $0.value.deadline <= current }.map(\.key)
            return ids.compactMap { waiters.removeValue(forKey: $0) }
        }
        for waiter in ready { waiter.continuation.resume() }
    }
    public func advance(seconds: UInt64) { advance(nanoseconds: seconds * 1_000_000_000) }
    public func sleep(untilNS deadline: UInt64) async throws {
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                lock.withLock {
                    if Task.isCancelled { continuation.resume(throwing: CancellationError()) }
                    else if current >= deadline { continuation.resume() }
                    else { waiters[id] = Waiter(deadline: deadline, continuation: continuation) }
                }
            }
        } onCancel: {
            let waiter = self.lock.withLock { self.waiters.removeValue(forKey: id) }
            waiter?.continuation.resume(throwing: CancellationError())
        }
    }
}

public struct FixtureTokenizer: ContentTokenizer {
    public init() {}
    public func encodeContent(_ text: String) throws -> [Int] { text.utf8.map(Int.init) }
}

public actor CountingPreparer: EmbeddingPreparer {
    public let gate = AsyncGate()
    public private(set) var calls = 0
    private let base = SerialTokenizationPreparer(tokenizer: FixtureTokenizer())
    public init() {}
    public func prepare(_ texts: [String]) async throws -> PredictRequest {
        calls += 1
        await gate.wait()
        return try await base.prepare(texts)
    }
}

public actor ProductionFaithfulPreparer: EmbeddingPreparer {
    private let predictor: MockPredictor
    private let base = SerialTokenizationPreparer(tokenizer: FixtureTokenizer())
    public var failWith: (any Error)?
    public private(set) var calls = 0

    public init(predictor: MockPredictor) {
        self.predictor = predictor
    }

    public func setFailure(_ error: (any Error)?) {
        failWith = error
    }

    public func prepare(_ texts: [String]) async throws -> PredictRequest {
        calls += 1
        guard await predictor.isLoaded else { throw EmbedANEError.modelNotLoaded }
        if let failWith { throw failWith }
        return try await base.prepare(texts)
    }
}

/// The mock predictors echo a per-input "seed": the first CONTENT id. Text and
/// token-array inputs carry the upstream chat-template prefix
/// (`<|im_start|>user\n`), so the seed skips it; raw `TokenizedInput(contentIDs:)`
/// fixtures have no prefix and echo their first id unchanged.
public func contentSeed(_ input: TokenizedInput) -> Int {
    let prefix = Array(PromptTemplate.wrap([], lead: .text).dropLast())
    let real = Array(input.ids.prefix(input.nTokens))
    let content = real.starts(with: prefix) ? Array(real.dropFirst(prefix.count)) : real
    return content.first ?? 0
}

public actor MockPredictor: EmbeddingPredictor {
    public enum OutputMode: Sendable { case normal, wrongDimension, wrongBatch, nonFinite, wrongTimings }
    public let loadGate = AsyncGate()
    public let predictGate = AsyncGate()
    public let unloadGate = AsyncGate()
    public private(set) var loadCount = 0
    public private(set) var predictCount = 0
    public private(set) var unloadCount = 0
    public private(set) var recordedFirstIDs: [Int] = []
    /// Real (unpadded) ids of every input the predictor received, in order.
    public private(set) var recordedInputIDs: [[Int]] = []
    public private(set) var isLoaded = false
    private var loadFailures = 0
    private var predictionFailure: EmbedANEError?
    private var unexpectedFailure = false
    private var mode: OutputMode = .normal
    private let clock: ManualRuntimeClock?
    public init(clock: ManualRuntimeClock? = nil) { self.clock = clock }
    public func setLoadFailures(_ count: Int) { loadFailures = count }
    public func setPredictionFailure(_ error: EmbedANEError?) { predictionFailure = error }
    public func setUnexpectedFailure(_ enabled: Bool) { unexpectedFailure = enabled }
    public func setOutputMode(_ value: OutputMode) { mode = value }
    public func load() async throws -> LoadReport {
        loadCount += 1
        await loadGate.wait()
        if loadFailures > 0 { loadFailures -= 1; throw FixtureError.deliberateFailure }
        isLoaded = true
        return LoadReport(perChunkNS: Array(repeating: 1, count: 6), residentBytes: 123,
                          computePlanChecked: false)
    }
    public func predict(_ request: PredictRequest) async throws -> PredictResult {
        predictCount += 1
        recordedFirstIDs.append(request.inputs.first.map(contentSeed) ?? 0)
        recordedInputIDs += request.inputs.map { Array($0.ids.prefix($0.nTokens)) }
        await predictGate.wait()
        if let predictionFailure { throw predictionFailure }
        if unexpectedFailure { throw FixtureError.deliberateFailure }
        let seed = request.inputs.first.map(contentSeed) ?? 0
        clock?.advance(nanoseconds: UInt64(seed + 1))
        var embeddings = request.inputs.map { input in
            [Float(contentSeed(input)), Float(input.nTokens)] + Array(repeating: Float(0), count: 2046)
        }
        switch mode {
        case .normal: break
        case .wrongDimension: embeddings[0].removeLast()
        case .wrongBatch: embeddings.removeLast()
        case .nonFinite: embeddings[0][0] = .nan
        case .wrongTimings: break
        }
        return PredictResult(embeddings: embeddings, tableLookupNS: 3,
                             perChunkNS: mode == .wrongTimings ? [1] : [1, 2, 3, 4, 5, 6])
    }
    public func unload() async throws -> UnloadReport {
        unloadCount += 1
        await unloadGate.wait()
        isLoaded = false
        return UnloadReport(residentBytes: 99)
    }
    public func releaseAll() async {
        await loadGate.open(); await predictGate.open(); await unloadGate.open()
    }
}

public func eventually(_ condition: @escaping @Sendable () async -> Bool) async throws {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: .seconds(5))
    while !(await condition()) {
        guard clock.now < deadline else { throw FixtureError.timeout }
        try await Task.sleep(for: .milliseconds(1))
    }
}

public func fixtureRequest(_ id: Int = 1) throws -> PredictRequest {
    try PredictRequest(inputs: [TokenizedInput(contentIDs: [id])])
}
