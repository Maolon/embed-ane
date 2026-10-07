import Foundation

/// Admission/state ownership only. Blocking tokenization and CoreML work belong
/// to the dedicated workers, never this actor's executor.
public actor LifecycleActor {
    private enum Request {
        case unpreparedText(texts: [String], preparer: any EmbeddingPreparer)
        case preparedText(PredictRequest)
        case image(ImageEmbeddingRequest)

        var count: Int {
            switch self {
            case let .unpreparedText(texts, _): return texts.count
            case let .preparedText(request): return request.inputs.count
            case .image: return 1
            }
        }
        var tokenizeNS: UInt64 {
            switch self {
            case .unpreparedText: return 0
            case let .preparedText(request): return request.tokenizeNS
            case .image: return 0
            }
        }
        var isImage: Bool {
            if case .image = self { return true }
            return false
        }
    }
    private struct Entry {
        let request: Request
        let startedNS: UInt64
        let enqueuedNS: UInt64
        let continuation: CheckedContinuation<EmbeddingExecution, any Error>
    }

    private let predictor: any EmbeddingPredictor
    private let clock: any RuntimeClock
    private let readResidentBytes: @Sendable () -> UInt64
    private var state: LifecycleState = .unloaded
    private var loadTask: Task<LoadReport, any Error>?
    private var unloadTask: Task<UnloadReport, any Error>?
    private var loadReport: LoadReport?
    private var queue: [Entry] = []
    private var inFlight = 0
    private var preparing = 0
    private var maxQueueDepth: Int
    private let maxBatch: Int
    private var idleTimeoutS: Double
    private var autoLoad: Bool
    private var idleTask: Task<Void, Never>?
    private var idleGeneration: UInt64 = 0
    private var lastActivityNS: UInt64
    private var consecutiveFailures = 0
    private var retryAtNS: UInt64?
    private var lastError: String?
    private var admitted: UInt64 = 0
    private var completed: UInt64 = 0
    private var failures: UInt64 = 0
    private var window: [RequestTiming] = []

    public init(predictor: any EmbeddingPredictor, maxQueueDepth: Int = 16,
                maxBatch: Int = 8, idleTimeoutS: Double = 0,
                autoLoad: Bool = false,
                clock: any RuntimeClock = MonotonicRuntimeClock(),
                residentBytes: @escaping @Sendable () -> UInt64 = { ProcessMemory.residentBytes() }) throws {
        try Self.validateLive(idleTimeoutS: idleTimeoutS, maxQueueDepth: maxQueueDepth)
        guard (1...8).contains(maxBatch) else { throw EmbedANEError.invalidRequest("max_batch must be 1...8.", param: "max_batch") }
        self.predictor = predictor; self.maxQueueDepth = maxQueueDepth
        self.maxBatch = maxBatch; self.idleTimeoutS = idleTimeoutS
        self.autoLoad = autoLoad
        self.clock = clock; self.readResidentBytes = residentBytes; lastActivityNS = clock.nowNS()
    }

    deinit {
        idleTask?.cancel()
        loadTask?.cancel()
        unloadTask?.cancel()
    }

    public func load() async throws -> LoadReport {
        try checkCancellation()
        if let loadTask { return try await loadTask.value }
        if state == .ready, let loadReport { return loadReport }
        if state == .unloading { throw EmbedANEError.busy }
        if state == .failed, let retryAtNS, clock.nowNS() < retryAtNS {
            throw EmbedANEError.failed("Load backoff has not elapsed.")
        }
        cancelIdle()
        state = .loading
        // Unstructured by design: cancellation of a waiter cannot propagate here.
        let operation = Task<LoadReport, any Error> { [self] in
            do {
                let report = try await predictor.load()
                guard report.perChunkNS.count == 6 else {
                    throw EmbedANEError.abiMismatch(chunk: 0, reason: "Load report must contain six chunk timings.")
                }
                handleLoadSuccess(report)
                return report
            } catch {
                handleLoadFailure(error, isAuto: false)
                throw error
            }
        }
        loadTask = operation
        return try await operation.value
    }

    public func unload() async throws -> UnloadReport {
        try checkCancellation()
        guard queue.isEmpty, inFlight == 0, preparing == 0 else { throw EmbedANEError.busy }
        if let unloadTask { return try await unloadTask.value }
        if state == .loading { throw EmbedANEError.busy }
        if state == .unloaded { return UnloadReport(residentBytes: readResidentBytes()) }
        cancelIdle(); state = .unloading
        let operation = Task<UnloadReport, any Error> { [self] in
            do {
                let report = try await predictor.unload()
                state = .unloaded; loadReport = nil; unloadTask = nil
                retryAtNS = nil; lastActivityNS = clock.nowNS()
                return report
            } catch {
                unloadTask = nil; enterFailed(code: "unload_failed")
                throw error
            }
        }
        unloadTask = operation
        return try await operation.value
    }

    /// The preparation count is NOT queue admission. Unload/eviction must still
    /// respect it, so a worker cannot lose tokenizer assets during validation.
    public func embed(_ texts: [String], using preparer: any EmbeddingPreparer) async throws -> EmbeddingExecution {
        try checkCancellation()
        guard !texts.isEmpty else { throw EmbedANEError.emptyInput(index: nil) }
        guard texts.count <= maxBatch else { throw EmbedANEError.batchTooLarge(texts.count) }
        if let index = texts.firstIndex(where: \.isEmpty) { throw EmbedANEError.emptyInput(index: index) }
        guard state != .unloading else { throw EmbedANEError.busy }
        let started = clock.nowNS()
        if state == .ready {
            preparing += 1; cancelIdle()
            let request: PredictRequest
            do { request = try await preparer.prepare(texts) }
            catch {
                preparing -= 1; lastActivityNS = clock.nowNS(); armIdleIfQuiescent()
                throw error
            }
            preparing -= 1
            do {
                guard request.inputs.count == texts.count else {
                    throw EmbedANEError.abiMismatch(chunk: 0, reason: "Preparer changed the batch size.")
                }
                return try await admit(.preparedText(request), startedNS: started)
            } catch {
                lastActivityNS = clock.nowNS(); armIdleIfQuiescent()
                throw error
            }
        } else {
            return try await admit(.unpreparedText(texts: texts, preparer: preparer), startedNS: started)
        }
    }

    /// Useful for clients with an already validated batch; cancellation is only
    /// observed BEFORE admission. Accepted work is never cancelled by its waiter.
    public func submit(_ request: PredictRequest) async throws -> EmbeddingExecution {
        try await admit(.preparedText(request), startedNS: clock.nowNS())
    }

    public var supportsImages: Bool { predictor is any MultimodalEmbeddingPredictor }

    public func submitImage(_ request: ImageEmbeddingRequest) async throws -> EmbeddingExecution {
        guard supportsImages else { throw EmbedANEError.visionNotConfigured }
        return try await admit(.image(request), startedNS: clock.nowNS())
    }

    private func admit(_ request: Request, startedNS: UInt64) async throws -> EmbeddingExecution {
        try checkCancellation()
        guard request.count <= maxBatch else { throw EmbedANEError.batchTooLarge(request.count) }
        switch state {
        case .unloaded:
            guard autoLoad else { throw EmbedANEError.modelNotLoaded }
            startAutoLoad()
        case .loading:
            guard autoLoad else { throw EmbedANEError.loading }
        case .unloading:
            throw EmbedANEError.busy
        case .failed:
            throw EmbedANEError.failed("Explicit load is required after backoff.")
        case .ready:
            break
        }
        guard queue.count < maxQueueDepth else { throw EmbedANEError.overloaded }
        cancelIdle()
        return try await withCheckedThrowingContinuation { continuation in
            admitted &+= 1
            queue.append(Entry(request: request, startedNS: startedNS,
                               enqueuedNS: clock.nowNS(), continuation: continuation))
            startNext()
        }
    }

    private func startAutoLoad() {
        guard loadTask == nil else { return }
        cancelIdle()
        state = .loading
        let operation = Task<LoadReport, any Error> { [self] in
            do {
                let report = try await predictor.load()
                guard report.perChunkNS.count == 6 else {
                    throw EmbedANEError.abiMismatch(chunk: 0, reason: "Load report must contain six chunk timings.")
                }
                handleLoadSuccess(report)
                return report
            } catch {
                handleLoadFailure(error, isAuto: true)
                throw error
            }
        }
        loadTask = operation
    }

    private func handleLoadSuccess(_ report: LoadReport) {
        state = .ready
        loadReport = report
        loadTask = nil
        consecutiveFailures = 0
        retryAtNS = nil
        lastError = nil
        lastActivityNS = clock.nowNS()
        startNext()
    }

    private func handleLoadFailure(_ error: any Error, isAuto: Bool) {
        loadTask = nil
        loadReport = nil
        if isAuto {
            state = .unloaded
            retryAtNS = nil
            consecutiveFailures = 0
            lastError = (error as? EmbedANEError)?.code ?? "load_failed"
        } else {
            enterFailed(code: (error as? EmbedANEError)?.code ?? "load_failed")
        }
        let rejected = queue
        queue.removeAll(keepingCapacity: true)
        for pending in rejected {
            record(makeTiming(pending, predictionStarted: clock.nowNS(), result: nil), failed: true)
            pending.continuation.resume(throwing: EmbedANEError.failed("Model failed to load."))
        }
    }

    private func startNext() {
        guard state == .ready, inFlight == 0, !queue.isEmpty else {
            armIdleIfQuiescent(); return
        }
        let entry = queue.removeFirst()
        inFlight = 1
        let started = clock.nowNS()
        Task { [self] in
            var prepareFailed = false
            do {
                let result: PredictResult
                let promptTokens: Int
                let tokenizeNS: UInt64
                switch entry.request {
                case let .preparedText(request):
                    result = try await predictor.predict(request)
                    try result.validate(for: request)
                    promptTokens = request.promptTokens; tokenizeNS = request.tokenizeNS
                case let .unpreparedText(texts, preparer):
                    preparing += 1
                    let request: PredictRequest
                    do {
                        request = try await preparer.prepare(texts)
                        guard request.inputs.count == texts.count else {
                            throw EmbedANEError.abiMismatch(chunk: 0, reason: "Preparer changed the batch size.")
                        }
                        preparing -= 1
                    } catch {
                        preparing -= 1
                        prepareFailed = true
                        throw error
                    }
                    result = try await predictor.predict(request)
                    try result.validate(for: request)
                    promptTokens = request.promptTokens
                    tokenizeNS = request.tokenizeNS
                case let .image(request):
                    guard let vision = predictor as? any MultimodalEmbeddingPredictor else { throw EmbedANEError.visionNotConfigured }
                    let image = try await vision.predictImage(request)
                    try image.validate()
                    result = image.prediction; promptTokens = image.promptTokens; tokenizeNS = image.tokenizeNS
                }
                let timing = makeTiming(entry, predictionStarted: started, result: result, tokenizeNS: tokenizeNS)
                record(timing, failed: false)
                inFlight = 0; lastActivityNS = clock.nowNS()
                entry.continuation.resume(returning: EmbeddingExecution(
                    embeddings: result.embeddings, promptTokens: promptTokens, timing: timing))
                startNext()
            } catch {
                record(makeTiming(entry, predictionStarted: started, result: nil), failed: true)
                inFlight = 0; lastActivityNS = clock.nowNS()
                entry.continuation.resume(throwing: error)
                // Prepare failure or bad image input must not kill the model or drain peers.
                if prepareFailed || (entry.request.isImage && (error as? EmbedANEError)?.httpStatus == 400) {
                    startNext()
                    return
                }
                enterFailed(code: (error as? EmbedANEError)?.code ?? "prediction_failed")
                // Accepted waiters must be completed even after the worker fails.
                let rejected = queue; queue.removeAll(keepingCapacity: true)
                for pending in rejected {
                    record(makeTiming(pending, predictionStarted: clock.nowNS(), result: nil), failed: true)
                    pending.continuation.resume(throwing: EmbedANEError.failed("Prediction worker failed."))
                }
            }
        }
    }

    public func updateLiveSettings(idleTimeoutS: Double, maxQueueDepth: Int, autoLoad: Bool? = nil) throws {
        try Self.validateLive(idleTimeoutS: idleTimeoutS, maxQueueDepth: maxQueueDepth)
        self.idleTimeoutS = idleTimeoutS; self.maxQueueDepth = maxQueueDepth
        if let autoLoad { self.autoLoad = autoLoad }
        cancelIdle(); armIdleIfQuiescent()
    }

    public func snapshot() -> RuntimeSnapshot {
        RuntimeSnapshot(state: state, queueDepth: queue.count, inFlight: inFlight,
                        preparing: preparing, residentBytes: readResidentBytes(), retryAtNS: retryAtNS)
    }

    public func statistics() -> RuntimeStatistics {
        let total = TimingPercentiles(samples: window.map(\.totalNS))
        return RuntimeStatistics(
            runtime: snapshot(), admittedRequests: admitted, completedRequests: completed,
            failedRequests: failures, windowCount: window.count, p50NS: total.p50NS, p95NS: total.p95NS,
            queueWait: TimingPercentiles(samples: window.map(\.queueWaitNS)),
            tokenization: TimingPercentiles(samples: window.map(\.tokenizeNS)),
            tableLookup: TimingPercentiles(samples: window.map(\.tableLookupNS)),
            perChunk: (0..<6).map { index in TimingPercentiles(samples: window.map { $0.perChunkNS[index] }) },
            residentBytesScope: "process", lastError: lastError)
    }

    private func checkCancellation() throws {
        if Task.isCancelled { throw EmbedANEError.cancelled }
    }
    private static func validateLive(idleTimeoutS: Double, maxQueueDepth: Int) throws {
        // Bound conversion to UInt64 nanoseconds without trapping on untrusted JSON/YAML.
        guard idleTimeoutS.isFinite, idleTimeoutS >= 0, idleTimeoutS <= 1_000_000_000 else {
            throw EmbedANEError.invalidRequest("idle_timeout_s must be finite and in 0...1000000000.", param: "idle_timeout_s")
        }
        guard maxQueueDepth > 0 else { throw EmbedANEError.invalidRequest("max_queue_depth must be positive.", param: "max_queue_depth") }
    }
    private func enterFailed(code: String) {
        cancelIdle(); state = .failed; loadReport = nil; lastError = code
        consecutiveFailures = min(consecutiveFailures + 1, 7)
        let seconds = min(UInt64(1) << (consecutiveFailures - 1), 60)
        retryAtNS = addingNS(clock.nowNS(), seconds * 1_000_000_000)
    }
    private func makeTiming(_ entry: Entry, predictionStarted: UInt64, result: PredictResult?, tokenizeNS: UInt64? = nil) -> RequestTiming {
        RequestTiming(queueWaitNS: elapsedNS(predictionStarted, since: entry.enqueuedNS),
                      tokenizeNS: tokenizeNS ?? entry.request.tokenizeNS, tableLookupNS: result?.tableLookupNS ?? 0,
                      perChunkNS: result?.perChunkNS ?? Array(repeating: 0, count: 6),
                      totalNS: elapsedNS(clock.nowNS(), since: entry.startedNS))
    }
    private func record(_ timing: RequestTiming, failed: Bool) {
        completed &+= 1
        if failed { failures &+= 1 }
        if window.count == 256 { window.removeFirst() }
        window.append(timing)
    }
    private func cancelIdle() {
        idleGeneration &+= 1; idleTask?.cancel(); idleTask = nil
    }
    private func armIdleIfQuiescent() {
        guard state == .ready, queue.isEmpty, inFlight == 0, preparing == 0,
              idleTimeoutS > 0, idleTask == nil else { return }
        let deadline = addingNS(lastActivityNS, UInt64(idleTimeoutS * 1_000_000_000))
        let generation = idleGeneration
        idleTask = Task { [weak self, clock] in
            do { try await clock.sleep(untilNS: deadline) }
            catch { return }
            await self?.idleFired(generation: generation, deadline: deadline)
        }
    }
    private func idleFired(generation: UInt64, deadline: UInt64) async {
        guard generation == idleGeneration, clock.nowNS() >= deadline,
              state == .ready, queue.isEmpty, inFlight == 0, preparing == 0 else { return }
        idleTask = nil
        do { _ = try await unload() }
        catch { /* unload already records a typed failed/backoff state */ }
    }
}
