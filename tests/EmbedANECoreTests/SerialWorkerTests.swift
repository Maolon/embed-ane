import EmbedANECore
import EmbedANETestSupport
import Foundation
import Testing

private final class WorkerTrace: @unchecked Sendable {
    private let lock = NSLock()
    private var active = 0
    private var maximum = 0
    private var factories = 0
    private var mainThreadObserved = false
    func constructed() { lock.withLock { factories += 1; mainThreadObserved = mainThreadObserved || Thread.isMainThread } }
    func enter() { lock.withLock { active += 1; maximum = max(maximum, active); mainThreadObserved = mainThreadObserved || Thread.isMainThread } }
    func leave() { lock.withLock { active -= 1 } }
    var snapshot: (Int, Int, Bool) { lock.withLock { (factories, maximum, mainThreadObserved) } }
}
private final class FixtureBlockingBackend: BlockingEmbeddingBackend {
    private let trace: WorkerTrace
    init(trace: WorkerTrace) { self.trace = trace; trace.constructed() }
    func load() throws -> LoadReport {
        trace.enter(); defer { trace.leave() }
        return .init(perChunkNS: Array(repeating: 0, count: 6), residentBytes: 0, computePlanChecked: false)
    }
    func predict(_ request: PredictRequest) throws -> PredictResult {
        trace.enter(); defer { trace.leave() }
        Thread.sleep(forTimeInterval: 0.002)
        return .init(embeddings: request.inputs.map { _ in Array(repeating: 0, count: 2048) })
    }
    func unload() throws -> UnloadReport {
        trace.enter(); defer { trace.leave() }
        return .init(residentBytes: 0)
    }
}

@Suite("Dedicated serial worker")
struct SerialWorkerTests {
    @Test func factoryAndCallsStayOffMainAndNeverOverlap() async throws {
        let trace = WorkerTrace()
        let worker = SerialEmbeddingWorker { FixtureBlockingBackend(trace: trace) }
        _ = try await worker.load()
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<10 {
                group.addTask { _ = try await worker.predict(fixtureRequest()) }
            }
            try await group.waitForAll()
        }
        _ = try await worker.unload()
        let snapshot = trace.snapshot
        #expect(snapshot.0 == 1 && snapshot.1 == 1 && !snapshot.2)
    }
}
