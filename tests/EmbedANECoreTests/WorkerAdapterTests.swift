import Dispatch
import EmbedANETestSupport
import Foundation
import Testing
@testable import EmbedANECore

private final class WorkerRecorder: @unchecked Sendable {
    let lock = NSLock()
    var events: [String] = []
    var active = 0
    var maximumActive = 0
    func operation(_ name: String) {
        lock.withLock { events.append(name); active += 1; maximumActive = max(maximumActive, active) }
        Thread.sleep(forTimeInterval: 0.001)
        lock.withLock { active -= 1 }
    }
    func snapshot() -> ([String], Int) { lock.withLock { (events, maximumActive) } }
}
private final class PreparedBlockingBackend: BlockingEmbeddingBackend, BlockingEmbeddingPreparer {
    let recorder: WorkerRecorder
    init(_ recorder: WorkerRecorder) { self.recorder = recorder; recorder.operation("factory") }
    deinit { recorder.operation("deinit") }
    func load() throws -> LoadReport {
        recorder.operation("load")
        return .init(perChunkNS: Array(repeating: 1, count: 6), residentBytes: 0, computePlanChecked: false)
    }
    func prepare(_ texts: [String]) throws -> PredictRequest {
        recorder.operation("prepare")
        return try PredictRequest(inputs: TokenizedInput.prepare(texts, tokenizer: FixtureTokenizer()))
    }
    func predict(_ request: PredictRequest) throws -> PredictResult {
        recorder.operation("predict")
        return .init(embeddings: request.inputs.map { [Float(contentSeed($0))] + Array(repeating: 0, count: 2047) })
    }
    func unload() throws -> UnloadReport { recorder.operation("unload"); return .init(residentBytes: 0) }
}
private final class ExecutorProbe: @unchecked Sendable {
    let key = DispatchSpecificKey<String>()
}

struct WorkerAdapterTests {
    @Test func sharedWorkerPreparesAndPredictsThroughLifecycle() async throws {
        let recorder = WorkerRecorder()
        let worker = SerialEmbeddingWorker { PreparedBlockingBackend(recorder) }
        let lifecycle = try LifecycleActor(predictor: worker)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<10 { group.addTask { _ = try await lifecycle.load() } }
            try await group.waitForAll()
        }
        let result = try await lifecycle.embed(["a", "b"], using: worker)
        #expect(result.embeddings.map { $0[0] } == [97, 98])
        #expect(result.promptTokens == 12) // 2 × (1 byte + 4 chat-template ids + <embedding>)
        _ = try await lifecycle.unload()
        #expect(recorder.snapshot().0 == ["factory", "load", "prepare", "predict", "unload"])
        #expect(recorder.snapshot().1 == 1)
    }
    @Test func invalidWholeBatchNeverReachesPredictor() async throws {
        let recorder = WorkerRecorder(); let worker = SerialEmbeddingWorker { PreparedBlockingBackend(recorder) }
        let lifecycle = try LifecycleActor(predictor: worker); _ = try await lifecycle.load()
        await #expect(throws: EmbedANEError.inputTooLong(index: 1, contentTokens: 512)) {
            try await lifecycle.embed(["ok", String(repeating: "a", count: 512)], using: worker)
        }
        #expect(!recorder.snapshot().0.contains("predict"))
        #expect(await lifecycle.statistics().admittedRequests == 0)
    }
    @Test func concurrentWorkerCallsRemainSerialAndFactoryRunsOnce() async throws {
        let recorder = WorkerRecorder(); let worker = SerialEmbeddingWorker { PreparedBlockingBackend(recorder) }
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<20 { group.addTask { _ = try await worker.prepare(["a"]) } }
            try await group.waitForAll()
        }
        #expect(recorder.snapshot().0.filter { $0 == "factory" }.count == 1)
        #expect(recorder.snapshot().0.filter { $0 == "prepare" }.count == 20)
        #expect(recorder.snapshot().1 == 1)
    }
    @Test func asyncMetadataExecutorResumesOnItsDedicatedQueue() async throws {
        let queue = DispatchQueue(label: "s4.executor.test")
        let probe = ExecutorProbe(); queue.setSpecific(key: probe.key, value: "worker")
        let executor = QueueTaskExecutor(queue: queue)
        let result = try await Task(executorPreference: executor) {
            let before = DispatchQueue.getSpecific(key: probe.key)
            try await Task.sleep(for: .milliseconds(1))
            let after = DispatchQueue.getSpecific(key: probe.key)
            return before == "worker" && after == "worker" && !Thread.isMainThread
        }.value
        #expect(result)
    }
    @Test func coreMLConfigurationRejectsNonFrozenComputeUnitsWithoutLoading() throws {
        for units in [ComputeUnitsSetting.cpuOnly, .cpuAndGPU, .all] {
            let config = ServiceConfiguration(modelRoot: "/tmp/unused", computeUnits: units)
            #expect(throws: EmbedANEError.self) { try CoreMLRuntime(configuration: config, acquireLease: { () }) }
        }
    }
}

private final class LeaseProbe: @unchecked Sendable {
    let recorder: WorkerRecorder
    init(_ recorder: WorkerRecorder) { self.recorder = recorder; recorder.operation("lease_acquired") }
    deinit { recorder.operation("lease_released") }
}

struct CoreMLActivationBoundaryTests {
    @Test func failedVerificationReleasesLeaseBeforeAnyModelConstruction() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let bundle = root.appendingPathComponent(ModelABI.defaultModelID)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        try Data("manifest_version: 999\n".utf8).write(to: bundle.appendingPathComponent("manifest.yaml"))
        let recorder = WorkerRecorder()
        let runtime = try CoreMLRuntime(configuration: .init(modelRoot: root.path), acquireLease: { LeaseProbe(recorder) })
        // Verification fails BEFORE tokenizer/table/model initialization. This
        // does not load a CoreML graph on CI or require any real model assets.
        for _ in 0..<2 {
            await #expect(throws: (any Error).self) { try await runtime.load() }
        }
        #expect(recorder.snapshot().0 == ["lease_acquired", "lease_released", "lease_acquired", "lease_released"])
        #expect(runtime.verificationReport() == nil)
    }
}
