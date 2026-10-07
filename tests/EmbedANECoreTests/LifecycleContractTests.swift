import EmbedANECore
import EmbedANETestSupport
import Testing

@Suite("Lifecycle contracts")
struct LifecycleContractTests {
    @Test func tenLoadsShareOneOperationDespiteCancelledWaiter() async throws {
        let predictor = MockPredictor()
        await predictor.loadGate.close()
        let runtime = try LifecycleActor(predictor: predictor)
        let first = Task { try await runtime.load() }
        try await eventually { await predictor.loadCount == 1 }
        let others = (0..<9).map { _ in Task { try await runtime.load() } }
        first.cancel()
        await predictor.loadGate.open()
        _ = try await first.value
        for task in others { _ = try await task.value }
        #expect(await predictor.loadCount == 1)
        #expect(await runtime.snapshot().state == .ready)
    }

    @Test func boundedFIFOAndAcceptedCancellation() async throws {
        let predictor = MockPredictor()
        await predictor.predictGate.close()
        let runtime = try LifecycleActor(predictor: predictor, maxQueueDepth: 2)
        _ = try await runtime.load()
        let first = Task { try await runtime.submit(fixtureRequest(1)) }
        try await eventually { await runtime.snapshot().inFlight == 1 }
        let second = Task { try await runtime.submit(fixtureRequest(2)) }
        try await eventually { await runtime.snapshot().queueDepth == 1 }
        let third = Task { try await runtime.submit(fixtureRequest(3)) }
        try await eventually { await runtime.snapshot().queueDepth == 2 }
        await #expect(throws: EmbedANEError.overloaded) { try await runtime.submit(fixtureRequest(4)) }
        await #expect(throws: EmbedANEError.busy) { try await runtime.unload() }
        second.cancel()
        await predictor.predictGate.open()
        _ = try await first.value; _ = try await second.value; _ = try await third.value
        #expect(await predictor.recordedFirstIDs == [1, 2, 3])
        #expect(await runtime.statistics().admittedRequests == 3)
        #expect(await runtime.snapshot().queueDepth == 0)
    }

    @Test func unreadyAndLoadingAreDistinct() async throws {
        let predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor)
        await #expect(throws: EmbedANEError.modelNotLoaded) { try await runtime.submit(fixtureRequest()) }
        await predictor.loadGate.close()
        let loading = Task { try await runtime.load() }
        try await eventually { await runtime.snapshot().state == .loading }
        await #expect(throws: EmbedANEError.loading) { try await runtime.submit(fixtureRequest()) }
        await #expect(throws: EmbedANEError.busy) { try await runtime.unload() }
        await predictor.loadGate.open(); _ = try await loading.value
    }

    @Test func defaultNeverEvicts() async throws {
        let clock = ManualRuntimeClock(), predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor, clock: clock)
        _ = try await runtime.load()
        #expect(clock.pendingSleeps == 0)
        clock.advance(seconds: 10_000)
        #expect(await runtime.snapshot().state == .ready)
        #expect(await predictor.unloadCount == 0)
    }

    @Test func evictionStartsOnlyAfterQueueAndFlightDrain() async throws {
        let clock = ManualRuntimeClock(), predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor, idleTimeoutS: 5, clock: clock)
        _ = try await runtime.load()
        try await eventually { clock.pendingSleeps == 1 }
        await predictor.predictGate.close()
        let first = Task { try await runtime.submit(fixtureRequest(1)) }
        try await eventually { await runtime.snapshot().inFlight == 1 }
        let second = Task { try await runtime.submit(fixtureRequest(2)) }
        try await eventually { await runtime.snapshot().queueDepth == 1 }
        try await eventually { clock.pendingSleeps == 0 }
        clock.advance(seconds: 100)
        #expect(await predictor.unloadCount == 0)
        await predictor.predictGate.open()
        _ = try await first.value; _ = try await second.value
        try await eventually { clock.pendingSleeps == 1 }
        clock.advance(seconds: 4)
        #expect(await runtime.snapshot().state == .ready)
        clock.advance(seconds: 1)
        try await eventually { await runtime.snapshot().state == .unloaded }
        #expect(await predictor.unloadCount == 1)
    }

    @Test func preparationPreventsUnloadWithoutBeingAdmitted() async throws {
        let clock = ManualRuntimeClock(), predictor = MockPredictor(), preparer = CountingPreparer()
        let runtime = try LifecycleActor(predictor: predictor, idleTimeoutS: 2, clock: clock)
        _ = try await runtime.load()
        await preparer.gate.close()
        let request = Task { try await runtime.embed(["a"], using: preparer) }
        try await eventually { await runtime.snapshot().preparing == 1 }
        #expect(await runtime.statistics().admittedRequests == 0)
        clock.advance(seconds: 20)
        await #expect(throws: EmbedANEError.busy) { try await runtime.unload() }
        #expect(await predictor.unloadCount == 0)
        await preparer.gate.open(); _ = try await request.value
        try await eventually { clock.pendingSleeps == 1 }
        clock.advance(seconds: 2)
        try await eventually { await runtime.snapshot().state == .unloaded }
    }

    @Test func liveDisableCancelsEviction() async throws {
        let clock = ManualRuntimeClock(), predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor, idleTimeoutS: 1, clock: clock)
        _ = try await runtime.load()
        try await eventually { clock.pendingSleeps == 1 }
        try await runtime.updateLiveSettings(idleTimeoutS: 0, maxQueueDepth: 16)
        try await eventually { clock.pendingSleeps == 0 }
        clock.advance(seconds: 100)
        #expect(await runtime.snapshot().state == .ready)
    }

    @Test func loweringQueueDepthDoesNotDropAcceptedWork() async throws {
        let predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor, maxQueueDepth: 2)
        _ = try await runtime.load(); await predictor.predictGate.close()
        let first = Task { try await runtime.submit(fixtureRequest(1)) }
        try await eventually { await runtime.snapshot().inFlight == 1 }
        let second = Task { try await runtime.submit(fixtureRequest(2)) }
        try await eventually { await runtime.snapshot().queueDepth == 1 }
        let third = Task { try await runtime.submit(fixtureRequest(3)) }
        try await eventually { await runtime.snapshot().queueDepth == 2 }
        try await runtime.updateLiveSettings(idleTimeoutS: 0, maxQueueDepth: 1)
        await #expect(throws: EmbedANEError.overloaded) { try await runtime.submit(fixtureRequest(4)) }
        await predictor.predictGate.open()
        _ = try await first.value; _ = try await second.value; _ = try await third.value
        #expect(await predictor.recordedFirstIDs == [1, 2, 3])
    }

    @Test func backoffIsExponentialCappedAndResetsOnSuccess() async throws {
        let clock = ManualRuntimeClock(), predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor, clock: clock)
        await predictor.setLoadFailures(8)
        for delay: UInt64 in [1, 2, 4, 8, 16, 32, 60, 60] {
            await #expect(throws: FixtureError.self) { try await runtime.load() }
            #expect(await runtime.snapshot().retryAtNS == clock.nowNS() + delay * 1_000_000_000)
            let count = await predictor.loadCount
            await #expect(throws: EmbedANEError.failed("Load backoff has not elapsed.")) { try await runtime.load() }
            #expect(await predictor.loadCount == count)
            clock.advance(seconds: delay)
        }
        _ = try await runtime.load()
        #expect(await runtime.snapshot().retryAtNS == nil)
        _ = try await runtime.unload()
        await predictor.setLoadFailures(1)
        await #expect(throws: FixtureError.self) { try await runtime.load() }
        #expect(await runtime.snapshot().retryAtNS == clock.nowNS() + 1_000_000_000)
    }

    @Test func failedPredictionCompletesEveryAcceptedWaiter() async throws {
        let predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor)
        _ = try await runtime.load(); await predictor.predictGate.close()
        let first = Task { try await runtime.submit(fixtureRequest()) }
        try await eventually { await runtime.snapshot().inFlight == 1 }
        let queued = Task { try await runtime.submit(fixtureRequest()) }
        try await eventually { await runtime.snapshot().queueDepth == 1 }
        await predictor.setUnexpectedFailure(true); await predictor.predictGate.open()
        await #expect(throws: FixtureError.self) { try await first.value }
        await #expect(throws: EmbedANEError.failed("Prediction worker failed.")) { try await queued.value }
        let stats = await runtime.statistics()
        #expect(stats.runtime.state == .failed)
        #expect(stats.runtime.queueDepth == 0)
        #expect(stats.completedRequests == 2 && stats.failedRequests == 2)
    }

    @Test func nonconformingPredictorNeverProducesSuccessfulExecution() async throws {
        for mode in [MockPredictor.OutputMode.wrongBatch, .wrongDimension, .nonFinite, .wrongTimings] {
            let predictor = MockPredictor()
            let runtime = try LifecycleActor(predictor: predictor)
            _ = try await runtime.load(); await predictor.setOutputMode(mode)
            await #expect(throws: EmbedANEError.self) { try await runtime.submit(fixtureRequest()) }
            #expect(await runtime.snapshot().state == .failed)
        }
    }

    @Test func cancellationBeforeAdmissionDoesNotPredict() async throws {
        let predictor = MockPredictor(), gate = AsyncGate(open: false)
        let runtime = try LifecycleActor(predictor: predictor)
        _ = try await runtime.load()
        let request = Task { await gate.wait(); return try await runtime.submit(fixtureRequest()) }
        request.cancel(); await gate.open()
        await #expect(throws: EmbedANEError.cancelled) { try await request.value }
        #expect(await predictor.predictCount == 0)
    }

    @Test func rollingWindowIsBoundedAndUsesNearestRank() async throws {
        let clock = ManualRuntimeClock()
        let timedPredictor = MockPredictor(clock: clock)
        let runtime = try LifecycleActor(predictor: timedPredictor, clock: clock, residentBytes: { 12345 })
        _ = try await runtime.load()
        for index in 0..<300 { _ = try await runtime.submit(fixtureRequest(index)) }
        let stats = await runtime.statistics()
        #expect(stats.completedRequests == 300 && stats.windowCount == 256)
        #expect(stats.p50NS == 172 && stats.p95NS == 288)
        #expect(stats.runtime.residentBytes == 12345 && stats.residentBytesScope == "process")
        #expect(stats.perChunk.count == 6)
        #expect(TimingPercentiles(samples: []).p50NS == 0)
        #expect(TimingPercentiles(samples: [1, 100]).p50NS == 1)
        #expect(TimingPercentiles(samples: [1, 100]).p95NS == 100)
    }

    @Test func wholeBatchTokenizationFailureNeverAdmitsPrefix() async throws {
        let predictor = MockPredictor(), preparer = CountingPreparer()
        let runtime = try LifecycleActor(predictor: predictor)
        _ = try await runtime.load()
        await #expect(throws: EmbedANEError.inputTooLong(index: 1, contentTokens: 512)) {
            try await runtime.embed(["a", String(repeating: "b", count: 512)], using: preparer)
        }
        #expect(await runtime.statistics().admittedRequests == 0)
        #expect(await predictor.predictCount == 0)
        #expect(await runtime.snapshot().preparing == 0)
    }

    @Test func concurrentUnloadsAreSingleFlight() async throws {
        let predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor)
        _ = try await runtime.load(); await predictor.unloadGate.close()
        let first = Task { try await runtime.unload() }
        try await eventually { await predictor.unloadCount == 1 }
        let second = Task { try await runtime.unload() }
        await predictor.unloadGate.open()
        _ = try await first.value; _ = try await second.value
        #expect(await predictor.unloadCount == 1)
        #expect(await runtime.snapshot().state == .unloaded)
    }

    @Test func unloadedRequestWithAutoLoadTriggersLoadAndServes() async throws {
        let predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor, autoLoad: true)
        #expect(await runtime.snapshot().state == .unloaded)
        let exec = try await runtime.submit(fixtureRequest())
        #expect(exec.embeddings.count == 1)
        #expect(await runtime.snapshot().state == .ready)
        #expect(await predictor.loadCount == 1)
    }

    @Test func concurrentRequestsDuringAutoLoadSingleLoadAndAllServed() async throws {
        let predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor, maxQueueDepth: 4, autoLoad: true)
        await predictor.loadGate.close()
        let t1 = Task { try await runtime.submit(fixtureRequest(1)) }
        try await eventually { await runtime.snapshot().state == .loading }
        let t2 = Task { try await runtime.submit(fixtureRequest(2)) }
        let t3 = Task { try await runtime.submit(fixtureRequest(3)) }
        try await eventually { await runtime.snapshot().queueDepth == 3 }
        await predictor.loadGate.open()
        let r1 = try await t1.value
        let r2 = try await t2.value
        let r3 = try await t3.value
        #expect(r1.embeddings.count == 1)
        #expect(r2.embeddings.count == 1)
        #expect(r3.embeddings.count == 1)
        #expect(await predictor.loadCount == 1)
        #expect(await runtime.snapshot().state == .ready)
    }

    @Test func autoLoadQueueLimitEnforcedDuringLoad() async throws {
        let predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor, maxQueueDepth: 2, autoLoad: true)
        await predictor.loadGate.close()
        let t1 = Task { try await runtime.submit(fixtureRequest(1)) }
        try await eventually { await runtime.snapshot().state == .loading }
        let t2 = Task { try await runtime.submit(fixtureRequest(2)) }
        try await eventually { await runtime.snapshot().queueDepth == 2 }
        await #expect(throws: EmbedANEError.overloaded) { try await runtime.submit(fixtureRequest(3)) }
        await predictor.loadGate.open()
        let r1 = try await t1.value
        let r2 = try await t2.value
        #expect(r1.embeddings.count == 1)
        #expect(r2.embeddings.count == 1)
    }

    @Test func autoLoadFailureRevertsToUnloadedAndAwaitingRequestsFailAndRetrySucceeds() async throws {
        let predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor, autoLoad: true)
        await predictor.setLoadFailures(1)
        await predictor.loadGate.close()
        let t1 = Task { try await runtime.submit(fixtureRequest(1)) }
        try await eventually { await runtime.snapshot().state == .loading }
        let t2 = Task { try await runtime.submit(fixtureRequest(2)) }
        try await eventually { await runtime.snapshot().queueDepth == 2 }
        await predictor.loadGate.open()
        await #expect(throws: EmbedANEError.failed("Model failed to load.")) { try await t1.value }
        await #expect(throws: EmbedANEError.failed("Model failed to load.")) { try await t2.value }
        #expect(await runtime.snapshot().state == .unloaded)
        #expect(await runtime.snapshot().queueDepth == 0)
        let retry = try await runtime.submit(fixtureRequest(3))
        #expect(retry.embeddings.count == 1)
        #expect(await runtime.snapshot().state == .ready)
        #expect(await predictor.loadCount == 2)
    }

    @Test func idleEvictionStillTriggersAfterHotLoadCycle() async throws {
        let clock = ManualRuntimeClock(), predictor = MockPredictor()
        let runtime = try LifecycleActor(predictor: predictor, idleTimeoutS: 5, autoLoad: true, clock: clock)
        #expect(await runtime.snapshot().state == .unloaded)
        let exec = try await runtime.submit(fixtureRequest())
        #expect(exec.embeddings.count == 1)
        #expect(await runtime.snapshot().state == .ready)
        try await eventually { clock.pendingSleeps == 1 }
        clock.advance(seconds: 5)
        try await eventually { await runtime.snapshot().state == .unloaded }
        #expect(await predictor.unloadCount == 1)
    }

    @Test func unloadedTextRequestWithAutoLoadTriggersLoadAndServes() async throws {
        let predictor = MockPredictor()
        let preparer = ProductionFaithfulPreparer(predictor: predictor)
        let runtime = try LifecycleActor(predictor: predictor, autoLoad: true)
        #expect(await runtime.snapshot().state == .unloaded)
        let exec = try await runtime.embed(["hello"], using: preparer)
        #expect(exec.embeddings.count == 1)
        #expect(await runtime.snapshot().state == .ready)
        #expect(await predictor.loadCount == 1)
        #expect(await predictor.predictCount == 1)
        #expect(await preparer.calls == 1)
    }

    @Test func concurrentTextRequestsDuringAutoLoadSingleLoadAndAllServed() async throws {
        let predictor = MockPredictor()
        let preparer = ProductionFaithfulPreparer(predictor: predictor)
        let runtime = try LifecycleActor(predictor: predictor, maxQueueDepth: 4, autoLoad: true)
        await predictor.loadGate.close()
        let t1 = Task { try await runtime.embed(["first"], using: preparer) }
        try await eventually { await runtime.snapshot().state == .loading }
        let t2 = Task { try await runtime.embed(["second"], using: preparer) }
        let t3 = Task { try await runtime.embed(["third"], using: preparer) }
        try await eventually { await runtime.snapshot().queueDepth == 3 }
        await predictor.loadGate.open()
        let r1 = try await t1.value
        let r2 = try await t2.value
        let r3 = try await t3.value
        #expect(r1.embeddings.count == 1)
        #expect(r2.embeddings.count == 1)
        #expect(r3.embeddings.count == 1)
        #expect(await predictor.loadCount == 1)
        #expect(await predictor.predictCount == 3)
        #expect(await runtime.snapshot().state == .ready)
        #expect(await runtime.snapshot().queueDepth == 0)
    }

    @Test func workerPrepareFailureResumesEntryAndKeepsModelReadyForPeers() async throws {
        let predictor = MockPredictor()
        let preparer = ProductionFaithfulPreparer(predictor: predictor)
        let runtime = try LifecycleActor(predictor: predictor, maxQueueDepth: 4, autoLoad: true)
        await predictor.loadGate.close()
        let badText = String(repeating: "b", count: 512)
        let t1 = Task { try await runtime.embed([badText], using: preparer) }
        try await eventually { await runtime.snapshot().state == .loading }
        let t2 = Task { try await runtime.embed(["good text"], using: preparer) }
        try await eventually { await runtime.snapshot().queueDepth == 2 }
        await predictor.loadGate.open()
        await #expect(throws: EmbedANEError.inputTooLong(index: 0, contentTokens: 512)) {
            try await t1.value
        }
        let r2 = try await t2.value
        #expect(r2.embeddings.count == 1)
        #expect(await runtime.snapshot().state == .ready)
        #expect(await predictor.loadCount == 1)
        #expect(await predictor.predictCount == 1)
    }

    @Test func unloadedTextRequestWithoutAutoLoadRejectsWithModelNotLoaded() async throws {
        let predictor = MockPredictor()
        let preparer = ProductionFaithfulPreparer(predictor: predictor)
        let runtime = try LifecycleActor(predictor: predictor, autoLoad: false)
        #expect(await runtime.snapshot().state == .unloaded)
        await #expect(throws: EmbedANEError.modelNotLoaded) {
            try await runtime.embed(["hello"], using: preparer)
        }
        #expect(await predictor.loadCount == 0)
        #expect(await runtime.snapshot().state == .unloaded)
    }
}
