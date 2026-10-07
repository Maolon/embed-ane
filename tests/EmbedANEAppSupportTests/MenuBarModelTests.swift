import EmbedANEAppSupport
import EmbedANECore
import EmbedANETestSupport
import Foundation
import Testing

@Suite("App presentation and lifecycle composition", .serialized)
@MainActor struct MenuBarModelTests {
    @Test func startupIsSingleInstanceAndNeverDownloads() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        f.model.start(); f.model.start()
        try await f.ready()
        f.model.start()
        #expect(f.factory.count == 1)
        #expect(await f.session.runCount == 1)
        #expect(await f.session.predictor.loadCount == 1)
        #expect(await f.library.installs == 0)
        #expect(f.model.presentation.kind == .ready)
        try await f.model.stopWhenIdle()
        #expect(f.model.phase == .stopped)
    }
    @Test func catalogBeforeFirstSnapshotDoesNotConsumeInitialLoad() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        await f.session.snapshotGate.close()
        f.model.start()
        try await eventually { @MainActor in f.model.phase == .running && !f.model.models.isEmpty && !f.model.scanning }
        #expect(f.model.snapshot == nil)
        #expect(await f.session.predictor.loadCount == 0)
        await f.session.snapshotGate.open()
        try await eventually { @MainActor in f.model.snapshot?.state == .ready && !f.model.commanding }
        #expect(await f.session.predictor.loadCount == 1)
    }
    @Test func missingInstallLeavesAUsableUnloadedServer() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        await f.library.setCatalog(.init())
        f.model.start()
        try await eventually { @MainActor in f.model.phase == .running && f.model.snapshot != nil && !f.model.scanning }
        #expect(f.model.snapshot?.state == .unloaded)
        #expect(!f.model.canLoad)
        #expect(await f.session.predictor.loadCount == 0)
        #expect(await f.library.installs == 0)
        #expect(!(await f.model.workAtQuit()))
    }
    @Test func bindFailureDoesNotAutoLoadOrIncrementPort() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        await f.session.setRunFailure(true)
        f.model.start()
        try await eventually { @MainActor in f.model.phase == .failed }
        #expect(f.model.listeningPort == nil)
        #expect(f.model.configuration?.port == 8080)
        #expect(f.model.notice?.code == "io_error")
        #expect(f.model.presentation.kind == .error)
        #expect(f.model.presentation.action == .restartServer)
        #expect(await f.session.predictor.loadCount == 0)
    }
    @Test func loadingStateDisablesDuplicateLoadAndUnload() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        await f.session.predictor.loadGate.close()
        f.model.start()
        try await eventually { await f.session.predictor.loadCount == 1 }
        await f.model.refresh()
        #expect(f.model.presentation.kind == .working)
        #expect(!f.model.canLoad && !f.model.canUnload)
        #expect(await f.model.workAtQuit())
        await f.model.load()
        #expect(await f.session.predictor.loadCount == 1)
        await f.session.predictor.loadGate.open()
        try await eventually { @MainActor in !f.model.commanding }
    }
    @Test func inFlightUnloadIsRefusedWithoutCancellingPrediction() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.session.predictor.predictGate.close()
        let work = Task { try await f.session.lifecycle.submit(fixtureRequest(19)) }
        try await eventually { await f.session.predictor.predictCount == 1 }
        await f.model.refresh()
        #expect(!f.model.canUnload)
        await f.model.unload()
        #expect(f.model.notice?.code == "busy")
        #expect(await f.session.predictor.unloadCount == 0)
        await f.session.predictor.predictGate.open()
        #expect(try await work.value.embeddings[0][0] == 19)
        await f.model.unload()
        #expect(f.model.snapshot?.state == .unloaded)
    }
    @Test func quitReadsFreshRuntimeInsteadOfTheMenuCache() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.session.predictor.predictGate.close()
        let work = Task { try await f.session.lifecycle.submit(fixtureRequest()) }
        try await eventually { await f.session.predictor.predictCount == 1 }
        #expect(await f.model.workAtQuit())
        await f.session.predictor.predictGate.open()
        _ = try await work.value
    }
    @Test func unreadableStateRequiresQuitConfirmation() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.session.setSnapshotFailure(true)
        #expect(await f.model.workAtQuit())
    }
    @Test func liveSettingsAreEffectiveImmediately() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        let initial = try #require(f.model.desired)
        var draft = SettingsDraft(initial)
        draft.idleTimeoutS = "600"; draft.maxQueueDepth = "3"
        await f.model.save(draft)
        #expect(f.model.notice == nil)
        #expect(f.model.effective?.idleTimeoutS == 600)
        #expect(f.model.effective?.maxQueueDepth == 3)
        #expect(f.model.restartRequired.isEmpty)
    }
    @Test func restartSettingsDoNotMasqueradeAsEffective() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        let initial = try #require(f.model.desired)
        var draft = SettingsDraft(initial)
        draft.port = "9001"; draft.modelID = "other-model"; draft.modelRoot = "/tmp/other-root"
        draft.maxBatch = "2"; draft.computeUnits = .cpuOnly
        await f.model.save(draft)
        #expect(f.model.notice == nil)
        #expect(f.model.effective == initial)
        #expect(f.model.desired?.port == 9001)
        #expect(f.model.desired?.computeUnits == .cpuOnly)
        #expect(f.model.restartRequired == ["compute_units", "max_batch", "model_id", "model_root", "port"])
        #expect(await f.session.predictor.loadCount == 1)
    }
    @Test func modelPickerRejectsUnknownIDs() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.model.selectModel("unverified")
        #expect(f.model.notice?.code == "verification_failed")
        #expect(await f.session.applyCount == 0)
    }
    @Test func pickerReverifiesBeforeSavingInsteadOfTrustingCache() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.library.setCatalog(.init())
        await f.model.selectModel("other-model")
        #expect(f.model.notice?.code == "verification_failed")
        #expect(await f.session.applyCount == 0)
    }
    @Test func homeRelativeRootIsExpandedBeforeSelectionVerification() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        var draft = SettingsDraft(try #require(f.model.desired))
        draft.modelRoot = "~/models"; draft.modelID = "other-model"
        await f.model.save(draft)
        #expect(await f.library.lastRoot == "/tmp/fixture-home/models")
    }
    @Test func failedSettingsWriteKeepsEffectiveValues() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        let old = try #require(f.model.desired)
        await f.session.setApplyFailure(true)
        var draft = SettingsDraft(old); draft.port = "9000"
        await f.model.save(draft)
        #expect(f.model.notice?.code == "io_error")
        #expect(f.model.desired == old)
    }
    @Test func startupConfigurationFailureNeverCreatesARuntimeOrDownloads() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        await f.configuration.setFailRead(true)
        f.model.start()
        try await eventually { @MainActor in f.model.phase == .failed }
        #expect(f.factory.count == 0)
        #expect(await f.library.installs == 0)
        #expect(f.model.notice?.code == "invalid_spec")
    }
    @Test func unsupportedSavedComputeSettingCanBeRepairedWhileStopped() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        var config = try await f.configuration.read(); config.computeUnits = .cpuOnly
        await f.configuration.save(config)
        f.model.start()
        try await eventually { @MainActor in f.model.phase == .failed }
        var draft = SettingsDraft(try #require(f.model.desired)); draft.computeUnits = .cpuAndNE
        await f.model.save(draft)
        #expect(f.model.configuration?.computeUnits == .cpuAndNE)
        #expect(f.model.notice == nil)
        try await f.ready()
    }
    @Test func installationIsExplicitAndDoesNotActivateOrReplace() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        #expect(await f.library.installs == 0)
        f.model.install(from: URL(fileURLWithPath: "/tmp/model.yaml"))
        try await eventually { @MainActor in !f.model.installing }
        #expect(await f.library.installs == 1)
        #expect(f.model.installMessage?.contains("Verified installation") == true)
        #expect(f.model.progress?.phase == "promoted")
        #expect(await f.session.predictor.loadCount == 1)
        #expect(await f.session.predictor.unloadCount == 0)
    }
    @Test func concurrentInstallIsBlockedAndCancellationIsNotSuccess() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.library.setDelayedInstall(true)
        f.model.install(from: URL(fileURLWithPath: "/tmp/model.yaml"))
        try await eventually { await f.library.installs == 1 }
        f.model.install(from: URL(fileURLWithPath: "/tmp/second.yaml"))
        #expect(await f.model.workAtQuit())
        f.model.cancelInstall()
        try await eventually { @MainActor in !f.model.installing }
        #expect(await f.library.installs == 1)
        #expect(f.model.installMessage?.contains("cancelled") == true)
        #expect(f.model.progress?.phase != "promoted")
    }
    @Test func catalogFailureInvalidatesSelectableModels() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        #expect(!f.model.models.isEmpty)
        await f.library.setFailScan(true)
        await f.model.refreshInstalls()
        #expect(f.model.models.isEmpty)
        #expect(f.model.notice?.code == "verification_failed")
    }
    @Test func userUnloadDoesNotTriggerAutomaticReload() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.model.unload()
        await f.model.refreshInstalls(); await f.model.refresh()
        #expect(f.model.snapshot?.state == .unloaded)
        #expect(await f.session.predictor.loadCount == 1)
        #expect(f.model.presentation.kind == .standby)
        #expect(f.model.canLoad)
    }
    @Test func unloadedPresentationReflectsAutoLoadPolicy() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.model.unload()
        await f.model.refresh()
        #expect(f.model.presentation.kind == .standby)
        #expect(f.model.canLoad)

        var draft = SettingsDraft(f.model.effective ?? ServiceConfiguration())
        draft.autoLoad = false
        await f.model.save(draft)
        await f.model.refresh()
        #expect(f.model.presentation.kind == .unloaded)
        #expect(f.model.presentation.action == .load)
        #expect(f.model.canLoad)
    }
    @Test func restartServerStartsANewSessionAndLoadsAgain() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.model.restartServer()
        try await f.ready()
        #expect(f.factory.count == 2)
        #expect(await f.session.runCount == 2)
        #expect(!f.model.restarting)
        #expect(f.model.presentation.kind == .ready)
    }
    @Test func restartServerGivesUpWhileWorkContinues() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        await f.session.predictor.loadGate.close()
        f.model.start()
        try await eventually { await f.session.predictor.loadCount == 1 }
        await f.model.restartServer(drainTimeout: .zero)
        #expect(f.model.notice?.code == "busy")
        #expect(f.factory.count == 1)
        #expect(f.model.phase == .running)
        await f.session.predictor.loadGate.open()
    }
    @Test func cacheWarmthIsProbedAfterLoad() async throws {
        let f = try AppFixture(cacheWarm: false); defer { f.model.cancelForTermination() }
        try await f.ready()
        try await eventually { @MainActor in f.model.cacheWarm == false }
        let plain = try AppFixture(); defer { plain.model.cancelForTermination() }
        try await plain.ready()
        #expect(plain.model.cacheWarm == nil)
    }
    @Test func keepLoadedTogglesTheIdleTimeout() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.model.setKeepLoaded(false)
        await f.model.refresh()
        #expect(f.model.effective?.idleTimeoutS == 300)
        await f.model.setKeepLoaded(true)
        await f.model.refresh()
        #expect(f.model.effective?.idleTimeoutS == 0)
    }
    @Test func defaultHubRepoIsLowercase() {
        #expect(AppHubDefaults.defaultRepo == "maolon/WeMM-Embedding-2B-CoreML-ANE")
    }
    @Test func downloadFromHubHappyPathSetsProgressionAndGuidance() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        #expect(await f.library.installs == 0)
        f.model.downloadFromHub(repo: AppHubDefaults.defaultRepo)
        try await eventually { @MainActor in !f.model.installing }
        #expect(await f.library.installs == 1)
        #expect(await f.library.lastRepo == AppHubDefaults.defaultRepo)
        #expect(f.model.installMessage?.contains("Verified installation") == true)
        #expect(f.model.installMessage?.contains("Choose Use This Model to switch to it.") == true)
        #expect(f.model.progress?.phase == "promoted")
    }
    @Test func downloadFromHubRejectsEmptyRepo() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        f.model.downloadFromHub(repo: "   ")
        #expect(!f.model.installing)
        #expect(f.model.notice?.code == "invalid_request")
    }
    @Test func downloadFromHubSpecNotFoundSurfacesNotice() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.library.setFailFetchSpec(true)
        f.model.downloadFromHub(repo: "nonexistent/repo")
        try await eventually { @MainActor in !f.model.installing }
        #expect(f.model.notice?.code == "transport_error")
    }
    @Test func replaceUsesCapturedSpecWithoutRefetching() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        let chunkManifests = (0..<6).map {
            """
              - path: chunks/chunk\($0).mlmodelc/manifest.txt
                sha256: \(String(repeating: "0", count: 64))
                size: 10
            """
        }.joined(separator: "\n")
        let specV1 = """
        spec_version: 1
        model:
          id: captured-model-id
          dim: 2048
          max_seq: 512
          normalize: l2
        source:
          repo: maolon/WeMM-Embedding-2B-CoreML-ANE
          revision: \(String(repeating: "a", count: 40))
          endpoint: https://huggingface.co
        files:
          - path: tokenizer.json
            sha256: \(String(repeating: "0", count: 64))
            size: 10
          - path: tokenizer_config.json
            sha256: \(String(repeating: "0", count: 64))
            size: 10
          - path: embed_table.fp16.npy
            sha256: \(String(repeating: "0", count: 64))
            size: 10
        \(chunkManifests)
        runtime:
          compute_units: cpu_and_ne
          pad_side: right
          mask_dtype: fp32
        """
        await f.library.setFixtureSpecYAML(specV1)
        await f.library.setConflictOnInstall(.conflict("Different or invalid bundle already installed as captured-model-id; use --replace after unloading"))

        f.model.downloadFromHub(repo: "maolon/WeMM-Embedding-2B-CoreML-ANE")
        try await eventually { @MainActor in !f.model.installing }

        guard let conflict = f.model.conflictResolution else {
            Issue.record("Expected conflictResolution")
            return
        }
        #expect(conflict.modelID == "captured-model-id")
        let tempDir = conflict.specURL.deletingLastPathComponent()
        #expect(FileManager.default.fileExists(atPath: tempDir.path))

        // Mutate the library to return a different spec for subsequent fetches.
        let specV2 = """
        spec_version: 1
        model:
          id: mutated-model-id
          dim: 2048
          max_seq: 512
          normalize: l2
        source:
          repo: maolon/WeMM-Embedding-2B-CoreML-ANE
          revision: \(String(repeating: "b", count: 40))
          endpoint: https://huggingface.co
        files:
          - path: tokenizer.json
            sha256: \(String(repeating: "1", count: 64))
            size: 10
          - path: tokenizer_config.json
            sha256: \(String(repeating: "1", count: 64))
            size: 10
          - path: embed_table.fp16.npy
            sha256: \(String(repeating: "1", count: 64))
            size: 10
        \(chunkManifests)
        runtime:
          compute_units: cpu_and_ne
          pad_side: right
          mask_dtype: fp32
        """
        await f.library.setFixtureSpecYAML(specV2)

        // Resolve conflict with replace: true. It must use captured specV1, NOT mutated specV2.
        f.model.resolveConflict(replace: true)
        try await eventually { @MainActor in !f.model.installing }

        #expect(await f.library.replaces == 1)
        #expect(f.model.installOutcome?.kind == .success(modelID: "captured-model-id"))
        #expect(!FileManager.default.fileExists(atPath: tempDir.path))
    }
    @Test func cancellationDuringSpecAcquisitionRendersCancelledOutcome() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.library.setDelayFetchSpec(true)
        f.model.downloadFromHub(repo: "maolon/WeMM-Embedding-2B-CoreML-ANE")
        #expect(f.model.installing)
        f.model.cancelInstall()
        try await eventually { @MainActor in !f.model.installing }
        #expect(f.model.installOutcome?.kind == .cancelled)
        #expect(f.model.installOutcome?.message.contains("cancelled") == true)
        #expect(f.model.installOutcome?.message.contains("Use This Model") == false)
    }
    @Test func conflictFromLockShowsNoReplace() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.library.setConflictOnInstall(.conflict("Another fetch or promotion is active for wemm-embedding-2b"))
        f.model.downloadFromHub(repo: "maolon/WeMM-Embedding-2B-CoreML-ANE")
        try await eventually { @MainActor in !f.model.installing }
        #expect(f.model.notice?.message.contains("Another fetch or promotion is active") == true)
        #expect(f.model.conflictResolution == nil)
    }
    @Test func dismissConflictCleansUpTempSpec() async throws {
        let f = try AppFixture(); defer { f.model.cancelForTermination() }
        try await f.ready()
        await f.library.setConflictOnInstall(.conflict("Different or invalid bundle already installed as wemm-embedding-2b; use --replace after unloading"))
        f.model.downloadFromHub(repo: "maolon/WeMM-Embedding-2B-CoreML-ANE")
        try await eventually { @MainActor in !f.model.installing }
        guard let conflict = f.model.conflictResolution else {
            Issue.record("Expected conflictResolution")
            return
        }
        let tempDir = conflict.specURL.deletingLastPathComponent()
        #expect(FileManager.default.fileExists(atPath: tempDir.path))
        f.model.dismissConflict()
        #expect(f.model.conflictResolution == nil)
        #expect(!FileManager.default.fileExists(atPath: tempDir.path))
    }
}
