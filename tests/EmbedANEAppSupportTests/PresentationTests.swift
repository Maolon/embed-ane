import EmbedANEAppSupport
import EmbedANECore
import Testing

@Suite("App settings and presentation values") struct PresentationTests {
    @Test(arguments: [LifecycleState.unloaded, .loading, .ready, .unloading, .failed])
    func everyLifecycleStateMapsToAVisibleStatus(state: LifecycleState) {
        let snapshot = AppSnapshot(state: state)
        let status = StatusPresentation.make(phase: .running, snapshot: snapshot, autoLoad: true)
        #expect(!status.title.isEmpty)
        if state == .loading || state == .unloading {
            #expect(snapshot.hasWork)
            #expect(status.kind == .working && status.action == nil)
        }
        if state == .unloaded { #expect(!snapshot.canUnload) }
    }
    @Test func serverPhasesAndModelStatesMapToSixKinds() {
        let ready = AppSnapshot(state: .ready), unloaded = AppSnapshot(state: .unloaded)
        func kind(_ phase: ServerPhase, _ snapshot: AppSnapshot?, autoLoad: Bool = true, restarting: Bool = false) -> StatusKind {
            StatusPresentation.make(phase: phase, snapshot: snapshot, autoLoad: autoLoad, restarting: restarting).kind
        }
        #expect(kind(.running, ready) == .ready)
        #expect(kind(.running, unloaded) == .standby)
        #expect(kind(.running, unloaded, autoLoad: false) == .unloaded)
        #expect(kind(.running, nil) == .working)
        #expect(kind(.starting, nil) == .working)
        #expect(kind(.stopping, ready) == .working)
        #expect(kind(.stopped, nil) == .stopped)
        #expect(kind(.stopped, nil, restarting: true) == .working)
        #expect(kind(.failed, nil) == .error)
        #expect(kind(.running, AppSnapshot(state: .failed)) == .error)
    }
    @Test func primaryActionFollowsTheState() {
        #expect(StatusPresentation.make(phase: .running, snapshot: AppSnapshot(state: .ready), autoLoad: true).action == .unload)
        #expect(StatusPresentation.make(phase: .running, snapshot: AppSnapshot(state: .unloaded), autoLoad: false).action == .load)
        #expect(StatusPresentation.make(phase: .running, snapshot: AppSnapshot(state: .failed), autoLoad: true).action == .retryLoad)
        #expect(StatusPresentation.make(phase: .failed, snapshot: nil, autoLoad: true).action == .restartServer)
        #expect(StatusPresentation.make(phase: .stopped, snapshot: nil, autoLoad: true).action == .startServer)
    }
    @Test func standbyQuotesTheMeasuredReloadTime() {
        let status = StatusPresentation.make(phase: .running, snapshot: AppSnapshot(state: .unloaded), autoLoad: true)
        #expect(status.detail == "Loads on next request (~15–30 s)")
    }
    @Test func workingAnimationWaitsHalfASecond() {
        let start = ContinuousClock.now
        #expect(!WorkingAnimation.shouldAnimate(workingSince: nil, now: start))
        #expect(!WorkingAnimation.shouldAnimate(workingSince: start, now: start + .milliseconds(499)))
        #expect(WorkingAnimation.shouldAnimate(workingSince: start, now: start + .milliseconds(500)))
    }
    @Test(arguments: [0, 1, 2]) func queuePredictionAndPreparationAllBlockUnload(kind: Int) {
        let snapshot = AppSnapshot(state: .ready, queueDepth: kind == 0 ? 1 : 0,
                                   inFlight: kind == 1 ? 1 : 0, preparing: kind == 2 ? 1 : 0)
        #expect(snapshot.hasWork && !snapshot.canUnload)
    }
    @Test func unchangedDraftProducesAnEmptyPatch() throws {
        let draft = SettingsDraft(ServiceConfiguration())
        #expect(!draft.hasChanges)
        #expect(try draft.overrides() == ConfigurationOverrides())
    }
    @Test func editSendsOnlyDirtyFields() throws {
        var draft = SettingsDraft(ServiceConfiguration())
        draft.port = "8888"
        #expect(draft.hasChanges)
        #expect(try draft.overrides() == ConfigurationOverrides(port: 8888))
    }
    @Test(arguments: ["text", "NaN", "inf", "-1", "1000000001"])
    func invalidTimeoutNeverProducesAPatch(value: String) {
        var draft = SettingsDraft(ServiceConfiguration()); draft.idleTimeoutS = value
        #expect(throws: EmbedANEError.self) { try draft.overrides() }
    }
    @Test(arguments: ["0", "65536", "not-a-port"])
    func invalidPortNeverProducesAPatch(value: String) {
        var draft = SettingsDraft(ServiceConfiguration()); draft.port = value
        #expect(throws: EmbedANEError.self) { try draft.overrides() }
    }
    @Test func persistedLowerPrecedenceValuesDoNotOverrideEffectiveSettings() {
        let effective = ServiceConfiguration(port: 8888, modelRoot: "/env/root", idleTimeoutS: 45, maxQueueDepth: 3)
        let saved = ServiceConfiguration(port: 9999, modelRoot: "/file/root", idleTimeoutS: 1, maxQueueDepth: 9)
        let desired = SettingsDraft.desired(effective: effective, persisted: saved, restartRequired: ["port"])
        #expect(desired.port == 9999)
        #expect(desired.modelRoot == "/env/root")
        #expect(desired.idleTimeoutS == 45 && desired.maxQueueDepth == 3)
    }
    @Test func transferFractionIsPerFileAndClamped() {
        #expect(InstallationProgress(phase: "verifying", received: 1, total: 2).fraction == nil)
        #expect(InstallationProgress(phase: "downloading", received: 1, total: 0).fraction == nil)
        #expect(InstallationProgress(phase: "downloading", received: -1, total: 2).fraction == 0)
        #expect(InstallationProgress(phase: "downloading", received: 1, total: 2).fraction == 0.5)
        #expect(InstallationProgress(phase: "downloading", received: 4, total: 2).fraction == 1)
    }

    @Test func visionEnabledInSnapshotWhenAllThreeKeysSet() {
        let noVision = AppSnapshot(state: .ready, effective: ServiceConfiguration())
        #expect(!noVision.isVisionEnabled)

        let withVision = AppSnapshot(
            state: .ready,
            effective: ServiceConfiguration(
                visionTowerPath: "/tower",
                visionExtropeChunksDirectory: "/chunks",
                visionPositionTablePath: "/pos.npy"
            )
        )
        #expect(withVision.isVisionEnabled)
    }

    @Test func presetRoundTripAndCustomSeconds() {
        #expect(IdleTimeoutPreset.from(seconds: 300) == .fiveMinutes)
        #expect(IdleTimeoutPreset.from(seconds: 1800) == .thirtyMinutes)
        #expect(IdleTimeoutPreset.from(seconds: 3600) == .oneHour)
        #expect(IdleTimeoutPreset.from(seconds: 0) == .custom)
        #expect(IdleTimeoutPreset.from(seconds: 45) == .custom)
        #expect(IdleTimeoutPreset.from(seconds: 999) == .custom)

        #expect(IdleTimeoutPreset.fiveMinutes.seconds == 300)
        #expect(IdleTimeoutPreset.thirtyMinutes.seconds == 1800)
        #expect(IdleTimeoutPreset.oneHour.seconds == 3600)
        #expect(IdleTimeoutPreset.custom.seconds == nil)

        var draft = SettingsDraft(ServiceConfiguration(idleTimeoutS: 300))
        #expect(draft.idlePreset == .fiveMinutes)
        draft.idlePreset = .thirtyMinutes
        #expect(draft.idleTimeoutS == "1800")
        draft.idlePreset = .oneHour
        #expect(draft.idleTimeoutS == "3600")
        draft.idleTimeoutS = "450"
        #expect(draft.idlePreset == .custom)
    }

    @Test func autoLoadToggleInDraftProducesOverrides() throws {
        var draft = SettingsDraft(ServiceConfiguration(autoLoad: true))
        #expect(!draft.hasChanges)
        draft.autoLoad = false
        #expect(draft.hasChanges)
        #expect(try draft.overrides() == ConfigurationOverrides(autoLoad: false))
        draft.autoLoad = true
        #expect(!draft.hasChanges)
    }

    @Test func computeUnitsPresentationDefaultsToNeuralEngineAndSupportsReset() throws {
        // Default draft renders Neural Engine row
        let defaultDraft = SettingsDraft(ServiceConfiguration())
        #expect(defaultDraft.computeUnits == .cpuAndNE)
        #expect(defaultDraft.computeUnitsLabel == "Neural Engine")
        #expect(defaultDraft.isComputeUnitsSupported)
        #expect(!defaultDraft.requiresComputeUnitsReset)
        #expect(defaultDraft.computeUnitsWarning == nil)

        // A persisted cpu_only config renders warning + reset action
        var unsupportedDraft = SettingsDraft(ServiceConfiguration(computeUnits: .cpuOnly))
        #expect(unsupportedDraft.computeUnits == .cpuOnly)
        #expect(unsupportedDraft.computeUnitsLabel == "CPU only")
        #expect(!unsupportedDraft.isComputeUnitsSupported)
        #expect(unsupportedDraft.requiresComputeUnitsReset)
        let warning = try #require(unsupportedDraft.computeUnitsWarning)
        #expect(warning.contains("unsupported"))

        // Reset flips draft to cpuAndNE and clears the warning
        unsupportedDraft.resetComputeUnits()
        #expect(unsupportedDraft.computeUnits == .cpuAndNE)
        #expect(unsupportedDraft.computeUnitsLabel == "Neural Engine")
        #expect(unsupportedDraft.isComputeUnitsSupported)
        #expect(!unsupportedDraft.requiresComputeUnitsReset)
        #expect(unsupportedDraft.computeUnitsWarning == nil)

        // Overrides report the repaired computeUnits value
        #expect(unsupportedDraft.hasChanges)
        let overrides = try unsupportedDraft.overrides()
        #expect(overrides.computeUnits == .cpuAndNE)
    }

    @Test(arguments: [ComputeUnitsSetting.cpuOnly, .cpuAndGPU, .all])
    func unsupportedComputeUnitsRendersWarningAndCanReset(setting: ComputeUnitsSetting) throws {
        var draft = SettingsDraft(ServiceConfiguration(computeUnits: setting))
        #expect(!draft.isComputeUnitsSupported)
        #expect(draft.requiresComputeUnitsReset)
        #expect(draft.computeUnitsWarning != nil)
        draft.resetComputeUnits()
        #expect(draft.isComputeUnitsSupported)
        #expect(!draft.requiresComputeUnitsReset)
        #expect(draft.computeUnitsWarning == nil)
        #expect(draft.computeUnitsLabel == "Neural Engine")
    }
}
