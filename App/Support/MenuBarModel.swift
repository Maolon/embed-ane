import EmbedANECore
import Foundation
import Observation

/// One instance for the entire process, not one instance per menu presentation.
/// Runtime polling continues when the popover and Settings window are closed.
@MainActor @Observable public final class MenuBarModel {
    public private(set) var phase: ServerPhase = .stopped
    public private(set) var listeningPort: Int?
    public private(set) var snapshot: AppSnapshot?
    public private(set) var configuration: ServiceConfiguration?
    public private(set) var models: [VerifiedModel] = []
    public private(set) var rejectedInstalls: [RejectedInstall] = []
    public private(set) var scanning = false
    public private(set) var commanding = false
    public private(set) var saving = false
    public private(set) var installing = false
    public private(set) var progress: InstallationProgress?
    public struct InstallOutcome: Sendable, Equatable {
        public enum Kind: Sendable, Equatable {
            case success(modelID: String)
            case cancelled
        }
        public let kind: Kind
        public let message: String
        public init(kind: Kind, message: String) {
            self.kind = kind; self.message = message
        }
    }
    public private(set) var installOutcome: InstallOutcome?
    public var installMessage: String? { installOutcome?.message }
    public struct ConflictResolution: Sendable, Equatable {
        public let modelID: String
        public let specURL: URL
        public let isTemporarySpec: Bool
        public init(modelID: String, specURL: URL, isTemporarySpec: Bool = false) {
            self.modelID = modelID; self.specURL = specURL; self.isTemporarySpec = isTemporarySpec
        }
    }
    public private(set) var conflictResolution: ConflictResolution?
    public private(set) var notice: AppNotice?
    public private(set) var quitting = false
    public private(set) var restarting = false
    /// Whether the compile cache still holds the models; nil until probed or when unknown.
    /// A cold cache turns an unload or restart into a long re-specialization.
    public private(set) var cacheWarm: Bool?
    @ObservationIgnored private let services: AppServices
    @ObservationIgnored private var session: (any AppSession)?
    @ObservationIgnored private var startupTask: Task<Void, Never>?
    @ObservationIgnored private var serverTask: Task<Void, Never>?
    @ObservationIgnored private var pollTask: Task<Void, Never>?
    @ObservationIgnored private var installTask: Task<Void, Never>?
    @ObservationIgnored private var generation: UInt64 = 0
    @ObservationIgnored private var refreshSequence: UInt64 = 0
    @ObservationIgnored private var appliedSequence: UInt64 = 0
    @ObservationIgnored private var catalogReady = false
    @ObservationIgnored private var initialLoadPending = false

    public init(services: AppServices) { self.services = services }
    public var effective: ServiceConfiguration? { snapshot?.effective ?? configuration }
    public var desired: ServiceConfiguration? { snapshot?.desired ?? configuration }
    public var restartRequired: [String] { snapshot?.restartRequired ?? [] }
    public var canInstall: Bool { !installing && !quitting && effective != nil }
    public var currentModelIsVerified: Bool {
        guard let id = effective?.modelID else { return false }
        return models.contains { $0.id == id }
    }
    public var canLoad: Bool {
        guard phase == .running, !commanding, !quitting, !restarting, currentModelIsVerified else { return false }
        return snapshot?.state == .unloaded || snapshot?.state == .failed
    }
    public var canUnload: Bool {
        !commanding && !quitting && !restarting && (phase == .running || phase == .failed) && snapshot?.canUnload == true
    }
    public var canRestart: Bool { !restarting && !quitting && !installing && phase != .starting && phase != .stopping }
    public var presentation: StatusPresentation {
        StatusPresentation.make(phase: phase, snapshot: snapshot,
                                autoLoad: effective?.autoLoad ?? true, restarting: restarting)
    }

    public func start() {
        guard !quitting, phase == .stopped || phase == .failed,
              startupTask == nil, serverTask == nil, !commanding, !installing else { return }
        generation &+= 1
        let epoch = generation
        phase = .starting; notice = nil; initialLoadPending = true; catalogReady = false
        startupTask = Task { [weak self] in
            guard let self else { return }
            defer { self.startupTask = nil }
            do {
                // A prior listener failure must not leave a second cascade owner.
                if let previous = self.session {
                    let state = try await previous.snapshot()
                    guard !state.hasWork else { throw EmbedANEError.busy }
                    try await previous.unload()
                }
                let config = try await self.services.configuration()
                try config.validate()
                guard epoch == self.generation, !Task.isCancelled else { return }
                self.configuration = config; self.snapshot = nil; self.models = []
                let session = try self.services.makeSession(config)
                self.session = session
                self.serverTask = Task { [weak self, session] in
                    do {
                        try await session.run { [weak self] port in await self?.didListen(port, epoch: epoch) }
                        self?.serverEnded(epoch: epoch, error: nil)
                    } catch {
                        self?.serverEnded(epoch: epoch, error: error)
                    }
                }
                self.pollTask = Task { [weak self, services = self.services] in
                    while !Task.isCancelled {
                        guard let self, epoch == self.generation else { return }
                        await self.refresh()
                        do { try await services.pause() } catch { return }
                    }
                }
                await self.refreshInstalls()
                await self.attemptInitialLoad()
            } catch {
                guard epoch == self.generation else { return }
                self.phase = .failed; self.notice = AppNotice(error: error)
                self.services.record(.serverFailed)
            }
        }
    }
    private func didListen(_ port: Int, epoch: UInt64) async {
        guard epoch == generation, !quitting else { return }
        listeningPort = port; phase = .running
        services.record(.serverListening)
        await refresh()
        Task { [weak self] in await self?.checkCacheHealth() }
        // Do not hold Hummingbird's on-listening callback during model loading.
        Task { [weak self] in await self?.attemptInitialLoad() }
    }
    private func serverEnded(epoch: UInt64, error: (any Error)?) {
        guard epoch == generation else { return }
        serverTask = nil; listeningPort = nil; pollTask?.cancel(); pollTask = nil
        if quitting || phase == .stopping {
            phase = .stopped; services.record(.serverStopped)
        } else {
            phase = .failed; initialLoadPending = false
            notice = error.map(AppNotice.init(error:)) ?? AppNotice(message: "The HTTP listener stopped unexpectedly.", code: "server_stopped")
            services.record(.serverFailed)
        }
    }
    private func attemptInitialLoad() async {
        // Catalogue completion can race the first snapshot returned by onListening.
        // Do not consume the one-shot load intent until both are available.
        guard initialLoadPending, catalogReady, snapshot != nil, phase == .running, !quitting else { return }
        initialLoadPending = false
        if canLoad { await load() }
    }
    public func refresh() async {
        guard let session, !quitting else { return }
        refreshSequence &+= 1
        let sequence = refreshSequence, epoch = generation
        do {
            let value = try await session.snapshot()
            guard epoch == generation, sequence >= appliedSequence, !quitting else { return }
            appliedSequence = sequence; snapshot = value
        } catch {
            guard epoch == generation, !quitting else { return }
            notice = AppNotice(error: error)
        }
    }
    public func refreshInstalls() async {
        guard !scanning, !quitting, let config = effective else { return }
        scanning = true
        let epoch = generation
        defer { scanning = false }
        do {
            let catalog = try await services.library.scan(root: URL(fileURLWithPath: services.expandRoot(config.modelRoot), isDirectory: true))
            guard epoch == generation, !quitting else { return }
            models = catalog.models.sorted { $0.id < $1.id }; rejectedInstalls = catalog.rejected
            catalogReady = true; services.record(.catalogChecked)
        } catch {
            guard epoch == generation, !quitting else { return }
            // Failed verification must not leave a stale selectable catalogue.
            models = []; rejectedInstalls = []; catalogReady = false
            notice = AppNotice(error: error); services.record(.catalogFailed)
        }
    }
    public func load() async {
        initialLoadPending = false
        guard canLoad, let session else { return }
        commanding = true; notice = nil
        defer { commanding = false }
        do { try await session.load(); services.record(.loadCompleted) }
        catch { notice = AppNotice(error: error); services.record(.loadFailed) }
        await refresh()
        await checkCacheHealth()
    }
    public func unload() async {
        initialLoadPending = false
        guard !commanding, !quitting, let session else { return }
        // The UI snapshot is only a hint; the lifecycle makes the final decision.
        commanding = true; notice = nil
        defer { commanding = false }
        do {
            let current = try await session.snapshot()
            guard !current.hasWork else { throw EmbedANEError.busy }
            try await session.unload(); services.record(.unloadCompleted)
        } catch { notice = AppNotice(error: error); services.record(.unloadFailed) }
        await refresh()
    }
    /// Re-reads the compile cache size. Walks the model folders, so call it on
    /// state changes and before a costly action, not on every poll.
    @discardableResult public func checkCacheHealth() async -> Bool? {
        guard let config = effective else { return nil }
        let warm = await services.cacheHealth(config)
        cacheWarm = warm
        return warm
    }
    /// Restarts the HTTP server and its model worker in place of quitting the app.
    /// Waits briefly for in-flight work instead of interrupting requests.
    public func restartServer(drainTimeout: Duration = .seconds(30)) async {
        guard canRestart else { return }
        restarting = true
        defer { restarting = false }
        if phase == .running {
            let deadline = ContinuousClock.now + drainTimeout
            while await workAtQuit() {
                guard ContinuousClock.now < deadline, !quitting else {
                    notice = AppNotice(message: "Requests are still running. Restart again when the server is idle.", code: "busy")
                    return
                }
                do { try await services.pause() } catch { return }
            }
            do { try await stopWhenIdle() }
            catch { notice = AppNotice(error: error); return }
        }
        guard !quitting else { return }
        restarting = false
        start()
    }
    /// Menu shortcut for the idle-timeout policy: 0 keeps the model loaded.
    public func setKeepLoaded(_ keep: Bool) async {
        await apply(ConfigurationOverrides(idleTimeoutS: keep ? 0 : 300))
    }
    public func selectModel(_ id: String) async {
        guard models.contains(where: { $0.id == id }) else {
            notice = AppNotice(message: "Only verified installations can be selected.", code: "verification_failed")
            return
        }
        await apply(ConfigurationOverrides(modelID: id))
    }
    public func save(_ draft: SettingsDraft) async {
        do { await apply(try draft.overrides()) }
        catch { notice = AppNotice(error: error) }
    }
    private func apply(_ patch: ConfigurationOverrides) async {
        guard !saving, !quitting else { return }
        saving = true; notice = nil
        defer { saving = false }
        do {
            // Reverify a changed model against the requested root before saving.
            // Cached picker entries are not an activation or verification bypass.
            if let id = patch.modelID {
                guard let root = patch.modelRoot ?? desired?.modelRoot else { throw EmbedANEError.modelNotLoaded }
                // Expand through the configuration store before constructing a file URL.
                let catalog = try await services.library.scan(root: URL(fileURLWithPath: services.expandRoot(root), isDirectory: true))
                guard catalog.models.contains(where: { $0.id == id }) else {
                    throw EmbedANEError.verification(path: id, reason: "Selected model is not a verified installation.")
                }
            }
            if phase == .running, let session {
                try await session.apply(patch)
                await refresh()
            } else {
                // Startup failures (bad port or unsupported saved compute units)
                // must still be repairable without a functioning HTTP listener.
                let current = try await services.configuration()
                let candidate = try current.applying(patch)
                try await services.saveConfiguration(candidate)
                configuration = candidate; snapshot = nil
            }
            services.record(.settingsSaved)
        } catch { notice = AppNotice(error: error); services.record(.settingsFailed) }
    }
    public func install(from url: URL, replace: Bool = false) {
        performInstall(specAcquisition: { _ in url }, isTemporarySpec: false, replace: replace)
    }

    public func downloadFromHub(repo: String, endpoint: URL? = nil, replace: Bool = false) {
        let cleanRepo = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanRepo.isEmpty else {
            notice = AppNotice(message: "Enter a repository name in org/name format.", code: "invalid_request")
            return
        }
        performInstall(
            specAcquisition: { [services] continuation in
                continuation.yield(InstallationProgress(phase: "fetching_spec", path: "spec.yaml"))
                if let endpoint {
                    return try await services.library.fetchSpec(repo: cleanRepo, endpoint: endpoint)
                } else {
                    return try await services.library.fetchSpec(repo: cleanRepo)
                }
            },
            isTemporarySpec: true,
            replace: replace
        )
    }

    public func resolveConflict(replace: Bool) {
        guard let conflict = conflictResolution else { return }
        conflictResolution = nil
        if replace {
            // P1 fix: Reuse the captured specURL directly without re-fetching from mutable main.
            performInstall(
                specAcquisition: { _ in conflict.specURL },
                isTemporarySpec: conflict.isTemporarySpec,
                replace: true
            )
        } else {
            cleanTemporarySpec(conflict.specURL, isTemporary: conflict.isTemporarySpec)
        }
    }

    public func dismissConflict() {
        if let conflict = conflictResolution {
            cleanTemporarySpec(conflict.specURL, isTemporary: conflict.isTemporarySpec)
        }
        conflictResolution = nil
    }

    private func cleanTemporarySpec(_ url: URL?, isTemporary: Bool) {
        guard isTemporary, let url else { return }
        let dir = url.deletingLastPathComponent()
        try? FileManager.default.removeItem(at: dir)
    }

    private func performInstall(
        specAcquisition: @escaping @Sendable (AsyncStream<InstallationProgress>.Continuation) async throws -> URL,
        isTemporarySpec: Bool,
        replace: Bool
    ) {
        guard canInstall, let config = effective else { return }
        if let existingConflict = conflictResolution {
            cleanTemporarySpec(existingConflict.specURL, isTemporary: existingConflict.isTemporarySpec)
            conflictResolution = nil
        }
        installing = true; notice = nil; installOutcome = nil; conflictResolution = nil
        progress = InstallationProgress(phase: isTemporarySpec ? "fetching_spec" : "resolving", path: "spec.yaml")
        let epoch = generation
        services.record(.installStarted)
        // A bounded stream coalesces byte-level callbacks instead of allocating
        // one main-actor task for every network chunk.
        let (stream, continuation) = AsyncStream.makeStream(of: InstallationProgress.self, bufferingPolicy: .bufferingNewest(1))
        installTask = Task { [weak self] in
            guard let self else { continuation.finish(); return }
            let observer = Task { [weak self] in
                for await update in stream {
                    guard let self, epoch == self.generation, !self.quitting else { return }
                    self.progress = update
                }
            }
            defer {
                continuation.finish(); observer.cancel()
                self.installing = false; self.installTask = nil
            }
            var acquiredSpecURL: URL?
            do {
                let specURL = try await specAcquisition(continuation)
                acquiredSpecURL = specURL
                try Task.checkCancellation()
                let root = URL(fileURLWithPath: self.services.expandRoot(config.modelRoot), isDirectory: true)
                let result = try await self.services.library.install(
                    spec: specURL,
                    root: root,
                    replace: replace,
                    progress: { continuation.yield($0) }
                )
                continuation.finish(); await observer.value
                guard epoch == self.generation, !self.quitting else {
                    self.cleanTemporarySpec(acquiredSpecURL, isTemporary: isTemporarySpec)
                    return
                }
                self.cleanTemporarySpec(acquiredSpecURL, isTemporary: isTemporarySpec)
                self.progress = InstallationProgress(phase: "promoted")
                let warningText = result.warnings.isEmpty ? "" : " " + result.warnings.joined(separator: " ")
                let message = "Verified installation: \(result.modelID). Choose Use This Model to switch to it." + warningText
                self.installOutcome = InstallOutcome(
                    kind: .success(modelID: result.modelID),
                    message: message
                )
                self.services.record(.installCompleted)
                await self.refreshInstalls()
                // Install never silently replaces or activates the current model.
            } catch {
                continuation.finish(); await observer.value
                guard epoch == self.generation, !self.quitting else {
                    self.cleanTemporarySpec(acquiredSpecURL, isTemporary: isTemporarySpec)
                    return
                }
                if error is CancellationError || (error as? EmbedANEError) == .cancelled {
                    self.cleanTemporarySpec(acquiredSpecURL, isTemporary: isTemporarySpec)
                    self.installOutcome = InstallOutcome(
                        kind: .cancelled,
                        message: "Installation cancelled. Recoverable staging is retained for resume."
                    )
                    self.services.record(.installCancelled)
                } else if let conflictError = error as? EmbedANEError, case let .conflict(reason) = conflictError {
                    // P2 a: Only offer Replace when conflict is an existing install; lock/resolution failures show plain notice.
                    let isReplaceable = reason.contains("already installed") || reason.contains("--replace")
                    if isReplaceable, let acquiredSpecURL {
                        let text = (try? String(contentsOf: acquiredSpecURL, encoding: .utf8)) ?? ""
                        let modelID = (try? ModelSpec.parse(text))?.model.id
                            ?? (try? StrictYAML.decode(SpecModelIDProbe.self, from: text))?.model.id
                            ?? "existing model"
                        self.conflictResolution = ConflictResolution(modelID: modelID, specURL: acquiredSpecURL, isTemporarySpec: isTemporarySpec)
                    } else {
                        self.cleanTemporarySpec(acquiredSpecURL, isTemporary: isTemporarySpec)
                    }
                    self.notice = AppNotice(error: error)
                    self.services.record(.installFailed)
                } else {
                    self.cleanTemporarySpec(acquiredSpecURL, isTemporary: isTemporarySpec)
                    self.notice = AppNotice(error: error); self.services.record(.installFailed)
                }
            }
        }
    }
    public func cancelInstall() { installTask?.cancel() }
    public func dismissNotice() { notice = nil }
    public func report(_ error: any Error) { notice = AppNotice(error: error) }

    /// Fresh actor data, not the half-second menu cache, drives the quit warning.
    public func workAtQuit() async -> Bool {
        if installing || commanding || saving || phase == .starting { return true }
        guard let session else { return false }
        do { return try await session.snapshot().hasWork }
        catch { return true } // Missing state is not evidence of quiescence.
    }
    /// Used only after the App delegate has obtained the user's quit decision.
    /// Process exit releases all OS resources; this does NOT pretend to drain or
    /// cancel a shared CoreML operation through an individual lifecycle waiter.
    public func cancelForTermination() {
        quitting = true; phase = .stopping; initialLoadPending = false; generation &+= 1
        startupTask?.cancel(); serverTask?.cancel(); pollTask?.cancel(); installTask?.cancel()
        if let conflict = conflictResolution {
            cleanTemporarySpec(conflict.specURL, isTemporary: conflict.isTemporarySpec)
            conflictResolution = nil
        }
        services.record(.quitting)
    }
    /// Test/controlled idle teardown. Refuses active work instead of implementing
    /// the future draining mode. A raced admission is still rejected by unload.
    public func stopWhenIdle() async throws {
        guard !(await workAtQuit()) else { throw EmbedANEError.busy }
        phase = .stopping; initialLoadPending = false
        pollTask?.cancel(); pollTask = nil
        let running = serverTask
        running?.cancel(); await running?.value
        if let session { try await session.unload() }
        self.session = nil; snapshot = nil; listeningPort = nil; serverTask = nil; phase = .stopped
    }
}

private struct SpecModelIDProbe: Decodable {
    struct Model: Decodable { let id: String }
    let model: Model
}
