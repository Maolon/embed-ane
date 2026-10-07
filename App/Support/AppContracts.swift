import EmbedANECore
import Foundation

public enum AppHubDefaults {
    public static let defaultRepo = "maolon/WeMM-Embedding-2B-CoreML-ANE"
}

/// UI-facing value types do not own models, HTTP sockets, or installation locks.
public enum ServerPhase: String, Sendable { case stopped, starting, running, failed, stopping }

public struct AppSnapshot: Sendable, Equatable, Codable {
    public let state: LifecycleState
    public let queueDepth: Int
    public let inFlight: Int
    public let preparing: Int
    public let residentBytes: UInt64
    public let windowCount: Int
    public let p50NS: UInt64
    public let p95NS: UInt64
    public let effective: ServiceConfiguration
    public let desired: ServiceConfiguration
    public let restartRequired: [String]

    public var autoLoad: Bool { effective.autoLoad }

    public init(state: LifecycleState, queueDepth: Int = 0, inFlight: Int = 0,
                preparing: Int = 0, residentBytes: UInt64 = 0, windowCount: Int = 0,
                p50NS: UInt64 = 0, p95NS: UInt64 = 0,
                autoLoad: Bool? = nil,
                effective: ServiceConfiguration = .init(), desired: ServiceConfiguration? = nil,
                restartRequired: [String] = []) {
        self.state = state; self.queueDepth = queueDepth; self.inFlight = inFlight
        self.preparing = preparing; self.residentBytes = residentBytes
        self.windowCount = windowCount; self.p50NS = p50NS; self.p95NS = p95NS
        var eff = effective
        if let autoLoad { eff.autoLoad = autoLoad }
        self.effective = eff; self.desired = desired ?? eff
        self.restartRequired = restartRequired
    }
    public var hasWork: Bool {
        queueDepth > 0 || inFlight > 0 || preparing > 0 || state == .loading || state == .unloading
    }
    public var canUnload: Bool { !hasWork && (state == .ready || state == .failed) }
    public var isVisionEnabled: Bool {
        effective.visionTowerPath != nil && effective.visionExtropeChunksDirectory != nil && effective.visionPositionTablePath != nil
    }
}

/// Implemented by the in-process HTTP composition, not by a remote control client.
public protocol AppSession: Sendable {
    func run(onListening: @escaping @Sendable (Int) async -> Void) async throws
    func snapshot() async throws -> AppSnapshot
    func load() async throws
    func unload() async throws
    func apply(_ overrides: ConfigurationOverrides) async throws
}

public struct VerifiedModel: Identifiable, Sendable, Equatable {
    public let id: String
    public let commit: String
    public init(id: String, commit: String) { self.id = id; self.commit = commit }
}
public struct RejectedInstall: Sendable, Equatable, Identifiable {
    public let id: String
    public let code: String
    public init(id: String, code: String) { self.id = id; self.code = code }
}
public struct InstallCatalog: Sendable {
    public let models: [VerifiedModel]
    public let rejected: [RejectedInstall]
    public init(models: [VerifiedModel] = [], rejected: [RejectedInstall] = []) {
        self.models = models; self.rejected = rejected
    }
}
public struct InstallationProgress: Sendable, Equatable {
    public let phase: String
    public let path: String?
    public let received: Int64
    public let total: Int64
    public init(phase: String, path: String? = nil, received: Int64 = 0, total: Int64 = 0) {
        self.phase = phase; self.path = path; self.received = received; self.total = total
    }
    /// Per-file transfer progress, never misrepresented as overall bundle progress.
    public var fraction: Double? {
        guard phase == "downloading", total > 0 else { return nil }
        return min(1, max(0, Double(received) / Double(total)))
    }
}
public struct InstallationResult: Sendable {
    public let modelID: String
    public let warnings: [String]
    public init(modelID: String, warnings: [String] = []) { self.modelID = modelID; self.warnings = warnings }
}
public protocol AppModelLibrary: Sendable {
    func scan(root: URL) async throws -> InstallCatalog
    func install(spec: URL, root: URL, replace: Bool,
                 progress: @escaping @Sendable (InstallationProgress) -> Void) async throws -> InstallationResult
    func fetchSpec(repo: String, endpoint: URL) async throws -> URL
}

extension AppModelLibrary {
    public func install(spec: URL, root: URL,
                        progress: @escaping @Sendable (InstallationProgress) -> Void) async throws -> InstallationResult {
        try await install(spec: spec, root: root, replace: false, progress: progress)
    }
    public func fetchSpec(repo: String) async throws -> URL {
        try await fetchSpec(repo: repo, endpoint: URL(string: "https://huggingface.co")!)
    }
}

/// Only fixed event codes are logged: no prompts, vectors, tokens, URLs, or raw errors.
public enum AppEvent: String, Sendable {
    case serverListening = "server_listening", serverStopped = "server_stopped", serverFailed = "server_failed"
    case loadCompleted = "load_completed", loadFailed = "load_failed"
    case unloadCompleted = "unload_completed", unloadFailed = "unload_failed"
    case settingsSaved = "settings_saved", settingsFailed = "settings_failed"
    case catalogChecked = "catalog_checked", catalogFailed = "catalog_failed"
    case installStarted = "install_started", installCompleted = "install_completed"
    case installFailed = "install_failed", installCancelled = "install_cancelled"
    case workerProbeFailed = "worker_probe_failed"
    case workerProbeWaiting = "worker_probe_waiting"
    case evictionDeferredColdCache = "eviction_deferred_cold_cache"
    case quitting
}
public struct AppNotice: Sendable, Equatable {
    public let message: String
    public let code: String
    public init(message: String, code: String) { self.message = message; self.code = code }
    public init(error: any Error) {
        if let error = error as? EmbedANEError {
            self.init(message: error.message, code: error.code)
        } else if error is CancellationError {
            self.init(message: "Operation cancelled.", code: "cancelled")
        } else {
            self.init(message: "The operation failed. No configuration or model replacement was assumed successful.", code: "internal_error")
        }
    }
}

/// Composition seams for unit tests. Production wiring lives in App/Sources.
public struct AppServices: Sendable {
    public let configuration: @Sendable () async throws -> ServiceConfiguration
    public let saveConfiguration: @Sendable (ServiceConfiguration) async throws -> Void
    public let makeSession: @Sendable (ServiceConfiguration) throws -> any AppSession
    public let library: any AppModelLibrary
    public let expandRoot: @Sendable (String) -> String
    public let record: @Sendable (AppEvent) -> Void
    public let pause: @Sendable () async throws -> Void
    /// Compile-cache warmth for a configuration; nil when it cannot be determined.
    public let cacheHealth: @Sendable (ServiceConfiguration) async -> Bool?
    public init(configuration: @escaping @Sendable () async throws -> ServiceConfiguration,
                saveConfiguration: @escaping @Sendable (ServiceConfiguration) async throws -> Void,
                makeSession: @escaping @Sendable (ServiceConfiguration) throws -> any AppSession,
                library: any AppModelLibrary,
                expandRoot: @escaping @Sendable (String) -> String = { ($0 as NSString).expandingTildeInPath },
                record: @escaping @Sendable (AppEvent) -> Void = { _ in },
                pause: @escaping @Sendable () async throws -> Void = { try await Task.sleep(for: .milliseconds(500)) },
                cacheHealth: @escaping @Sendable (ServiceConfiguration) async -> Bool? = { _ in nil }) {
        self.configuration = configuration; self.saveConfiguration = saveConfiguration
        self.makeSession = makeSession; self.library = library; self.expandRoot = expandRoot; self.record = record; self.pause = pause
        self.cacheHealth = cacheHealth
    }
}
