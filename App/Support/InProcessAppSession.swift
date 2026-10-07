import EmbedANECore
import EmbedANEDownload
import EmbedANEHTTP
import Foundation

/// App composition of existing public APIs. It does not spawn the CLI, connect
/// to an unrelated local server, or create a second LifecycleActor.
public struct InProcessAppSession: AppSession, Sendable {
    public let server: EmbeddingHTTPServer
    private let store: ConfigurationStore
    private let io = AppIO()

    public init(configuration: ServiceConfiguration, predictor: any EmbeddingPredictor,
                preparer: any EmbeddingPreparer, store: ConfigurationStore,
                tokenDirectory: URL? = nil, portOverride: Int? = nil) throws {
        self.store = store
        server = try EmbeddingHTTPServer(configuration: configuration, predictor: predictor,
            preparer: preparer, store: store, tokenDirectory: tokenDirectory, portOverride: portOverride)
    }
    public static func production(configuration: ServiceConfiguration, store: ConfigurationStore) throws -> Self {
        let expanded = store.expandingPaths(configuration)
        let root = URL(fileURLWithPath: expanded.modelRoot, isDirectory: true)
        let id = expanded.modelID
        let components = try ServingRuntimeFactory.make(configuration: expanded, acquireLease: {
            try ModelUseLease(modelRoot: root, modelID: id)
        }, auditComputePlan: false)
        return try Self(configuration: expanded, predictor: components.predictor, preparer: components.preparer, store: store)
    }
    public func run(onListening: @escaping @Sendable (Int) async -> Void) async throws {
        try await server.run(onServerRunning: onListening)
    }
    public func load() async throws { _ = try await server.lifecycle.load() }
    public func unload() async throws { _ = try await server.lifecycle.unload() }
    public func apply(_ overrides: ConfigurationOverrides) async throws { _ = try await server.settings.apply(overrides) }
    public func snapshot() async throws -> AppSnapshot {
        let settings = await server.settings.current()
        let stats = await server.lifecycle.statistics()
        var requested = settings.configuration
        if !settings.restartRequired.isEmpty {
            let saved = try await io.run { [store] in
                var config = try store.load()
                config.modelRoot = store.expandedRoot(config.modelRoot)
                return config
            }
            requested = SettingsDraft.desired(effective: settings.configuration, persisted: saved,
                                               restartRequired: settings.restartRequired)
        }
        return AppSnapshot(state: stats.runtime.state, queueDepth: stats.runtime.queueDepth,
            inFlight: stats.runtime.inFlight, preparing: stats.runtime.preparing,
            residentBytes: stats.runtime.residentBytes, windowCount: stats.windowCount,
            p50NS: stats.p50NS, p95NS: stats.p95NS, effective: settings.configuration,
            desired: requested, restartRequired: settings.restartRequired)
    }
}
