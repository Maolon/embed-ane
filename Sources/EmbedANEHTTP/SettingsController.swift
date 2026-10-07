import EmbedANECore
import Foundation

public struct EffectiveSettings: Codable, Sendable {
    public let configuration: ServiceConfiguration
    public let restartRequired: [String]
    private enum CodingKeys: String, CodingKey { case restartRequired = "restart_required" }

    public init(configuration: ServiceConfiguration, restartRequired: [String]) {
        self.configuration = configuration
        self.restartRequired = restartRequired
    }

    public init(from decoder: any Swift.Decoder) throws {
        self.configuration = try ServiceConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.restartRequired = try c.decodeIfPresent([String].self, forKey: .restartRequired) ?? []
    }

    public func encode(to encoder: any Swift.Encoder) throws {
        try configuration.encode(to: encoder)
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(restartRequired, forKey: .restartRequired)
    }
}

public actor SettingsController {
    private var effective: ServiceConfiguration
    private var desired: ServiceConfiguration
    private let lifecycle: LifecycleActor
    private let store: ConfigurationStore
    private var updating = false

    public init(configuration: ServiceConfiguration, lifecycle: LifecycleActor, store: ConfigurationStore) {
        effective = configuration; desired = configuration
        self.lifecycle = lifecycle; self.store = store
    }
    public func current() -> EffectiveSettings {
        EffectiveSettings(configuration: effective, restartRequired: restartFields())
    }
    public func apply(_ patch: ConfigurationOverrides) async throws -> EffectiveSettings {
        // A second PUT cannot interleave across the lifecycle actor hop and
        // overwrite either the durable desired config or the effective snapshot.
        guard !updating else { throw EmbedANEError.conflict("A settings update is already in progress.") }
        updating = true
        defer { updating = false }
        let candidate = store.expandingPaths(try desired.applying(patch))
        try candidate.validate()
        // Persistence failure leaves runtime settings unchanged.
        try store.save(candidate)
        try await lifecycle.updateLiveSettings(idleTimeoutS: candidate.idleTimeoutS,
                                               maxQueueDepth: candidate.maxQueueDepth,
                                               autoLoad: candidate.autoLoad)
        effective.idleTimeoutS = candidate.idleTimeoutS
        effective.maxQueueDepth = candidate.maxQueueDepth
        effective.autoLoad = candidate.autoLoad
        desired = candidate
        return current()
    }
    private func restartFields() -> [String] {
        var fields: [String] = []
        if effective.port != desired.port { fields.append("port") }
        if effective.modelID != desired.modelID { fields.append("model_id") }
        if effective.modelRoot != desired.modelRoot { fields.append("model_root") }
        if effective.maxBatch != desired.maxBatch { fields.append("max_batch") }
        if effective.computeUnits != desired.computeUnits { fields.append("compute_units") }
        if effective.engineBackend != desired.engineBackend { fields.append("engine_backend") }
        if effective.visionTowerPath != desired.visionTowerPath { fields.append("vision_tower_path") }
        if effective.visionExtropeChunksDirectory != desired.visionExtropeChunksDirectory { fields.append("vision_extrope_chunks_dir") }
        if effective.visionPositionTablePath != desired.visionPositionTablePath { fields.append("vision_position_table_path") }
        if effective.visionResizeMode != desired.visionResizeMode { fields.append("vision_resize_mode") }
        return fields.sorted()
    }
}
