import EmbedANECore
import Foundation

public enum IdleTimeoutPreset: String, CaseIterable, Sendable, Identifiable, Equatable {
    case fiveMinutes = "300"
    case thirtyMinutes = "1800"
    case oneHour = "3600"
    case custom = "custom"

    public var id: String { rawValue }
    public var label: String {
        switch self {
        case .fiveMinutes: "After 5 minutes"
        case .thirtyMinutes: "After 30 minutes"
        case .oneHour: "After 1 hour"
        case .custom: "Custom…"
        }
    }
    public var seconds: Double? {
        switch self {
        case .fiveMinutes: 300
        case .thirtyMinutes: 1800
        case .oneHour: 3600
        case .custom: nil
        }
    }
    public static func from(seconds: Double) -> IdleTimeoutPreset {
        if seconds == 300 { return .fiveMinutes }
        if seconds == 1800 { return .thirtyMinutes }
        if seconds == 3600 { return .oneHour }
        return .custom
    }
}

/// Editing is local. Polling never overwrites an in-progress form, and Apply
/// sends only changed fields so unrelated HTTP settings updates are preserved.
public struct SettingsDraft: Equatable, Sendable {
    public let baseline: ServiceConfiguration
    public var port: String
    public var modelID: String
    public var modelRoot: String
    public var idleTimeoutS: String
    public var autoLoad: Bool
    public var maxQueueDepth: String
    public var maxBatch: String
    public var computeUnits: ComputeUnitsSetting

    public init(_ configuration: ServiceConfiguration) {
        baseline = configuration
        port = String(configuration.port); modelID = configuration.modelID
        modelRoot = configuration.modelRoot; idleTimeoutS = String(configuration.idleTimeoutS)
        autoLoad = configuration.autoLoad
        maxQueueDepth = String(configuration.maxQueueDepth); maxBatch = String(configuration.maxBatch)
        computeUnits = configuration.computeUnits
    }
    public func overrides() throws -> ConfigurationOverrides {
        guard let port = Int(port), let idle = Double(idleTimeoutS),
              let queue = Int(maxQueueDepth), let batch = Int(maxBatch) else {
            throw EmbedANEError.invalidRequest("Port, timeout, queue depth and batch size must be numbers.", param: "settings")
        }
        let configuration = ServiceConfiguration(port: port, modelID: modelID, modelRoot: modelRoot,
            idleTimeoutS: idle, autoLoad: autoLoad, maxQueueDepth: queue, maxBatch: batch, computeUnits: computeUnits)
        try configuration.validate()
        return ConfigurationOverrides(
            port: port == baseline.port ? nil : port,
            modelID: modelID == baseline.modelID ? nil : modelID,
            modelRoot: modelRoot == baseline.modelRoot ? nil : modelRoot,
            idleTimeoutS: idle == baseline.idleTimeoutS ? nil : idle,
            autoLoad: autoLoad == baseline.autoLoad ? nil : autoLoad,
            maxQueueDepth: queue == baseline.maxQueueDepth ? nil : queue,
            maxBatch: batch == baseline.maxBatch ? nil : batch,
            computeUnits: computeUnits == baseline.computeUnits ? nil : computeUnits)
    }
    public var hasChanges: Bool { self != SettingsDraft(baseline) }

    public var idlePreset: IdleTimeoutPreset {
        get {
            guard let val = Double(idleTimeoutS), val > 0 else { return .custom }
            return IdleTimeoutPreset.from(seconds: val)
        }
        set {
            if let sec = newValue.seconds {
                idleTimeoutS = String(Int(sec))
            }
        }
    }

    public static let standardComputeUnitsLabel = "Neural Engine"

    public var computeUnitsLabel: String {
        switch computeUnits {
        case .cpuAndNE: Self.standardComputeUnitsLabel
        default: computeUnits.appLabel
        }
    }

    public var isComputeUnitsSupported: Bool {
        computeUnits == .cpuAndNE
    }

    public var requiresComputeUnitsReset: Bool {
        !isComputeUnitsSupported
    }

    public var computeUnitsWarning: String? {
        if isComputeUnitsSupported { return nil }
        return "This model requires CPU + Neural Engine. Configured '\(computeUnits.appLabel)' is unsupported and prevents startup."
    }

    public mutating func resetComputeUnits() {
        computeUnits = .cpuAndNE
    }

    /// SettingsController's effective values are authoritative. Disk supplies
    /// only the values explicitly marked pending, not lower-precedence defaults.
    public static func desired(effective: ServiceConfiguration, persisted: ServiceConfiguration,
                               restartRequired: [String]) -> ServiceConfiguration {
        var desired = effective
        for field in restartRequired {
            switch field {
            case "port": desired.port = persisted.port
            case "model_id": desired.modelID = persisted.modelID
            case "model_root": desired.modelRoot = persisted.modelRoot
            case "max_batch": desired.maxBatch = persisted.maxBatch
            case "compute_units": desired.computeUnits = persisted.computeUnits
            case "vision_resize_mode": desired.visionResizeMode = persisted.visionResizeMode
            case "vision_tower_path": desired.visionTowerPath = persisted.visionTowerPath
            case "vision_extrope_chunks_dir": desired.visionExtropeChunksDirectory = persisted.visionExtropeChunksDirectory
            case "vision_position_table_path": desired.visionPositionTablePath = persisted.visionPositionTablePath
            default: break
            }
        }
        return desired
    }
}
