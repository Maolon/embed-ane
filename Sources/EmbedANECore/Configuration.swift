import Foundation
import Yams

public enum ComputeUnitsSetting: String, Codable, Sendable, CaseIterable {
    case cpuAndNE = "cpu_and_ne", cpuOnly = "cpu_only", cpuAndGPU = "cpu_and_gpu", all

    public var appLabel: String {
        switch self {
        case .cpuAndNE: "CPU + Neural Engine"
        case .cpuOnly: "CPU only"
        case .cpuAndGPU: "CPU + GPU"
        case .all: "All compute units"
        }
    }
}

public enum EngineBackendSetting: String, Codable, Sendable, CaseIterable {
    case coreml
    case coreai
}

public struct ServiceConfiguration: Codable, Sendable, Equatable {
    public var configVersion: Int
    public var port: Int
    public var modelID: String
    public var modelRoot: String
    public var idleTimeoutS: Double
    public var autoLoad: Bool
    public var maxQueueDepth: Int
    public var maxBatch: Int
    public var computeUnits: ComputeUnitsSetting
    public var engineBackend: EngineBackendSetting
    public var visionTowerPath: String?
    public var visionExtropeChunksDirectory: String?
    public var visionPositionTablePath: String?
    public var visionResizeMode: VisionResizeMode
    enum CodingKeys: String, CodingKey, CaseIterable {
        case configVersion = "config_version", port, modelID = "model_id", modelRoot = "model_root"
        case idleTimeoutS = "idle_timeout_s", autoLoad = "auto_load", maxQueueDepth = "max_queue_depth", maxBatch = "max_batch"
        case computeUnits = "compute_units"
        case engineBackend = "engine_backend"
        case visionTowerPath = "vision_tower_path", visionExtropeChunksDirectory = "vision_extrope_chunks_dir"
        case visionPositionTablePath = "vision_position_table_path"
        case visionResizeMode = "vision_resize_mode"
    }
    public init(port: Int = 8080, modelID: String = ModelABI.defaultModelID,
                modelRoot: String = "~/.embed-ane/models", idleTimeoutS: Double = 0,
                autoLoad: Bool = true,
                maxQueueDepth: Int = 16, maxBatch: Int = 8, computeUnits: ComputeUnitsSetting = .cpuAndNE,
                engineBackend: EngineBackendSetting = .coreml,
                visionTowerPath: String? = nil, visionExtropeChunksDirectory: String? = nil,
                visionPositionTablePath: String? = nil,
                visionResizeMode: VisionResizeMode = .smart) {
        configVersion = 1; self.port = port; self.modelID = modelID; self.modelRoot = modelRoot
        self.idleTimeoutS = idleTimeoutS; self.autoLoad = autoLoad; self.maxQueueDepth = maxQueueDepth
        self.maxBatch = maxBatch; self.computeUnits = computeUnits
        self.engineBackend = engineBackend
        self.visionTowerPath = visionTowerPath; self.visionExtropeChunksDirectory = visionExtropeChunksDirectory
        self.visionPositionTablePath = visionPositionTablePath
        self.visionResizeMode = visionResizeMode
    }
    public init(from decoder: any Swift.Decoder) throws {
        try StrictCoding.rejectUnknown(decoder, allowed: CodingKeys.allCases.map(\.rawValue))
        let c = try decoder.container(keyedBy: CodingKeys.self)
        configVersion = try c.decode(Int.self, forKey: .configVersion)
        // Only absence means default; an explicit null is a schema error.
        port = try c.contains(.port) ? c.decode(Int.self, forKey: .port) : 8080
        modelID = try c.contains(.modelID) ? c.decode(String.self, forKey: .modelID) : ModelABI.defaultModelID
        modelRoot = try c.contains(.modelRoot) ? c.decode(String.self, forKey: .modelRoot) : "~/.embed-ane/models"
        idleTimeoutS = try c.contains(.idleTimeoutS) ? c.decode(Double.self, forKey: .idleTimeoutS) : 0
        autoLoad = try c.contains(.autoLoad) ? c.decode(Bool.self, forKey: .autoLoad) : true
        maxQueueDepth = try c.contains(.maxQueueDepth) ? c.decode(Int.self, forKey: .maxQueueDepth) : 16
        maxBatch = try c.contains(.maxBatch) ? c.decode(Int.self, forKey: .maxBatch) : 8
        computeUnits = try c.contains(.computeUnits) ? c.decode(ComputeUnitsSetting.self, forKey: .computeUnits) : .cpuAndNE
        engineBackend = try c.contains(.engineBackend) ? c.decode(EngineBackendSetting.self, forKey: .engineBackend) : .coreml
        visionTowerPath = try c.contains(.visionTowerPath) ? c.decode(String.self, forKey: .visionTowerPath) : nil
        visionExtropeChunksDirectory = try c.contains(.visionExtropeChunksDirectory) ? c.decode(String.self, forKey: .visionExtropeChunksDirectory) : nil
        visionPositionTablePath = try c.contains(.visionPositionTablePath) ? c.decode(String.self, forKey: .visionPositionTablePath) : nil
        visionResizeMode = try c.contains(.visionResizeMode) ? c.decode(VisionResizeMode.self, forKey: .visionResizeMode) : .smart
        try validate()
    }
    public func encode(to encoder: any Swift.Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(configVersion, forKey: .configVersion)
        try c.encode(port, forKey: .port)
        try c.encode(modelID, forKey: .modelID)
        try c.encode(modelRoot, forKey: .modelRoot)
        try c.encode(idleTimeoutS, forKey: .idleTimeoutS)
        try c.encode(autoLoad, forKey: .autoLoad)
        try c.encode(maxQueueDepth, forKey: .maxQueueDepth)
        try c.encode(maxBatch, forKey: .maxBatch)
        try c.encode(computeUnits, forKey: .computeUnits)
        try c.encode(engineBackend, forKey: .engineBackend)
        try c.encodeIfPresent(visionTowerPath, forKey: .visionTowerPath)
        try c.encodeIfPresent(visionExtropeChunksDirectory, forKey: .visionExtropeChunksDirectory)
        try c.encodeIfPresent(visionPositionTablePath, forKey: .visionPositionTablePath)
        if visionTowerPath != nil || visionResizeMode != .smart {
            try c.encode(visionResizeMode, forKey: .visionResizeMode)
        }
    }
    public func validate() throws {
        guard configVersion == 1 else { throw EmbedANEError.invalidRequest("config_version must be 1.", param: "config_version") }
        guard (1...65535).contains(port) else { throw EmbedANEError.invalidRequest("port must be 1...65535.", param: "port") }
        do { try ModelSpec.validateID(modelID) }
        catch { throw EmbedANEError.invalidRequest("Invalid model_id.", param: "model_id") }
        guard !modelRoot.isEmpty, !modelRoot.contains("\0"),
              modelRoot.hasPrefix("/") || modelRoot == "~" || modelRoot.hasPrefix("~/") else {
            throw EmbedANEError.invalidRequest("model_root must be absolute or home-relative.", param: "model_root")
        }
        guard idleTimeoutS.isFinite, (0...1_000_000_000).contains(idleTimeoutS) else {
            throw EmbedANEError.invalidRequest("idle_timeout_s must be finite and in 0...1000000000.", param: "idle_timeout_s")
        }
        guard maxQueueDepth > 0 else { throw EmbedANEError.invalidRequest("max_queue_depth must be positive.", param: "max_queue_depth") }
        guard (1...8).contains(maxBatch) else { throw EmbedANEError.invalidRequest("max_batch must be 1...8.", param: "max_batch") }
        let paths = [("vision_tower_path", visionTowerPath), ("vision_extrope_chunks_dir", visionExtropeChunksDirectory),
                     ("vision_position_table_path", visionPositionTablePath)]
        let present = paths.compactMap(\.1).count
        guard present == 0 || present == 3 else {
            throw EmbedANEError.invalidRequest("Configure vision_tower_path, vision_extrope_chunks_dir and vision_position_table_path together, or omit all three.", param: "vision_tower_path")
        }
        for (key, path) in paths {
            if let path, path.isEmpty || path.contains("\0") || !(path.hasPrefix("/") || path.hasPrefix("~/")) {
                throw EmbedANEError.invalidRequest("\(key) must be absolute or home-relative.", param: key)
            }
        }
    }
    public func applying(_ patch: ConfigurationOverrides) throws -> Self {
        var value = self
        if let port = patch.port { value.port = port }
        if let modelID = patch.modelID { value.modelID = modelID }
        if let modelRoot = patch.modelRoot { value.modelRoot = modelRoot }
        if let idleTimeoutS = patch.idleTimeoutS { value.idleTimeoutS = idleTimeoutS }
        if let autoLoad = patch.autoLoad { value.autoLoad = autoLoad }
        if let maxQueueDepth = patch.maxQueueDepth { value.maxQueueDepth = maxQueueDepth }
        if let maxBatch = patch.maxBatch { value.maxBatch = maxBatch }
        if let computeUnits = patch.computeUnits { value.computeUnits = computeUnits }
        if let engineBackend = patch.engineBackend { value.engineBackend = engineBackend }
        if let path = patch.visionTowerPath { value.visionTowerPath = path }
        if let path = patch.visionExtropeChunksDirectory { value.visionExtropeChunksDirectory = path }
        if let path = patch.visionPositionTablePath { value.visionPositionTablePath = path }
        if let mode = patch.visionResizeMode { value.visionResizeMode = mode }
        try value.validate()
        return value
    }
}

public struct ConfigurationOverrides: Codable, Sendable, Equatable {
    public var port: Int?
    public var modelID: String?
    public var modelRoot: String?
    public var idleTimeoutS: Double?
    public var autoLoad: Bool?
    public var maxQueueDepth: Int?
    public var maxBatch: Int?
    public var computeUnits: ComputeUnitsSetting?
    public var engineBackend: EngineBackendSetting?
    public var visionTowerPath: String?
    public var visionExtropeChunksDirectory: String?
    public var visionPositionTablePath: String?
    public var visionResizeMode: VisionResizeMode?
    enum CodingKeys: String, CodingKey, CaseIterable {
        case port, modelID = "model_id", modelRoot = "model_root", idleTimeoutS = "idle_timeout_s"
        case autoLoad = "auto_load"
        case maxQueueDepth = "max_queue_depth", maxBatch = "max_batch", computeUnits = "compute_units"
        case engineBackend = "engine_backend"
        case visionTowerPath = "vision_tower_path", visionExtropeChunksDirectory = "vision_extrope_chunks_dir"
        case visionPositionTablePath = "vision_position_table_path"
        case visionResizeMode = "vision_resize_mode"
    }
    public init(port: Int? = nil, modelID: String? = nil, modelRoot: String? = nil,
                idleTimeoutS: Double? = nil, autoLoad: Bool? = nil,
                maxQueueDepth: Int? = nil, maxBatch: Int? = nil,
                computeUnits: ComputeUnitsSetting? = nil,
                engineBackend: EngineBackendSetting? = nil,
                visionTowerPath: String? = nil,
                visionExtropeChunksDirectory: String? = nil, visionPositionTablePath: String? = nil,
                visionResizeMode: VisionResizeMode? = nil) {
        self.port = port; self.modelID = modelID; self.modelRoot = modelRoot
        self.idleTimeoutS = idleTimeoutS; self.autoLoad = autoLoad; self.maxQueueDepth = maxQueueDepth
        self.maxBatch = maxBatch; self.computeUnits = computeUnits
        self.engineBackend = engineBackend
        self.visionTowerPath = visionTowerPath; self.visionExtropeChunksDirectory = visionExtropeChunksDirectory
        self.visionPositionTablePath = visionPositionTablePath
        self.visionResizeMode = visionResizeMode
    }
    public init(from decoder: any Swift.Decoder) throws {
        try StrictCoding.rejectUnknown(decoder, allowed: CodingKeys.allCases.map(\.rawValue))
        let c = try decoder.container(keyedBy: CodingKeys.self)
        port = try c.contains(.port) ? c.decode(Int.self, forKey: .port) : nil
        modelID = try c.contains(.modelID) ? c.decode(String.self, forKey: .modelID) : nil
        modelRoot = try c.contains(.modelRoot) ? c.decode(String.self, forKey: .modelRoot) : nil
        idleTimeoutS = try c.contains(.idleTimeoutS) ? c.decode(Double.self, forKey: .idleTimeoutS) : nil
        autoLoad = try c.contains(.autoLoad) ? c.decode(Bool.self, forKey: .autoLoad) : nil
        maxQueueDepth = try c.contains(.maxQueueDepth) ? c.decode(Int.self, forKey: .maxQueueDepth) : nil
        maxBatch = try c.contains(.maxBatch) ? c.decode(Int.self, forKey: .maxBatch) : nil
        computeUnits = try c.contains(.computeUnits) ? c.decode(ComputeUnitsSetting.self, forKey: .computeUnits) : nil
        engineBackend = try c.contains(.engineBackend) ? c.decode(EngineBackendSetting.self, forKey: .engineBackend) : nil
        visionTowerPath = try c.contains(.visionTowerPath) ? c.decode(String.self, forKey: .visionTowerPath) : nil
        visionExtropeChunksDirectory = try c.contains(.visionExtropeChunksDirectory) ? c.decode(String.self, forKey: .visionExtropeChunksDirectory) : nil
        visionPositionTablePath = try c.contains(.visionPositionTablePath) ? c.decode(String.self, forKey: .visionPositionTablePath) : nil
        visionResizeMode = try c.contains(.visionResizeMode) ? c.decode(VisionResizeMode.self, forKey: .visionResizeMode) : nil
    }
}

public struct ConfigurationStore: Sendable {
    public let file: URL
    public let home: URL
    public init(file: URL? = nil, home: URL = FileManager.default.homeDirectoryForCurrentUser) {
        self.home = home
        self.file = file ?? home.appendingPathComponent(".embed-ane/config.yaml")
    }
    public func load() throws -> ServiceConfiguration {
        let storage = StateFileStorage(directory: file.deletingLastPathComponent())
        guard let data = try storage.read(file.lastPathComponent) else { return ServiceConfiguration() }
        guard let text = String(data: data, encoding: .utf8) else { throw EmbedANEError.invalidSpec("Configuration must be UTF-8.") }
        return try StrictYAML.decode(ServiceConfiguration.self, from: text)
    }
    public func resolve(cli: ConfigurationOverrides = .init(),
                        environment: [String: String] = ProcessInfo.processInfo.environment) throws -> ServiceConfiguration {
        var effective = try load()
        if let root = environment["EMBED_ANE_MODEL_ROOT"], cli.modelRoot == nil { effective.modelRoot = root }
        // Apply all higher-precedence flags before validating the resulting value.
        effective = try effective.applying(cli)
        effective = expandingPaths(effective)
        try effective.validate()
        return effective
    }
    public func expandingPaths(_ configuration: ServiceConfiguration) -> ServiceConfiguration {
        var value = configuration
        let expandedModelRoot = expandedRoot(value.modelRoot)
        value.modelRoot = expandedModelRoot
        value.visionTowerPath = value.visionTowerPath.map(expandedRoot)
        value.visionExtropeChunksDirectory = value.visionExtropeChunksDirectory.map(expandedRoot)
        value.visionPositionTablePath = value.visionPositionTablePath.map(expandedRoot)

        if value.visionTowerPath == nil && value.visionExtropeChunksDirectory == nil && value.visionPositionTablePath == nil {
            let modelDir = URL(fileURLWithPath: expandedModelRoot, isDirectory: true).appendingPathComponent(value.modelID)
            if let detected = Self.detectVisionArtifacts(in: modelDir) {
                value.visionTowerPath = detected.tower
                value.visionExtropeChunksDirectory = detected.chunksDirectory
                value.visionPositionTablePath = detected.positionTable
            }
        }
        return value
    }
    public static func detectVisionArtifacts(in modelDirectory: URL) -> (tower: String, chunksDirectory: String, positionTable: String)? {
        let fm = FileManager.default
        let posTable = modelDirectory.appendingPathComponent("vision_abs_pos.fp32.npy")
        guard fm.fileExists(atPath: posTable.path) else { return nil }

        let coreaiTowerCandidates = [
            modelDirectory.appendingPathComponent("coreai/vision_tower.aimodel"),
            modelDirectory.appendingPathComponent("vision_tower.aimodel")
        ]
        let coreaiChunkDirCandidates = [
            modelDirectory.appendingPathComponent("coreai", isDirectory: true),
            modelDirectory.appendingPathComponent("chunks", isDirectory: true),
            modelDirectory
        ]
        for tower in coreaiTowerCandidates where fm.fileExists(atPath: tower.path) {
            for dir in coreaiChunkDirCandidates {
                let chunksExist = (0..<6).allSatisfy {
                    fm.fileExists(atPath: dir.appendingPathComponent("chunk\($0).aimodel").path)
                }
                if chunksExist {
                    return (tower: tower.path, chunksDirectory: dir.path, positionTable: posTable.path)
                }
            }
        }

        // Release multimodal layout: the two-frame tower at the root and the six
        // position-input (extrope) chunks as chunks/chunk{0..5}.mlmodelc.
        let tower = modelDirectory.appendingPathComponent("vision_tower.mlmodelc")
        let chunks = modelDirectory.appendingPathComponent("chunks", isDirectory: true)
        guard fm.fileExists(atPath: tower.path),
              (0..<6).allSatisfy({ fm.fileExists(atPath: chunks.appendingPathComponent("chunk\($0).mlmodelc").path) }) else { return nil }
        return (tower: tower.path, chunksDirectory: chunks.path, positionTable: posTable.path)
    }
    public func expandedRoot(_ path: String) -> String {
        if path == "~" { return home.path }
        if path.hasPrefix("~/") { return home.appendingPathComponent(String(path.dropFirst(2))).path }
        return path
    }
    public func save(_ configuration: ServiceConfiguration) throws {
        try configuration.validate()
        let yaml = try YAMLEncoder().encode(configuration)
        try StateFileStorage(directory: file.deletingLastPathComponent())
            .write(file.lastPathComponent, data: Data(yaml.utf8))
    }
}

extension ConfigurationStore {
    public func saveIfAbsent(_ configuration: ServiceConfiguration) throws {
        try configuration.validate()
        let yaml = try YAMLEncoder().encode(configuration)
        try StateFileStorage(directory: file.deletingLastPathComponent())
            .write(file.lastPathComponent, data: Data(yaml.utf8), replace: false)
    }
}
