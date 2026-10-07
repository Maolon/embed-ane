import Foundation

/// Two-runtime composition (injected predictors, CoreAI): the plain text path, while image work
/// uses the injected vision artifacts. LifecycleActor owns admission for BOTH. The
/// plain runtime's installation lease is retained until vision is unloaded too.
public actor ServingMultimodalRuntime: MultimodalEmbeddingPredictor, EmbeddingPreparer {
    private let text: any EmbeddingPredictor
    private let preparer: any EmbeddingPreparer
    private let vision: any MultimodalEmbeddingPredictor
    private var loading = false

    public init(text: any EmbeddingPredictor, preparer: any EmbeddingPreparer, vision: any MultimodalEmbeddingPredictor) {
        self.text = text; self.preparer = preparer; self.vision = vision
    }

    public func load() async throws -> LoadReport {
        loading = true
        defer { loading = false }
        do {
            // A retry after failed inference may still own vision resources.
            // Release them before text.load() can replace its installation lease.
            _ = try await vision.unload()
            let plain = try await text.load()
            let image = try await vision.load()
            guard plain.perChunkNS.count == 6, image.perChunkNS.count == 6 else {
                throw EmbedANEError.abiMismatch(chunk: 0, reason: "Both decoders require six load timings.")
            }
            let timings = zip(plain.perChunkNS, image.perChunkNS).map { a, b in
                let sum = a.addingReportingOverflow(b)
                return sum.overflow ? UInt64.max : sum.partialValue
            }
            // Aggregate decoder timings, not a residency claim. Tower load cost
            // is included in the awaited load, before opening the listener.
            return .init(perChunkNS: timings, residentBytes: image.residentBytes,
                         computePlanChecked: false, visionLoaded: true)
        } catch {
            // If vision release itself fails, retain the protecting text lease.
            _ = try? await unload()
            throw error
        }
    }

    public func prepare(_ texts: [String]) async throws -> PredictRequest {
        guard !loading else { throw EmbedANEError.loading }
        return try await preparer.prepare(texts)
    }

    public func predict(_ request: PredictRequest) async throws -> PredictResult {
        try await text.predict(request)
    }

    public func predictImage(_ request: ImageEmbeddingRequest) async throws -> ImageEmbeddingResult {
        try await vision.predictImage(request)
    }

    public func unload() async throws -> UnloadReport {
        _ = try await vision.unload()
        return try await text.unload()
    }
}

public enum VisionServingArtifacts {
    /// Readability is deliberately NOT bundle verification/provenance.
    public static func paths(_ configuration: ServiceConfiguration) throws -> MultimodalModelPaths? {
        try configuration.validate()
        guard let tower = configuration.visionTowerPath,
              let chunksDirectory = configuration.visionExtropeChunksDirectory,
              let positions = configuration.visionPositionTablePath else { return nil }
        guard [tower, chunksDirectory, positions].allSatisfy({ $0.hasPrefix("/") }) else {
            throw EmbedANEError.invalidRequest("Resolve vision paths before constructing the server.", param: "vision_tower_path")
        }
        let directory = URL(fileURLWithPath: chunksDirectory, isDirectory: true)
        _ = try SecureDirectory(directory)
        let chunks: [URL]
        if FileManager.default.fileExists(atPath: directory.appendingPathComponent("chunk0.aimodel").path) {
            chunks = (0..<6).map { directory.appendingPathComponent("chunk\($0).aimodel", isDirectory: true) }
        } else {
            chunks = (0..<6).map { directory.appendingPathComponent("chunk\($0).mlmodelc", isDirectory: true) }
        }
        let towerURL = URL(fileURLWithPath: tower, isDirectory: true)
        for model in [towerURL] + chunks { try checkDirectory(model) }
        let positionURL = URL(fileURLWithPath: positions)
        let positionDirectory = try SecureDirectory(positionURL.deletingLastPathComponent())
        let handle = FileHandle(fileDescriptor: try positionDirectory.openFile(positionURL.lastPathComponent), closeOnDealloc: true)
        try handle.close()
        let bundle = URL(fileURLWithPath: configuration.modelRoot, isDirectory: true).appendingPathComponent(configuration.modelID)
        return try .init(tower: towerURL, chunks: chunks, tokenizerDirectory: bundle,
                         embeddingTable: bundle.appendingPathComponent("embed_table.fp16.npy"), positionTable: positionURL)
    }

    private static func checkDirectory(_ url: URL) throws {
        let directory = try SecureDirectory(url)
        let keys: Set<URLResourceKey> = [.isSymbolicLinkKey, .isDirectoryKey, .isRegularFileKey]
        let entries = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: Array(keys))
        guard !entries.isEmpty else { throw EmbedANEError.verification(path: url.path, reason: "Empty model directory.") }
        for child in entries {
            let attributes = try child.resourceValues(forKeys: keys)
            guard attributes.isSymbolicLink != true else { throw EmbedANEError.unsafePath(child.path) }
            if attributes.isDirectory == true { try checkDirectory(child) }
            else {
                guard attributes.isRegularFile == true else { throw EmbedANEError.unsafePath(child.path) }
                let handle = FileHandle(fileDescriptor: try directory.openFile(child.lastPathComponent), closeOnDealloc: true)
                try handle.close()
            }
        }
    }
}

public struct ServingRuntimeComponents: Sendable {
    public let predictor: any EmbeddingPredictor
    public let preparer: any EmbeddingPreparer
    public let isMultimodal: Bool
    public let verification: (@Sendable () -> VerificationReport?)?
    public let audit: (@Sendable () -> [ChunkComputePlanAudit])?

    public init(predictor: any EmbeddingPredictor,
                preparer: any EmbeddingPreparer,
                isMultimodal: Bool = false,
                verification: (@Sendable () -> VerificationReport?)? = nil,
                audit: (@Sendable () -> [ChunkComputePlanAudit])? = nil) {
        self.predictor = predictor
        self.preparer = preparer
        self.isMultimodal = isMultimodal
        self.verification = verification
        self.audit = audit
    }
}

public enum ServingRuntimeFactory {
    public static func make(
        configuration: ServiceConfiguration,
        acquireLease: @escaping @Sendable () throws -> any Sendable,
        auditComputePlan: Bool = true,
        makeTextPredictor: (@Sendable (ServiceConfiguration, @escaping @Sendable () throws -> any Sendable) throws -> (any EmbeddingPredictor & EmbeddingPreparer))? = nil,
        makeVisionPredictor: (@Sendable (MultimodalModelPaths) throws -> any MultimodalEmbeddingPredictor)? = nil,
        makeVisionPredictorWithMode: (@Sendable (MultimodalModelPaths, VisionResizeMode) throws -> any MultimodalEmbeddingPredictor)? = nil
    ) throws -> ServingRuntimeComponents {
        // A verified multimodal bundle on Core ML: one cascade serves text,
        // image and video, protected by the installation lease.
        if makeTextPredictor == nil, makeVisionPredictor == nil, makeVisionPredictorWithMode == nil,
           configuration.engineBackend == .coreml, let visionPaths = try VisionServingArtifacts.paths(configuration) {
            let bundle = URL(fileURLWithPath: configuration.modelRoot, isDirectory: true)
                .appendingPathComponent(configuration.modelID, isDirectory: true)
            let unified = MultimodalPredictor(paths: visionPaths, resizeMode: configuration.visionResizeMode,
                installation: .init(bundle: bundle, modelID: configuration.modelID, acquireLease: acquireLease))
            return ServingRuntimeComponents(predictor: unified, preparer: unified, isMultimodal: true,
                                            verification: { unified.verificationReport() }, audit: { [] })
        }
        let textRuntime: any EmbeddingPredictor & EmbeddingPreparer
        if let makeTextPredictor {
            textRuntime = try makeTextPredictor(configuration, acquireLease)
        } else {
            switch configuration.engineBackend {
            case .coreml:
                textRuntime = try CoreMLRuntime(configuration: configuration, acquireLease: acquireLease,
                                                 auditComputePlan: auditComputePlan)
            case .coreai:
                textRuntime = try CoreAIRuntime(configuration: configuration, acquireLease: acquireLease)
            }
        }
        if let visionPaths = try VisionServingArtifacts.paths(configuration) {
            let vision: any MultimodalEmbeddingPredictor
            if let makeVisionPredictorWithMode {
                vision = try makeVisionPredictorWithMode(visionPaths, configuration.visionResizeMode)
            } else if let makeVisionPredictor {
                vision = try makeVisionPredictor(visionPaths)
            } else {
                switch configuration.engineBackend {
                case .coreml:
                    vision = MultimodalPredictor(paths: visionPaths, resizeMode: configuration.visionResizeMode)
                case .coreai:
                    vision = CoreAIMultimodalPredictor(paths: visionPaths, resizeMode: configuration.visionResizeMode)
                }
            }
            let combined = ServingMultimodalRuntime(text: textRuntime, preparer: textRuntime, vision: vision)
            return ServingRuntimeComponents(
                predictor: combined,
                preparer: combined,
                isMultimodal: true,
                verification: {
                    (textRuntime as? CoreMLRuntime)?.verificationReport() ?? (textRuntime as? CoreAIRuntime)?.verificationReport()
                },
                audit: {
                    (textRuntime as? CoreMLRuntime)?.computePlanAudit() ?? (textRuntime as? CoreAIRuntime)?.computePlanAudit() ?? []
                }
            )
        } else {
            return ServingRuntimeComponents(
                predictor: textRuntime,
                preparer: textRuntime,
                isMultimodal: false,
                verification: {
                    (textRuntime as? CoreMLRuntime)?.verificationReport() ?? (textRuntime as? CoreAIRuntime)?.verificationReport()
                },
                audit: {
                    (textRuntime as? CoreMLRuntime)?.computePlanAudit() ?? (textRuntime as? CoreAIRuntime)?.computePlanAudit() ?? []
                }
            )
        }
    }
}
