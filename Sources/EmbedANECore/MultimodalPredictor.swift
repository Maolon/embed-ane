import CoreML
import Dispatch
import Foundation

/// Explicit local model URLs. They are not a verified installation by themselves;
/// serving pairs them with `MultimodalInstallation` (lease + bundle verification).
public struct MultimodalModelPaths: Sendable {
    public let tower: URL
    public let chunks: [URL]
    public let tokenizerDirectory: URL
    public let embeddingTable: URL
    public let positionTable: URL
    public init(tower: URL, chunks: [URL], tokenizerDirectory: URL, embeddingTable: URL, positionTable: URL) throws {
        guard chunks.count == 6, ([tower, tokenizerDirectory, embeddingTable, positionTable] + chunks).allSatisfy(\.isFileURL) else {
            throw EmbedANEError.invalidRequest("Supply six extrope chunks and local model asset URLs.", param: "model")
        }
        self.tower = tower; self.chunks = chunks; self.tokenizerDirectory = tokenizerDirectory
        self.embeddingTable = embeddingTable; self.positionTable = positionTable
    }
}

/// A verified installation a serving predictor protects while loaded: the
/// lease is taken on the worker BEFORE bundle verification and released after
/// unload, as in the text-only runtime.
public struct MultimodalInstallation: Sendable {
    public let bundle: URL
    public let modelID: String
    public let acquireLease: @Sendable () throws -> any Sendable
    public init(bundle: URL, modelID: String, acquireLease: @escaping @Sendable () throws -> any Sendable) {
        self.bundle = bundle; self.modelID = modelID; self.acquireLease = acquireLease
    }
}

public struct MultimodalLoadReport: Codable, Sendable {
    public let towerNS: UInt64
    public let totalNS: UInt64
    public let decoder: LoadReport
}

public struct MultimodalPrediction: Codable, Sendable {
    public let embedding: [Float]
    public let promptTokens: Int
    public let imageTokens: Int
    public let grid: VisionGrid?
    public let preprocessNS: UInt64
    public let tokenizeNS: UInt64
    public let towerNS: UInt64
    public let tableLookupNS: UInt64
    public let perChunkNS: [UInt64]
    public let totalNS: UInt64
}

/// Native Swift image/text facade. All blocking CPU work, model loads, predicts,
/// and releases are serialized by the same worker used for the text runtime.
/// There is no subprocess, network image loading, or AppKit dependency.
/// The existing plain runtime is unchanged; this injected-path runtime uses
/// extrope for BOTH image+text and text-only input.
public final class MultimodalPredictor: EmbeddingPredictor, EmbeddingPreparer, Sendable {
    private final class Metadata: @unchecked Sendable {
        let lock = NSLock()
        var loading = false
        var verification: VerificationReport?
    }
    private let worker: SerialEmbeddingWorker
    private let metadata = Metadata()

    /// With an installation, this one cascade also serves text-only requests
    /// for a verified multimodal bundle (no second, plain-text cascade).
    public init(paths: MultimodalModelPaths, resizeMode: VisionResizeMode = .smart,
                installation: MultimodalInstallation? = nil) {
        let metadata = self.metadata
        worker = SerialEmbeddingWorker {
            MultimodalBackend(paths: paths, resizeMode: resizeMode, installation: installation,
                              verified: { report in metadata.lock.withLock { metadata.verification = report } })
        }
    }
    public func load() async throws -> LoadReport {
        metadata.lock.withLock { metadata.loading = true; metadata.verification = nil }
        defer { metadata.lock.withLock { metadata.loading = false } }
        return try await worker.load()
    }
    public func verificationReport() -> VerificationReport? { metadata.lock.withLock { metadata.verification } }
    public func loadVision() async throws -> MultimodalLoadReport {
        try await worker.perform { backend in
            guard let backend = backend as? MultimodalBackend else { throw EmbedANEError.modelNotLoaded }
            return try backend.loadResources()
        }
    }
    public func prepare(_ texts: [String]) async throws -> PredictRequest {
        // Do not queue a text request silently behind a slow (re)load.
        if metadata.lock.withLock({ metadata.loading }) { throw EmbedANEError.loading }
        return try await worker.prepare(texts)
    }
    public func predict(_ request: PredictRequest) async throws -> PredictResult { try await worker.predict(request) }
    public func embed(imageData: Data?, text: String) async throws -> MultimodalPrediction {
        try await worker.perform { backend in
            guard let backend = backend as? MultimodalBackend else { throw EmbedANEError.modelNotLoaded }
            return try backend.embed(imageData: imageData, text: text)
        }
    }
    /// Frames are decoded before taking the serial worker; only preprocessing
    /// and graph calls run on it.
    public func embed(videoURL: URL, text: String) async throws -> MultimodalPrediction {
        let video = try await VideoFrameReader.read(videoURL)
        return try await worker.perform { backend in
            guard let backend = backend as? MultimodalBackend else { throw EmbedANEError.modelNotLoaded }
            return try backend.embed(video: video, text: text)
        }
    }
    public func unload() async throws -> UnloadReport { try await worker.unload() }
}

private final class MultimodalBackend: BlockingEmbeddingBackend, BlockingEmbeddingPreparer {
    private let paths: MultimodalModelPaths
    private let resizeMode: VisionResizeMode
    private var tokenizer: LocalAssetTokenizer?
    private var preprocessor: VisionPreprocessor?
    private var tower: VisionTowerEngine?
    private var cascade: ExtropeCascadeEngine?
    private let installation: MultimodalInstallation?
    private let verified: @Sendable (VerificationReport) -> Void
    private var lease: (any Sendable)?

    init(paths: MultimodalModelPaths, resizeMode: VisionResizeMode, installation: MultimodalInstallation?,
         verified: @escaping @Sendable (VerificationReport) -> Void) {
        self.paths = paths; self.resizeMode = resizeMode
        self.installation = installation; self.verified = verified
    }
    deinit { clear() }
    func load() throws -> LoadReport { try loadResources().decoder }
    func loadResources() throws -> MultimodalLoadReport {
        clear()
        let start = DispatchTime.now().uptimeNanoseconds
        do {
            if let installation { lease = try installation.acquireLease() }
            return try autoreleasepool {
                let verification = try installation.map { try BundleVerifier.verify(at: $0.bundle, expectedID: $0.modelID) }
                let tokenizer = try LocalAssetTokenizer(assets: paths.tokenizerDirectory)
                let abi = VisionABIConstants.wemm
                for (spelling, id) in [("<|vision_start|>", abi.visionStartToken), ("<|image_pad|>", abi.imageToken),
                                       ("<|video_pad|>", abi.videoToken), ("<|vision_end|>", abi.visionEndToken)] {
                    guard try tokenizer.encodeContent(spelling) == [id] else {
                        throw EmbedANEError.verification(path: "tokenizer.json", reason: "Incorrect token id for \(spelling).")
                    }
                }
                let preprocessor = try VisionPreprocessor(positionTableURL: paths.positionTable, resizeMode: resizeMode)
                let table = try MappedEmbeddingTable(url: paths.embeddingTable)
                let tower = try VisionTowerEngine(url: paths.tower)
                let configuration = MLModelConfiguration(); configuration.computeUnits = .cpuAndNeuralEngine
                var models: [MLModel] = [], timings: [UInt64] = []
                for (index, url) in paths.chunks.enumerated() {
                    let chunkStart = DispatchTime.now().uptimeNanoseconds
                    let model: MLModel
                    do { model = try MLModel(contentsOf: url, configuration: configuration) }
                    catch { throw EmbedANEError.failed("Extrope chunk \(index) load failed: \(error)") }
                    try CoreMLCascadeEngine.validateDescription(model.modelDescription, chunk: index, variant: .extrope)
                    models.append(model)
                    timings.append(elapsedNS(DispatchTime.now().uptimeNanoseconds, since: chunkStart))
                }
                self.tokenizer = tokenizer; self.preprocessor = preprocessor; self.tower = tower
                cascade = ExtropeCascadeEngine(models: models, table: table)
                if let verification { verified(verification) }
                return .init(towerNS: tower.loadNS, totalNS: elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start),
                    decoder: .init(perChunkNS: timings, residentBytes: ProcessMemory.residentBytes(), computePlanChecked: false))
            }
        } catch { clear(); throw error }
    }
    func prepare(_ texts: [String]) throws -> PredictRequest {
        guard let tokenizer, cascade != nil else { throw EmbedANEError.modelNotLoaded }
        let start = DispatchTime.now().uptimeNanoseconds
        let inputs = try texts.map { try MultimodalTokenization.prepare(text: $0, grid: nil, tokenizer: tokenizer) }
        return try .init(inputs: inputs, tokenizeNS: elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start))
    }
    func predict(_ request: PredictRequest) throws -> PredictResult {
        guard let cascade else { throw EmbedANEError.modelNotLoaded }
        return try autoreleasepool {
            // All members are validated before the first graph call.
            for input in request.inputs { _ = try Mrope.positions(input: input) }
            var vectors: [[Float]] = [], timings = [UInt64](repeating: 0, count: 6)
            var lookupNS: UInt64 = 0
            for input in request.inputs {
                let result = try cascade.predict(input: input)
                vectors.append(contentsOf: result.embeddings)
                lookupNS = addingNS(lookupNS, result.tableLookupNS)
                for index in 0..<6 { timings[index] = addingNS(timings[index], result.perChunkNS[index]) }
            }
            let result = PredictResult(embeddings: vectors, tableLookupNS: lookupNS, perChunkNS: timings)
            try result.validate(for: request)
            return result
        }
    }
    func embed(imageData: Data?, text: String) throws -> MultimodalPrediction {
        guard let tokenizer, let preprocessor, let tower, let cascade else { throw EmbedANEError.modelNotLoaded }
        do {
            return try autoreleasepool {
                let start = DispatchTime.now().uptimeNanoseconds
                let prepared = try imageData.map { try preprocessor.prepare(imageData: $0) }
                let preprocessNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start)
                let tokenizeStart = DispatchTime.now().uptimeNanoseconds
                let input = try MultimodalTokenization.prepare(text: text, grid: prepared?.grid, tokenizer: tokenizer)
                let tokenizeNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: tokenizeStart)
                // Budget/placeholder validation occurs BEFORE either graph executes.
                let visual = try prepared.map { try tower.predict($0) }
                let result = try cascade.predict(input: input, grid: prepared?.grid, tokens: visual?.tokens)
                guard let embedding = result.embeddings.first else { throw EmbedANEError.failed("Missing multimodal embedding.") }
                return .init(embedding: embedding, promptTokens: input.nTokens, imageTokens: prepared?.grid.tokenCount ?? 0,
                    grid: prepared?.grid, preprocessNS: preprocessNS, tokenizeNS: tokenizeNS, towerNS: visual?.predictionNS ?? 0,
                    tableLookupNS: result.tableLookupNS, perChunkNS: result.perChunkNS,
                    totalNS: elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start))
            }
        } catch let err as EmbedANEError {
            throw err
        } catch {
            let desc = String(describing: error)
            throw EmbedANEError.failed("Multimodal prediction failed: \(desc)")
        }
    }
    /// Video and optional text: one tower call per frame-pair, each pair placed
    /// after its timestamp like upstream, then one decoder pass.
    func embed(video: DecodedVideo, text: String) throws -> MultimodalPrediction {
        guard let tokenizer, let preprocessor, let tower, let cascade else { throw EmbedANEError.modelNotLoaded }
        do {
            return try autoreleasepool {
                let abi = VisionABIConstants.wemm
                let start = DispatchTime.now().uptimeNanoseconds
                let plan = video.plan
                guard let first = video.frames.first, video.frames.count == plan.frameIndices.count else {
                    throw EmbedANEError.invalidRequest("Decoded video frames do not match the sampling plan.", param: "video")
                }
                let tokenizeStart = DispatchTime.now().uptimeNanoseconds
                let prefixes = try plan.timestamps.map { try tokenizer.encodeContent(VideoSampling.timestampText($0)) }
                let perPair = try VideoSampling.tokensPerPair(pairs: plan.pairs, textTokens: tokenizer.encodeContent(text).count,
                                                              timestampTokens: prefixes.reduce(0) { $0 + $1.count })
                var tokenizeNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: tokenizeStart)
                let size = try VideoSampling.resizedSize(frames: video.frames.count, width: first.width, height: first.height, tokensPerPair: perPair)
                let frames = try video.frames.map { try VisionPreprocessor.resize($0, to: size) }
                let prepared = try (0..<plan.pairs).map { try preprocessor.prepareResized(pair: (frames[2 * $0], frames[2 * $0 + 1])) }
                let preprocessNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start) - tokenizeNS
                let spanStart = DispatchTime.now().uptimeNanoseconds
                let spans = zip(prepared, prefixes).map { VisualSpan(grid: $0.grid, padToken: abi.videoToken, prefixIDs: $1) }
                let input = try MultimodalTokenization.prepare(text: text, spans: spans, lead: .video, tokenizer: tokenizer)
                tokenizeNS = addingNS(tokenizeNS, elapsedNS(DispatchTime.now().uptimeNanoseconds, since: spanStart))
                // Budget and placeholder validation occur BEFORE any graph executes.
                var towerNS: UInt64 = 0
                let visual = try prepared.map { pair -> VisionTokens in
                    let result = try tower.predict(pair)
                    towerNS = addingNS(towerNS, result.predictionNS)
                    return result.tokens
                }
                let result = try cascade.predict(input: input, grids: prepared.map(\.grid), tokens: visual)
                guard let embedding = result.embeddings.first else { throw EmbedANEError.failed("Missing video embedding.") }
                return .init(embedding: embedding, promptTokens: input.nTokens, imageTokens: visual.reduce(0) { $0 + $1.count },
                    grid: prepared.first?.grid, preprocessNS: preprocessNS, tokenizeNS: tokenizeNS, towerNS: towerNS,
                    tableLookupNS: result.tableLookupNS, perChunkNS: result.perChunkNS,
                    totalNS: elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start))
            }
        } catch let error as EmbedANEError {
            throw error
        } catch {
            throw EmbedANEError.failed("Video prediction failed: \(String(describing: error))")
        }
    }
    func unload() throws -> UnloadReport {
        clear()
        return .init(residentBytes: ProcessMemory.residentBytes())
    }
    private func clear() {
        autoreleasepool { cascade = nil; tower = nil; preprocessor = nil; tokenizer = nil }
        // Release installation protection only AFTER the autorelease pool drains.
        lease = nil
    }
}
