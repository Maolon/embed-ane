import Dispatch
import Foundation

final class CoreAIVisionTower {
    private let function: any CoreAIFunction
    let loadNS: UInt64

    init(
        url: URL,
        loader: any CoreAIAssetLoading = defaultCoreAILoader(),
        modelID: String,
        options: CoreAISpecializationOptions = .init(),
        bookmarkStore: any CoreAIBookmarkStoring = CoreAIBookmarkStore()
    ) throws {
        let start = DispatchTime.now().uptimeNanoseconds
        let loaded = try CoreAIAsyncBridge.runSync {
            try await loader.loadModel(
                assetURL: url,
                expectedFunction: "tower",
                modelID: modelID,
                assetName: "vision_tower",
                options: options,
                bookmarkStore: bookmarkStore
            )
        }
        let fn = loaded.function
        try CoreAIABI.validateTower(
            functionName: fn.name,
            inputs: fn.inputDescriptions,
            outputs: fn.outputDescriptions
        )
        self.function = fn
        self.loadNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start)
    }

    func predict(_ input: VisionPreparedInput) throws -> VisionTowerResult {
        let abi = VisionABIConstants.wemm
        let inputs: [String: CoreAITensor] = [
            "patches": CoreAITensor(shape: [abi.patchCapacity, abi.framePatchDimension], fp16: input.firstFramePatches),
            "abs_pos": CoreAITensor(shape: [abi.patchCapacity, abi.hiddenDimension], fp16: input.absolutePositions),
            "key_mask": CoreAITensor(shape: [abi.patchCapacity], fp16: input.keyMask),
            "cos": CoreAITensor(shape: [abi.patchCapacity, abi.visionHeadDimension], fp16: input.rotary.cosine),
            "sin": CoreAITensor(shape: [abi.patchCapacity, abi.visionHeadDimension], fp16: input.rotary.sine)
        ]
        let start = DispatchTime.now().uptimeNanoseconds
        let fn = self.function
        let out = try CoreAIAsyncBridge.runSync { [inputs, fn] in
            try await fn.run(inputs: inputs)
        }
        let predictionNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start)
        guard let tensor = out["visual_tokens"], let fp16 = tensor.fp16Values else {
            throw EmbedANEError.verification(path: "vision_tower", reason: "Missing visual_tokens output.")
        }
        let count = input.grid.tokenCount
        guard tensor.shape == [abi.tokenCapacity, ModelABI.dimension],
              fp16.count == abi.tokenCapacity * ModelABI.dimension,
              (1...abi.tokenCapacity).contains(count) else {
            throw EmbedANEError.verification(path: "vision_tower", reason: "Invalid tower output shape or token count.")
        }
        let sliced = Array(fp16.prefix(count * ModelABI.dimension))
        let tokens = try VisionTokens(values: sliced, count: count)
        return VisionTowerResult(tokens: tokens, predictionNS: predictionNS)
    }
}

public final class CoreAIMultimodalPredictor: EmbeddingPredictor, EmbeddingPreparer, MultimodalEmbeddingPredictor, Sendable {
    private let worker: SerialEmbeddingWorker

    public init(
        paths: MultimodalModelPaths,
        resizeMode: VisionResizeMode = .smart,
        loader: any CoreAIAssetLoading = defaultCoreAILoader(),
        bookmarkStore: any CoreAIBookmarkStoring = CoreAIBookmarkStore(),
        options: CoreAISpecializationOptions = .init()
    ) {
        worker = SerialEmbeddingWorker {
            CoreAIMultimodalBackend(paths: paths, resizeMode: resizeMode, loader: loader, bookmarkStore: bookmarkStore, options: options)
        }
    }

    public func load() async throws -> LoadReport {
        try await worker.load()
    }

    public func loadVision() async throws -> MultimodalLoadReport {
        try await worker.perform { backend in
            guard let backend = backend as? CoreAIMultimodalBackend else { throw EmbedANEError.modelNotLoaded }
            return try backend.loadResources()
        }
    }

    public func prepare(_ texts: [String]) async throws -> PredictRequest {
        try await worker.prepare(texts)
    }

    public func predict(_ request: PredictRequest) async throws -> PredictResult {
        try await worker.predict(request)
    }

    public func embed(imageData: Data?, text: String) async throws -> MultimodalPrediction {
        try await worker.perform { backend in
            guard let backend = backend as? CoreAIMultimodalBackend else { throw EmbedANEError.modelNotLoaded }
            return try backend.embed(imageData: imageData, text: text)
        }
    }

    public func predictImage(_ request: ImageEmbeddingRequest) async throws -> ImageEmbeddingResult {
        guard let imageData = request.imageData else {
            throw EmbedANEError.invalidRequest("Video embedding requires engine_backend: coreml.", param: "input")
        }
        let result = try await embed(imageData: imageData, text: request.text)
        return .init(embedding: result.embedding, promptTokens: result.promptTokens, tokenizeNS: result.tokenizeNS,
                     tableLookupNS: result.tableLookupNS, perChunkNS: result.perChunkNS)
    }

    public func unload() async throws -> UnloadReport {
        try await worker.unload()
    }
}

private final class CoreAIMultimodalBackend: BlockingEmbeddingBackend, BlockingEmbeddingPreparer {
    private let paths: MultimodalModelPaths
    private let resizeMode: VisionResizeMode
    private let loader: any CoreAIAssetLoading
    private let bookmarkStore: any CoreAIBookmarkStoring
    private let options: CoreAISpecializationOptions
    private var tokenizer: LocalAssetTokenizer?
    private var preprocessor: VisionPreprocessor?
    private var tower: CoreAIVisionTower?
    private var cascade: CoreAICascadeEngine?

    init(
        paths: MultimodalModelPaths,
        resizeMode: VisionResizeMode,
        loader: any CoreAIAssetLoading,
        bookmarkStore: any CoreAIBookmarkStoring,
        options: CoreAISpecializationOptions
    ) {
        self.paths = paths
        self.resizeMode = resizeMode
        self.loader = loader
        self.bookmarkStore = bookmarkStore
        self.options = options
    }

    deinit {
        clear()
    }

    func load() throws -> LoadReport {
        try loadResources().decoder
    }

    func loadResources() throws -> MultimodalLoadReport {
        clear()
        let start = DispatchTime.now().uptimeNanoseconds
        do {
            return try autoreleasepool {
                let tokenizer = try LocalAssetTokenizer(assets: paths.tokenizerDirectory)
                let abi = VisionABIConstants.wemm
                for (spelling, id) in [("<|vision_start|>", abi.visionStartToken), ("<|image_pad|>", abi.imageToken), ("<|vision_end|>", abi.visionEndToken)] {
                    guard try tokenizer.encodeContent(spelling) == [id] else {
                        throw EmbedANEError.verification(path: "tokenizer.json", reason: "Incorrect token id for \(spelling).")
                    }
                }
                let preprocessor = try VisionPreprocessor(positionTableURL: paths.positionTable, resizeMode: resizeMode)
                let table = try MappedEmbeddingTable(url: paths.embeddingTable)
                let modelID = paths.tokenizerDirectory.lastPathComponent
                let tower = try CoreAIVisionTower(
                    url: paths.tower,
                    loader: loader,
                    modelID: modelID,
                    options: options,
                    bookmarkStore: bookmarkStore
                )
                let urls = paths.chunks
                let loadedChunks = try CoreAIChunkLoad.parallel(
                    loader: loader,
                    urlAt: { urls[$0] },
                    modelID: modelID,
                    options: options,
                    bookmarkStore: bookmarkStore,
                    validate: { index, loaded in
                        try CoreAIABI.validateChunk(
                            chunk: index,
                            functionName: loaded.function.name,
                            inputs: loaded.function.inputDescriptions,
                            outputs: loaded.function.outputDescriptions
                        )
                    }
                )
                let functions: [any CoreAIFunction] = loadedChunks.map(\.function)
                let timings: [UInt64] = loadedChunks.map(\.loadNS)
                self.tokenizer = tokenizer
                self.preprocessor = preprocessor
                self.tower = tower
                cascade = CoreAICascadeEngine(functions: functions, table: table)
                return .init(
                    towerNS: tower.loadNS,
                    totalNS: elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start),
                    decoder: .init(perChunkNS: timings, residentBytes: ProcessMemory.residentBytes(), computePlanChecked: false)
                )
            }
        } catch {
            clear()
            throw error
        }
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
        return try autoreleasepool {
            let start = DispatchTime.now().uptimeNanoseconds
            let prepared = try imageData.map { try preprocessor.prepare(imageData: $0) }
            let preprocessNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start)
            let tokenizeStart = DispatchTime.now().uptimeNanoseconds
            let input = try MultimodalTokenization.prepare(text: text, grid: prepared?.grid, tokenizer: tokenizer)
            let tokenizeNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: tokenizeStart)
            let visual = try prepared.map { try tower.predict($0) }
            let result = try cascade.predict(input: input, grid: prepared?.grid, tokens: visual?.tokens)
            guard let embedding = result.embeddings.first else { throw EmbedANEError.failed("Missing multimodal embedding.") }
            return .init(
                embedding: embedding,
                promptTokens: input.nTokens,
                imageTokens: prepared?.grid.tokenCount ?? 0,
                grid: prepared?.grid,
                preprocessNS: preprocessNS,
                tokenizeNS: tokenizeNS,
                towerNS: visual?.predictionNS ?? 0,
                tableLookupNS: result.tableLookupNS,
                perChunkNS: result.perChunkNS,
                totalNS: elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start)
            )
        }
    }

    func unload() throws -> UnloadReport {
        clear()
        return .init(residentBytes: ProcessMemory.residentBytes())
    }

    private func clear() {
        autoreleasepool {
            cascade = nil
            tower = nil
            preprocessor = nil
            tokenizer = nil
        }
    }

    private func loadChunk(url: URL, chunk: Int, modelID: String) throws -> CoreAILoadedAsset {
        let loader = self.loader
        let bookmarkStore = self.bookmarkStore
        let options = self.options
        do {
            return try CoreAIAsyncBridge.runSync {
                try await loader.loadModel(
                    assetURL: url,
                    expectedFunction: "chunk\(chunk)",
                    modelID: modelID,
                    assetName: "chunk\(chunk)",
                    options: options,
                    bookmarkStore: bookmarkStore
                )
            }
        } catch {
            if let aneError = error as? EmbedANEError { throw aneError }
            throw EmbedANEError.failed("CoreAI chunk \(chunk) load failed: \(error)")
        }
    }
}
