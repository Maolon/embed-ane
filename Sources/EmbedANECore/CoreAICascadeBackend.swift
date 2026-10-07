import Dispatch
import Foundation

enum CoreAIVisionTokenSplicer {
    static func splice(_ tokens: VisionTokens, input: TokenizedInput, into hidden: inout [Float16]) throws {
        let positions = input.ids.prefix(input.nTokens).indices.filter { input.ids[$0] == VisionABIConstants.wemm.imageToken }
        guard let first = positions.first, positions == Array(first..<(first + tokens.count)),
              hidden.count == ModelABI.sequenceLength * ModelABI.dimension else {
            throw EmbedANEError.verification(path: "vision_splice", reason: "Expected contiguous fp16 hidden storage and one matching image-token span.")
        }
        tokens.values.withUnsafeBytes { source in
            if let base = source.baseAddress {
                hidden.withUnsafeMutableBytes { dest in
                    dest.baseAddress?.advanced(by: first * ModelABI.dimension * 2).copyMemory(from: base, byteCount: source.count)
                }
            }
        }
    }
}

final class CoreAICascadeEngine: CascadeEngine {
    typealias Tensor = CoreAITensor
    private let functions: [any CoreAIFunction]
    private let table: MappedEmbeddingTable
    private var currentRotary: (cos: CoreAITensor, sin: CoreAITensor)?

    init(functions: [any CoreAIFunction], table: MappedEmbeddingTable) {
        self.functions = functions
        self.table = table
    }

    func lookup(_ input: TokenizedInput) throws -> CoreAITensor {
        var hidden = [Float16](repeating: 0, count: ModelABI.sequenceLength * ModelABI.dimension)
        try hidden.withUnsafeMutableBytes { raw in
            try table.copyRows(ids: input.ids, into: raw)
        }
        let rotary = try Mrope.language(input: input, imageGrid: nil)
        self.currentRotary = (
            cos: CoreAITensor(shape: [1, 512, 64], fp16: rotary.cosine),
            sin: CoreAITensor(shape: [1, 512, 64], fp16: rotary.sine)
        )
        return CoreAITensor(shape: [1, 512, 2048], fp16: hidden)
    }

    func attentionMask(_ input: TokenizedInput) throws -> CoreAITensor {
        var mask = [Float16](repeating: 0, count: ModelABI.sequenceLength)
        for i in 0..<ModelABI.sequenceLength { mask[i] = Float16(input.mask[i]) }
        return CoreAITensor(shape: [1, 512], fp16: mask)
    }

    func prediction(chunk: Int, hidden: CoreAITensor, mask: CoreAITensor?) throws -> CoreAITensor {
        try prediction(chunk: chunk, hidden: hidden, mask: mask, rotary: currentRotary)
    }

    func prediction(chunk: Int, hidden: CoreAITensor, mask: CoreAITensor?, rotary: (cos: CoreAITensor, sin: CoreAITensor)?) throws -> CoreAITensor {
        guard functions.count == 6, (0..<6).contains(chunk) else {
            throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Missing cascade chunk \(chunk).")
        }
        guard let rotary else {
            throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Rotary inputs cos/sin required for CoreAI chunk.")
        }
        var inputs: [String: CoreAITensor] = [
            "hidden_in": hidden,
            "cos": rotary.cos,
            "sin": rotary.sin
        ]
        if chunk == 5 {
            guard let mask else {
                throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Missing attention_mask for chunk 5.")
            }
            inputs["attention_mask"] = mask
        } else if mask != nil {
            throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Mask may only enter chunk 5.")
        }

        let fn = functions[chunk]
        let out = try CoreAIAsyncBridge.runSync { [inputs, fn] in
            try await fn.run(inputs: inputs)
        }
        let outName = chunk == 5 ? "embedding" : "hidden_out"
        guard let outputTensor = out[outName] else {
            throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Missing output \(outName).")
        }
        return outputTensor
    }

    func readEmbedding(_ tensor: CoreAITensor) throws -> [Float] {
        guard tensor.shape == [1, 2048], let values = tensor.fp16Values, values.count == 2048 else {
            throw EmbedANEError.abiMismatch(chunk: 5, reason: "Invalid embedding tensor returned by chunk 5.")
        }
        // Pooling and L2 normalization were already performed INSIDE chunk 5.
        return values.map { Float($0) }
    }

    func predict(input: TokenizedInput, grid: VisionGrid? = nil, tokens: VisionTokens? = nil) throws -> PredictResult {
        guard (grid == nil) == (tokens == nil), grid?.tokenCount == tokens?.count else {
            throw EmbedANEError.invalidRequest("Image grid and visual tokens must agree.", param: "image")
        }
        let rotaryValues = try Mrope.language(input: input, imageGrid: grid)
        let rotary = (
            cos: CoreAITensor(shape: [1, 512, 64], fp16: rotaryValues.cosine),
            sin: CoreAITensor(shape: [1, 512, 64], fp16: rotaryValues.sine)
        )
        let start = DispatchTime.now().uptimeNanoseconds
        var hiddenArray = [Float16](repeating: 0, count: ModelABI.sequenceLength * ModelABI.dimension)
        try hiddenArray.withUnsafeMutableBytes { raw in
            try table.copyRows(ids: input.ids, into: raw)
        }
        if let tokens {
            try CoreAIVisionTokenSplicer.splice(tokens, input: input, into: &hiddenArray)
        }
        let hidden = CoreAITensor(shape: [1, 512, 2048], fp16: hiddenArray)
        let lookupNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start)
        let decoded = try SerialCascade.decode(
            hidden: hidden,
            attentionMask: { try self.attentionMask(input) },
            prediction: { try self.prediction(chunk: $0, hidden: $1, mask: $2, rotary: rotary) }
        )
        let vector = try readEmbedding(decoded.tensor)
        let result = PredictResult(embeddings: [vector], tableLookupNS: lookupNS, perChunkNS: decoded.perChunkNS)
        try result.validate(for: PredictRequest(inputs: [input]))
        guard vector.contains(where: { $0 != 0 }) else {
            throw EmbedANEError.abiMismatch(chunk: 5, reason: "Zero-length embedding returned by CoreAI graph.")
        }
        return result
    }
}

final class CoreAICascadeBackend: BlockingEmbeddingBackend, BlockingEmbeddingPreparer {
    private let bundle: URL
    private let modelID: String
    private let acquireLease: @Sendable () throws -> any Sendable
    private let verified: @Sendable (VerificationReport) -> Void
    private let loader: any CoreAIAssetLoading
    private let bookmarkStore: any CoreAIBookmarkStoring
    private let options: CoreAISpecializationOptions
    private let injectedTokenizer: (any ContentTokenizer)?
    private let injectedTable: MappedEmbeddingTable?
    private var lease: (any Sendable)?
    private var tokenizer: (any ContentTokenizer)?
    private var engine: CoreAICascadeEngine?

    init(
        bundle: URL,
        modelID: String,
        acquireLease: @escaping @Sendable () throws -> any Sendable = { () },
        verified: @escaping @Sendable (VerificationReport) -> Void = { _ in },
        loader: any CoreAIAssetLoading = defaultCoreAILoader(),
        bookmarkStore: any CoreAIBookmarkStoring = CoreAIBookmarkStore(),
        options: CoreAISpecializationOptions = .init(),
        injectedTokenizer: (any ContentTokenizer)? = nil,
        injectedTable: MappedEmbeddingTable? = nil
    ) {
        self.bundle = bundle
        self.modelID = modelID
        self.acquireLease = acquireLease
        self.verified = verified
        self.loader = loader
        self.bookmarkStore = bookmarkStore
        self.options = options
        self.injectedTokenizer = injectedTokenizer
        self.injectedTable = injectedTable
    }

    deinit {
        clearResources()
    }

    func load() throws -> LoadReport {
        clearResources()
        do {
            lease = try acquireLease()
            return try autoreleasepool {
                if FileManager.default.fileExists(atPath: bundle.appendingPathComponent("manifest.yaml").path) {
                    let verification = try BundleVerifier.verify(at: bundle, expectedID: modelID)
                    verified(verification)
                }
                let tokenizer: any ContentTokenizer
                if let injected = injectedTokenizer {
                    tokenizer = injected
                } else {
                    tokenizer = try LocalAssetTokenizer(assets: bundle)
                }
                let table: MappedEmbeddingTable
                if let injected = injectedTable {
                    table = injected
                } else {
                    let directory = try SecureDirectory(bundle)
                    table = try MappedEmbeddingTable(directory: directory)
                }

                let loadedChunks = try CoreAIChunkLoad.parallel(
                    loader: loader,
                    urlAt: { [bundle] index in Self.resolveChunkURLStatic(bundle: bundle, index: index) },
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
                self.engine = CoreAICascadeEngine(functions: functions, table: table)
                self.tokenizer = tokenizer
                return LoadReport(perChunkNS: timings, residentBytes: ProcessMemory.residentBytes(), computePlanChecked: false)
            }
        } catch {
            clearResources()
            throw error
        }
    }

    func prepare(_ texts: [String]) throws -> PredictRequest {
        guard let tokenizer, engine != nil else { throw EmbedANEError.modelNotLoaded }
        let start = DispatchTime.now().uptimeNanoseconds
        let inputs = try TokenizedInput.prepare(texts, tokenizer: tokenizer)
        return try PredictRequest(inputs: inputs,
            tokenizeNS: elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start))
    }

    func predict(_ request: PredictRequest) throws -> PredictResult {
        guard let engine else { throw EmbedANEError.modelNotLoaded }
        return try autoreleasepool { try SerialCascade.predict(request, engine: engine) }
    }

    func unload() throws -> UnloadReport {
        clearResources()
        return UnloadReport(residentBytes: ProcessMemory.residentBytes())
    }

    private func clearResources() {
        autoreleasepool {
            engine = nil
            tokenizer = nil
        }
        lease = nil
    }

    private static func resolveChunkURLStatic(bundle: URL, index: Int) -> URL {
        let candidates = [
            bundle.appendingPathComponent("coreai/chunk\(index).aimodel", isDirectory: true),
            bundle.appendingPathComponent("chunks/chunk\(index).aimodel", isDirectory: true),
            bundle.appendingPathComponent("chunk\(index).aimodel", isDirectory: true),
            bundle.appendingPathComponent("coreai/chunk\(index)", isDirectory: true)
        ]
        for candidate in candidates {
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        return candidates[0]
    }

    private func loadChunk(url: URL, chunk: Int) throws -> CoreAILoadedAsset {
        let loader = self.loader
        let bookmarkStore = self.bookmarkStore
        let modelID = self.modelID
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
