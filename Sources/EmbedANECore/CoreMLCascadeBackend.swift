import Dispatch
import CoreML
import Foundation

/// Only SerialEmbeddingWorker constructs, invokes and releases this object.
/// The lease factory is supplied by CLI/App composition, keeping Core independent
/// of Download while excluding replacement throughout verification and residency.
final class CoreMLCascadeBackend: BlockingEmbeddingBackend, BlockingEmbeddingPreparer {
    private let bundle: URL
    private let modelID: String
    private let acquireLease: @Sendable () throws -> any Sendable
    private let verified: @Sendable (VerificationReport) -> Void
    private var lease: (any Sendable)?
    private var tokenizer: LocalAssetTokenizer?
    private var engine: CoreMLCascadeEngine?

    init(bundle: URL, modelID: String,
         acquireLease: @escaping @Sendable () throws -> any Sendable,
         verified: @escaping @Sendable (VerificationReport) -> Void) {
        self.bundle = bundle; self.modelID = modelID
        self.acquireLease = acquireLease; self.verified = verified
    }

    deinit { clearResources() }

    func load() throws -> LoadReport {
        // Also resets a partially failed prediction session before a retry.
        clearResources()
        do {
            lease = try acquireLease()
            return try autoreleasepool {
                let verification = try BundleVerifier.verify(at: bundle, expectedID: modelID)
                let directory = try SecureDirectory(bundle)
                let tokenizer = try LocalAssetTokenizer(assets: bundle)
                let table = try MappedEmbeddingTable(directory: directory)
                let configuration = MLModelConfiguration()
                configuration.computeUnits = .cpuAndNeuralEngine
                var models: [MLModel] = []
                var timings: [UInt64] = []
                for index in 0..<6 {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let url = bundle.appendingPathComponent("chunks/chunk\(index).mlmodelc", isDirectory: true)
                    let model: MLModel
                    do { model = try MLModel(contentsOf: url, configuration: configuration) }
                    catch { throw EmbedANEError.failed("CoreML chunk \(index) load failed: \(error)") }
                    // Every chunk is checked before engine is installed and any
                    // prediction is possible. No warm-up prediction is hidden here.
                    try CoreMLCascadeEngine.validateDescription(model.modelDescription, chunk: index)
                    timings.append(elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start))
                    models.append(model)
                }
                engine = CoreMLCascadeEngine(models: models, table: table)
                self.tokenizer = tokenizer
                verified(verification)
                // The async public compute-plan API is inspected by CoreMLRuntime
                // on this worker's preferred executor before load completes.
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
        do {
            return try autoreleasepool { try SerialCascade.predict(request, engine: engine) }
        } catch let err as EmbedANEError {
            throw err
        } catch {
            let desc = String(describing: error)
            throw EmbedANEError.failed("Prediction failed: \(desc)")
        }
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
        // Release installation protection only AFTER the autorelease pool drains.
        lease = nil
    }
}

final class CoreMLCascadeEngine: CascadeEngine {
    typealias Tensor = MLMultiArray
    private let models: [MLModel]
    private let table: MappedEmbeddingTable
    private let variant: CascadeVariant
    init(models: [MLModel], table: MappedEmbeddingTable, variant: CascadeVariant = .plain) {
        self.models = models; self.table = table; self.variant = variant
    }

    static func validateDescription(_ description: MLModelDescription, chunk: Int, variant: CascadeVariant = .plain) throws {
        try CascadeABI.validate(chunk: chunk, inputs: CoreMLTensor.describe(description.inputDescriptionsByName),
            outputs: CoreMLTensor.describe(description.outputDescriptionsByName), variant: variant)
    }

    func lookup(_ input: TokenizedInput) throws -> MLMultiArray {
        let hidden = try MLMultiArray(shape: [1, 512, 2048], dataType: .float16)
        // Only the last two strides affect a [1,512,2048] tensor. Never assume
        // the allocator's layout when doing raw fp16 row copies.
        guard hidden.strides.count == 3, hidden.strides[1].intValue == 2048,
              hidden.strides[2].intValue == 1 else {
            throw EmbedANEError.abiMismatch(chunk: 0, reason: "Allocated hidden tensor is not C-contiguous.")
        }
        try table.copyRows(ids: input.ids, into: .init(start: hidden.dataPointer, count: MappedEmbeddingTable.hiddenByteCount))
        return hidden
    }
    func attentionMask(_ input: TokenizedInput) throws -> MLMultiArray {
        let mask = try MLMultiArray(shape: [1, 512], dataType: .float32)
        guard mask.strides.count == 2, mask.strides[1].intValue == 1 else {
            throw EmbedANEError.abiMismatch(chunk: 5, reason: "Allocated fp32 mask is not contiguous.")
        }
        let pointer = mask.dataPointer.assumingMemoryBound(to: Float.self)
        for index in 0..<512 { pointer[index] = input.mask[index] }
        return mask
    }
    func prediction(chunk: Int, hidden: MLMultiArray, mask: MLMultiArray?) throws -> MLMultiArray {
        try prediction(chunk: chunk, hidden: hidden, mask: mask, rotary: nil)
    }
    func prediction(chunk: Int, hidden: MLMultiArray, mask: MLMultiArray?, rotary: CoreMLRotary?) throws -> MLMultiArray {
        guard (variant == .extrope) == (rotary != nil) else {
            throw EmbedANEError.abiMismatch(chunk: chunk, reason: "External rotary inputs are required only for extrope chunks.")
        }
        guard models.count == 6, (0..<6).contains(chunk) else {
            throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Missing cascade chunk.")
        }
        try validateTensor(hidden, expected: CascadeABI.hidden, chunk: chunk)
        return try autoreleasepool {
            var inputs: [String: MLFeatureValue] = ["hidden_in": MLFeatureValue(multiArray: hidden)]
            if let rotary {
                try validateTensor(rotary.cosine, expected: CascadeABI.rotary, chunk: chunk)
                try validateTensor(rotary.sine, expected: CascadeABI.rotary, chunk: chunk)
                inputs["cos"] = MLFeatureValue(multiArray: rotary.cosine)
                inputs["sin"] = MLFeatureValue(multiArray: rotary.sine)
            }
            if chunk == 5 {
                guard let mask else { throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Missing fp32 attention mask.") }
                try validateTensor(mask, expected: CascadeABI.mask, chunk: chunk)
                inputs["attention_mask"] = MLFeatureValue(multiArray: mask)
            } else if mask != nil {
                throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Mask may only enter chunk 5.")
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
            let output: any MLFeatureProvider
            do { output = try models[chunk].prediction(from: provider) }
            catch { throw EmbedANEError.failed("CoreML chunk \(chunk) prediction failed: \(error)") }
            let name = chunk == 5 ? "embedding" : "hidden_out"
            guard let tensor = output.featureValue(for: name)?.multiArrayValue else {
                throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Missing output \(name).")
            }
            try validateTensor(tensor, expected: chunk == 5 ? CascadeABI.embedding : CascadeABI.hidden, chunk: chunk)
            return tensor // Exact MLMultiArray returned by the graph; no cast/copy.
        }
    }
    func readEmbedding(_ tensor: MLMultiArray) throws -> [Float] {
        try validateTensor(tensor, expected: CascadeABI.embedding, chunk: 5)
        guard tensor.strides.count == 2, tensor.strides[1].intValue > 0,
              tensor.strides[1].intValue <= Int.max / 2048 else {
            throw EmbedANEError.abiMismatch(chunk: 5, reason: "Invalid output strides.")
        }
        let stride = tensor.strides[1].intValue
        let values = tensor.dataPointer.assumingMemoryBound(to: Float.self)
        // Pooling and L2 normalization were already performed INSIDE chunk 5.
        return (0..<2048).map { values[$0 * stride] }
    }
    private func validateTensor(_ tensor: MLMultiArray, expected: CascadeTensorDescription, chunk: Int) throws {
        let type: MLMultiArrayDataType = expected.element == .float16 ? .float16 : .float32
        guard tensor.shape.map(\.intValue) == expected.shape, tensor.dataType == type else {
            throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Prediction tensor shape/dtype violates the frozen ABI.")
        }
    }
}
