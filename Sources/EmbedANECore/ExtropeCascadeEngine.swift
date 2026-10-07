import CoreML
import Dispatch
import Foundation

/// One vision span in a prompt: a still image, or one frame-pair of a video
/// (preceded by its timestamp text, e.g. `<4.1 seconds>`).
public struct VisualSpan: Sendable, Equatable {
    public let grid: VisionGrid
    /// `imageToken` for stills, `videoToken` for video frame-pairs.
    public let padToken: Int
    public let prefixIDs: [Int]
    public init(grid: VisionGrid, padToken: Int, prefixIDs: [Int] = []) {
        self.grid = grid; self.padToken = padToken; self.prefixIDs = prefixIDs
    }
}

/// Prompts in the upstream chat format, adapted from our MMService.embed.
/// Tokenization adds no automatic specials; TokenizedInput appends exactly one
/// <embedding>. Visual spans come first, then the text, as in the reference.
public enum MultimodalTokenization {
    public static func prepare(text: String, grid: VisionGrid?, tokenizer: any ContentTokenizer) throws -> TokenizedInput {
        let abi = VisionABIConstants.wemm
        return try prepare(text: text, spans: grid.map { [VisualSpan(grid: $0, padToken: abi.imageToken)] } ?? [],
                           lead: .image, tokenizer: tokenizer)
    }

    public static func prepare(text: String, spans: [VisualSpan], lead: PromptTemplate.Lead,
                               tokenizer: any ContentTokenizer) throws -> TokenizedInput {
        let abi = VisionABIConstants.wemm
        let textIDs = try tokenizer.encodeContent(text)
        let reserved = [abi.imageToken, abi.visionStartToken, abi.visionEndToken, abi.videoToken,
                        ModelABI.imStartToken, ModelABI.imEndToken, ModelABI.embeddingToken]
        guard !textIDs.contains(where: reserved.contains) else {
            throw EmbedANEError.invalidRequest("Pass images and videos separately; placeholder and chat-template tokens in text are not supported.", param: "input")
        }
        guard !spans.isEmpty else { return try PromptTemplate.text(text, tokenizer: tokenizer) }
        let visual = spans.flatMap { $0.prefixIDs + [abi.visionStartToken] + Array(repeating: $0.padToken, count: $0.grid.tokenCount) + [abi.visionEndToken] }
        let total = visual.count + textIDs.count + PromptTemplate.overhead(lead) + 1
        guard total <= ModelABI.sequenceLength else {
            let kind = spans.count == 1 && lead == .image ? "Image" : "Video"
            throw EmbedANEError.invalidRequest("\(kind) uses \(visual.count) tokens and text uses \(textIDs.count); with \(PromptTemplate.overhead(lead) + 1) prompt tokens the total must not exceed 512.", param: "input")
        }
        return try TokenizedInput(contentIDs: PromptTemplate.wrap(visual + textIDs, lead: lead))
    }
}

enum VisionTokenSplicer {
    /// Each run of image/video placeholder tokens, in prompt order.
    static func spans(_ input: TokenizedInput) -> [Range<Int>] {
        let abi = VisionABIConstants.wemm
        let ids = Array(input.ids.prefix(input.nTokens))
        var result: [Range<Int>] = [], index = 0
        while index < ids.count {
            guard ids[index] == abi.imageToken || ids[index] == abi.videoToken else { index += 1; continue }
            let start = index
            while index < ids.count, ids[index] == ids[start] { index += 1 }
            result.append(start..<index)
        }
        return result
    }

    static func splice(_ tokens: VisionTokens, input: TokenizedInput, into hidden: MLMultiArray) throws {
        try splice([tokens], input: input, into: hidden)
    }

    /// Validate every span before modifying table-lookup storage.
    static func splice(_ tokens: [VisionTokens], input: TokenizedInput, into hidden: MLMultiArray) throws {
        let runs = spans(input)
        guard runs.count == tokens.count, zip(runs, tokens).allSatisfy({ $0.count == $1.count }),
              hidden.dataType == .float16, hidden.shape.map(\.intValue) == CascadeABI.hidden.shape,
              hidden.strides.count == 3, hidden.strides[1].intValue == ModelABI.dimension, hidden.strides[2].intValue == 1 else {
            throw EmbedANEError.verification(path: "vision_splice", reason: "Expected contiguous fp16 hidden storage and one placeholder span per visual token block.")
        }
        for (run, block) in zip(runs, tokens) {
            block.values.withUnsafeBytes { source in
                if let base = source.baseAddress {
                    hidden.dataPointer.advanced(by: run.lowerBound * ModelABI.dimension * 2).copyMemory(from: base, byteCount: source.count)
                }
            }
        }
    }
}

/// A thin adapter over the existing table/tensor implementation and serial loop,
/// not a second copy of the six-chunk cascade. Every chunk gets the SAME rotary
/// tensors; the final mask and hidden hand-off retain the original ABI behavior.
final class ExtropeCascadeEngine {
    private let engine: CoreMLCascadeEngine
    init(models: [MLModel], table: MappedEmbeddingTable) {
        engine = CoreMLCascadeEngine(models: models, table: table, variant: .extrope)
    }

    func predict(input: TokenizedInput, grid: VisionGrid? = nil, tokens: VisionTokens? = nil) throws -> PredictResult {
        guard (grid == nil) == (tokens == nil) else {
            throw EmbedANEError.invalidRequest("Image grid and visual tokens must agree.", param: "image")
        }
        return try predict(input: input, grids: grid.map { [$0] } ?? [], tokens: tokens.map { [$0] } ?? [])
    }

    /// One grid and one token block per visual span, in prompt order.
    func predict(input: TokenizedInput, grids: [VisionGrid], tokens: [VisionTokens]) throws -> PredictResult {
        guard grids.count == tokens.count, zip(grids, tokens).allSatisfy({ $0.tokenCount == $1.count }) else {
            throw EmbedANEError.invalidRequest("Visual grids and visual tokens must agree.", param: "image")
        }
        let rotary = try CoreMLRotary(Mrope.language(input: input, grids: grids), shape: CascadeABI.rotary.shape)
        let start = DispatchTime.now().uptimeNanoseconds
        let hidden = try engine.lookup(input)
        if !tokens.isEmpty { try VisionTokenSplicer.splice(tokens, input: input, into: hidden) }
        let lookupNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start)
        let decoded = try SerialCascade.decode(hidden: hidden, attentionMask: { try engine.attentionMask(input) },
            prediction: { try engine.prediction(chunk: $0, hidden: $1, mask: $2, rotary: rotary) })
        let vector = try engine.readEmbedding(decoded.tensor)
        let result = PredictResult(embeddings: [vector], tableLookupNS: lookupNS, perChunkNS: decoded.perChunkNS)
        try result.validate(for: PredictRequest(inputs: [input]))
        guard vector.contains(where: { $0 != 0 }) else {
            throw EmbedANEError.abiMismatch(chunk: 5, reason: "Zero-length embedding returned by extrope graph.")
        }
        // Chunk 5 already final-normalizes, pools and L2-normalizes. Return its
        // fp32 values unchanged; do NOT mask numerical drift by re-normalizing.
        return result
    }
}
