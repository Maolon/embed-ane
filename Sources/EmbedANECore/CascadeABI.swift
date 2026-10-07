import Foundation

/// Value descriptions let the frozen ABI be tested without loading CoreML.
/// Equality ignores `flexible`: a shape-flexible declaration is a superset of
/// the frozen [1,512,2048] rank/shape/dtype contract. Our converted chunks
/// declare rank-flexible tensors while still matching the frozen ABI exactly
/// (verified on device); the runtime always feeds fixed shapes.
struct CascadeTensorDescription: Equatable, Sendable {
    enum Element: String, Sendable { case float16, float32, unsupported }
    let shape: [Int]
    let element: Element
    var optional = false
    var flexible = false
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.shape == rhs.shape && lhs.element == rhs.element && lhs.optional == rhs.optional
    }
}

enum CascadeVariant: Sendable { case plain, extrope }

enum CascadeABI {
    static let hidden = CascadeTensorDescription(shape: [1, 512, 2048], element: .float16)
    static let mask = CascadeTensorDescription(shape: [1, 512], element: .float32)
    static let embedding = CascadeTensorDescription(shape: [1, 2048], element: .float32)
    static let rotary = CascadeTensorDescription(shape: [1, ModelABI.sequenceLength, VisionABIConstants.wemm.rotaryDimension], element: .float16)

    static func validate(chunk: Int, inputs: [String: CascadeTensorDescription],
                         outputs: [String: CascadeTensorDescription], variant: CascadeVariant = .plain) throws {
        guard (0..<6).contains(chunk) else {
            throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Exactly six chunks (0...5) are required.")
        }
        var expectedInputs = ["hidden_in": hidden]
        if variant == .extrope { expectedInputs["cos"] = rotary; expectedInputs["sin"] = rotary }
        let expectedOutputs: [String: CascadeTensorDescription]
        if chunk == 5 {
            expectedInputs["attention_mask"] = mask
            expectedOutputs = ["embedding": embedding]
        } else { expectedOutputs = ["hidden_out": hidden] }
        guard inputs == expectedInputs, outputs == expectedOutputs else {
            throw EmbedANEError.abiMismatch(chunk: chunk,
                reason: "Expected fixed, required tensors: inputs \(expectedInputs), outputs \(expectedOutputs); got inputs \(inputs), outputs \(outputs).")
        }
    }
}

/// Synchronous queue-confined engine. Tensor identity is retained between calls;
/// the orchestrator never converts, normalizes, pools, or batches tensors.
protocol CascadeEngine: AnyObject {
    associatedtype Tensor
    func lookup(_ input: TokenizedInput) throws -> Tensor
    func attentionMask(_ input: TokenizedInput) throws -> Tensor
    func prediction(chunk: Int, hidden: Tensor, mask: Tensor?) throws -> Tensor
    func readEmbedding(_ tensor: Tensor) throws -> [Float]
}

enum SerialCascade {
    static func predict<E: CascadeEngine>(_ request: PredictRequest, engine: E,
        now: () -> UInt64 = { DispatchTime.now().uptimeNanoseconds }) throws -> PredictResult {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(request.inputs.count)
        var lookupNS: UInt64 = 0
        var chunkNS = Array(repeating: UInt64(0), count: 6)
        for input in request.inputs {
            let lookupStart = now()
            let hidden = try engine.lookup(input)
            lookupNS = addingNS(lookupNS, elapsedNS(now(), since: lookupStart))
            let decoded = try decode(hidden: hidden, attentionMask: { try engine.attentionMask(input) },
                prediction: { try engine.prediction(chunk: $0, hidden: $1, mask: $2) }, now: now)
            for chunk in 0..<6 { chunkNS[chunk] = addingNS(chunkNS[chunk], decoded.perChunkNS[chunk]) }
            vectors.append(try engine.readEmbedding(decoded.tensor))
        }
        let result = PredictResult(embeddings: vectors, tableLookupNS: lookupNS, perChunkNS: chunkNS)
        try result.validate(for: request)
        return result
    }

    /// Shared B=1 loop for plain and external-RoPE decoders. The output tensor
    /// is handed directly to the next graph; only chunk 5 receives a mask.
    static func decode<T>(hidden: T, attentionMask: () throws -> T,
        prediction: (Int, T, T?) throws -> T,
        now: () -> UInt64 = { DispatchTime.now().uptimeNanoseconds }) throws -> (tensor: T, perChunkNS: [UInt64]) {
        var current = hidden
        var timings: [UInt64] = []
        for chunk in 0..<6 {
            let mask = chunk == 5 ? try attentionMask() : nil
            let start = now()
            current = try prediction(chunk, current, mask)
            timings.append(elapsedNS(now(), since: start))
        }
        return (current, timings)
    }
}
