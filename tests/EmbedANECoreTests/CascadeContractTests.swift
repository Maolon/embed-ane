import Foundation
import Testing
@testable import EmbedANECore

private final class TestTensor {
    let seed: Int
    let mask: [Float]?
    init(seed: Int, mask: [Float]? = nil) { self.seed = seed; self.mask = mask }
}
private final class MockCascadeEngine: CascadeEngine {
    struct Call { let chunk: Int; let input: TestTensor; let output: TestTensor; let mask: TestTensor? }
    var calls: [Call] = []
    var lookups: [TestTensor] = []
    var failureChunk: Int?
    var malformedOutput = false
    func lookup(_ input: TokenizedInput) throws -> TestTensor {
        let tensor = TestTensor(seed: input.ids[0]); lookups.append(tensor); return tensor
    }
    func attentionMask(_ input: TokenizedInput) throws -> TestTensor { TestTensor(seed: 0, mask: input.mask) }
    func prediction(chunk: Int, hidden: TestTensor, mask: TestTensor?) throws -> TestTensor {
        let output = TestTensor(seed: hidden.seed)
        calls.append(.init(chunk: chunk, input: hidden, output: output, mask: mask))
        if failureChunk == chunk { throw EmbedANEError.failed("fixture") }
        return output
    }
    func readEmbedding(_ tensor: TestTensor) throws -> [Float] {
        if malformedOutput { return [1] }
        return [Float(tensor.seed), 3] + Array(repeating: 0, count: 2046)
    }
}

struct CascadeContractTests {
    @Test func allSixFrozenDescriptions() throws {
        for index in 0..<6 {
            var inputs = ["hidden_in": CascadeABI.hidden]
            if index == 5 { inputs["attention_mask"] = CascadeABI.mask }
            try CascadeABI.validate(chunk: index, inputs: inputs,
                outputs: index == 5 ? ["embedding": CascadeABI.embedding] : ["hidden_out": CascadeABI.hidden])
        }
    }
    @Test(arguments: ["input_name", "output_name", "hidden_shape", "hidden_dtype", "early_mask", "missing_mask",
                      "mask_dtype", "mask_shape", "output_shape", "optional", "extra_output"])
    func rejectsDescriptionMismatch(_ mutation: String) throws {
        let chunk = mutation == "early_mask" ? 0 : 5
        var inputs = ["hidden_in": CascadeABI.hidden, "attention_mask": CascadeABI.mask]
        var outputs = chunk == 5 ? ["embedding": CascadeABI.embedding] : ["hidden_out": CascadeABI.hidden]
        switch mutation {
        case "input_name": inputs["hidden_in"] = nil; inputs["hidden"] = CascadeABI.hidden
        case "output_name": outputs = ["other": CascadeABI.embedding]
        case "hidden_shape": inputs["hidden_in"] = .init(shape: [2, 512, 2048], element: .float16)
        case "hidden_dtype": inputs["hidden_in"] = .init(shape: [1, 512, 2048], element: .float32)
        case "early_mask": break
        case "missing_mask": inputs["attention_mask"] = nil
        case "mask_dtype": inputs["attention_mask"] = .init(shape: [1, 512], element: .float16)
        case "mask_shape": inputs["attention_mask"] = .init(shape: [512], element: .float32)
        case "output_shape": outputs["embedding"] = .init(shape: [2048], element: .float32)
        case "optional": inputs["hidden_in"]?.optional = true
        case "flexible": inputs["hidden_in"]?.flexible = true
        default: outputs["extra"] = CascadeABI.embedding
        }
        #expect(throws: EmbedANEError.self) { try CascadeABI.validate(chunk: chunk, inputs: inputs, outputs: outputs) }
    }
    @Test func acceptsFlexibleShapeDeclaration() throws {
        // Real converted chunks declare shape-flexible tensors (superset of the
        // frozen ABI); shape/rank/dtype/optionality still match exactly.
        var inputs = ["hidden_in": CascadeABI.hidden, "attention_mask": CascadeABI.mask]
        var outputs = ["embedding": CascadeABI.embedding]
        inputs["hidden_in"]?.flexible = true
        outputs["embedding"]?.flexible = true
        try CascadeABI.validate(chunk: 5, inputs: inputs, outputs: outputs)
    }
    @Test func serialBatchPreservesTensorIdentityAndNoNormalization() throws {
        let engine = MockCascadeEngine()
        let inputs = try (1...8).map { try TokenizedInput(contentIDs: [$0]) }
        let request = try PredictRequest(inputs: inputs)
        var ticks: UInt64 = 0
        let result = try SerialCascade.predict(request, engine: engine, now: { ticks += 1; return ticks })
        #expect(engine.lookups.count == 8)
        #expect(engine.calls.map(\.chunk) == Array(repeating: Array(0..<6), count: 8).flatMap { $0 })
        for item in 0..<8 {
            let first = item * 6
            #expect(engine.calls[first].input === engine.lookups[item])
            for chunk in 0..<6 {
                let call = engine.calls[first + chunk]
                if chunk > 0 { #expect(call.input === engine.calls[first + chunk - 1].output) }
                if chunk == 5 { #expect(call.mask?.mask == inputs[item].mask) }
                else { #expect(call.mask == nil) }
            }
            #expect(result.embeddings[item][0] == Float(item + 1))
            #expect(result.embeddings[item][1] == 3) // Deliberately non-unit mock output, unchanged.
        }
        #expect(result.perChunkNS == Array(repeating: UInt64(8), count: 6))
        #expect(result.tableLookupNS == 8)
        #expect(request.promptTokens == 16)
    }
    @Test func chunkFailureStopsCurrentCascadeAndRemainingBatch() throws {
        let engine = MockCascadeEngine(); engine.failureChunk = 2
        let request = try PredictRequest(inputs: [TokenizedInput(contentIDs: [1]), TokenizedInput(contentIDs: [2])])
        #expect(throws: EmbedANEError.self) { try SerialCascade.predict(request, engine: engine) }
        #expect(engine.calls.map(\.chunk) == [0, 1, 2]); #expect(engine.lookups.count == 1)
    }
    @Test func malformedFinalVectorIsRejected() throws {
        let engine = MockCascadeEngine(); engine.malformedOutput = true
        let request = try PredictRequest(inputs: [TokenizedInput(contentIDs: [1])])
        #expect(throws: EmbedANEError.self) { try SerialCascade.predict(request, engine: engine) }
    }
    @Test func maximumUsageIs4096WithoutTruncation() throws {
        let input = try TokenizedInput(contentIDs: Array(repeating: 7, count: 511))
        let request = try PredictRequest(inputs: Array(repeating: input, count: 8))
        #expect(request.promptTokens == 4096)
        #expect(throws: EmbedANEError.self) { try TokenizedInput(contentIDs: Array(repeating: 7, count: 512)) }
    }
}
