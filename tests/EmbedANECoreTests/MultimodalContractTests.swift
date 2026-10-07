import CoreML
import Foundation
import Testing
@testable import EmbedANECore

struct MultimodalContractTests {
    @Test func extropeRequiresBothRotaryInputsInEveryChunk() throws {
        for index in 0..<6 {
            var inputs = ["hidden_in": CascadeABI.hidden, "cos": CascadeABI.rotary, "sin": CascadeABI.rotary]
            if index == 5 { inputs["attention_mask"] = CascadeABI.mask }
            let outputs = index == 5 ? ["embedding": CascadeABI.embedding] : ["hidden_out": CascadeABI.hidden]
            try CascadeABI.validate(chunk: index, inputs: inputs, outputs: outputs, variant: .extrope)
            #expect(throws: EmbedANEError.self) { try CascadeABI.validate(chunk: index, inputs: inputs, outputs: outputs) }
            for key in ["cos", "sin"] {
                var missing = inputs; missing[key] = nil
                #expect(throws: EmbedANEError.self) { try CascadeABI.validate(chunk: index, inputs: missing, outputs: outputs, variant: .extrope) }
                var wrong = inputs; wrong[key] = .init(shape: [1, 512, 128], element: .float16)
                #expect(throws: EmbedANEError.self) { try CascadeABI.validate(chunk: index, inputs: wrong, outputs: outputs, variant: .extrope) }
                wrong[key] = .init(shape: [1, 512, 64], element: .float32)
                #expect(throws: EmbedANEError.self) { try CascadeABI.validate(chunk: index, inputs: wrong, outputs: outputs, variant: .extrope) }
            }
        }
    }
    @Test func towerABIRejectsWrongFeatureNameDtypeAndOptionality() throws {
        try VisionTowerABI.validate(inputs: VisionTowerABI.inputs, outputs: ["var_2722": VisionTowerABI.output])
        #expect(throws: EmbedANEError.self) { try VisionTowerABI.validate(inputs: VisionTowerABI.inputs, outputs: ["visual_tokens": VisionTowerABI.output]) }
        for key in VisionTowerABI.inputs.keys {
            var inputs = VisionTowerABI.inputs; inputs[key] = nil
            #expect(throws: EmbedANEError.self) { try VisionTowerABI.validate(inputs: inputs, outputs: ["var_2722": VisionTowerABI.output]) }
            inputs = VisionTowerABI.inputs; inputs[key]?.optional = true
            #expect(throws: EmbedANEError.self) { try VisionTowerABI.validate(inputs: inputs, outputs: ["var_2722": VisionTowerABI.output]) }
        }
    }
    @Test func towerOutputRespectsStridesAndSlicesOnlyRealTokens() throws {
        let storage = try MLMultiArray(shape: [384, 4096], dataType: .float16)
        let pointer = storage.dataPointer.assumingMemoryBound(to: Float16.self)
        pointer.initialize(repeating: .nan, count: storage.count)
        for row in 0..<2 { for column in 0..<2048 { pointer[row * 4096 + column * 2] = Float16(row + column % 11) } }
        let view = try MLMultiArray(dataPointer: storage.dataPointer, shape: [384, 2048], dataType: .float16,
            strides: [4096, 2], deallocator: { _ in })
        let tokens = try VisionTowerEngine.readTokens(view, count: 2)
        withExtendedLifetime(storage) {}
        #expect(tokens.count == 2 && tokens.values.count == 4096)
        #expect(tokens.values[0] == 0 && tokens.values[2048] == 1)
        #expect(tokens.values[4095] == Float16(1 + 2047 % 11))
        #expect(throws: EmbedANEError.self) { try VisionTowerEngine.readTokens(view, count: 0) }
        #expect(throws: EmbedANEError.self) { try VisionTokens(values: [.nan], count: 1) }
    }
    @Test func spliceChangesOnlyImageRowsAndRejectsBeforeWriting() throws {
        let grid = try VisionGrid(height: 2, width: 4)
        let input = try MultimodalTokenization.prepare(text: "", grid: grid, tokenizer: VisionFixtureTokenizer(ids: [7]))
        let hidden = try MLMultiArray(shape: [1, 512, 2048], dataType: .float16)
        let pointer = hidden.dataPointer.assumingMemoryBound(to: Float16.self)
        pointer.initialize(repeating: -2, count: hidden.count)
        let tokens = try VisionTokens(values: Array(repeating: 3, count: 2048) + Array(repeating: 4, count: 2048), count: 2)
        try VisionTokenSplicer.splice(tokens, input: input, into: hidden)
        // Rows: 0 <|im_start|>, 1 user, 2 <vision_start>, 3-4 image, 5 <vision_end>, ...
        #expect(pointer[0] == -2 && pointer[3 * 2048 - 1] == -2)
        #expect(pointer[3 * 2048] == 3 && pointer[4 * 2048 - 1] == 3)
        #expect(pointer[4 * 2048] == 4 && pointer[5 * 2048 - 1] == 4)
        #expect(pointer[5 * 2048] == -2 && pointer[hidden.count - 1] == -2)
        let bad = try VisionTokens(values: Array(repeating: 9, count: 2048), count: 1)
        #expect(throws: EmbedANEError.self) { try VisionTokenSplicer.splice(bad, input: input, into: hidden) }
        #expect(pointer[3 * 2048] == 3 && pointer[4 * 2048] == 4)
    }
    @Test func fp16TensorBuilderPreservesBitsAndRejectsNonfinite() throws {
        let values: [Float16] = [0, -0.5, 0.25, 100]
        let array = try CoreMLTensor.half(values, shape: [2, 2])
        let pointer = array.dataPointer.assumingMemoryBound(to: Float16.self)
        #expect((0..<4).map { pointer[$0].bitPattern } == values.map(\.bitPattern))
        #expect(throws: EmbedANEError.self) { try CoreMLTensor.half([.infinity], shape: [1]) }
    }
    @Test func injectedPathsFailEarlyAndUnloadedFacadeStaysSafe() async throws {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let paths = try MultimodalModelPaths(tower: missing, chunks: Array(repeating: missing, count: 6),
            tokenizerDirectory: missing, embeddingTable: missing, positionTable: missing)
        let runtime = MultimodalPredictor(paths: paths)
        await #expect(throws: EmbedANEError.self) { try await runtime.embed(imageData: nil, text: "test") }
        // Missing tokenizer assets fail before any MLModel constructor is reached.
        await #expect(throws: EmbedANEError.self) { try await runtime.loadVision() }
        _ = try await runtime.unload()
        await #expect(throws: EmbedANEError.self) { try await runtime.prepare(["test"]) }
        #expect(throws: EmbedANEError.self) {
            try MultimodalModelPaths(tower: missing, chunks: [], tokenizerDirectory: missing, embeddingTable: missing, positionTable: missing)
        }
    }
}
