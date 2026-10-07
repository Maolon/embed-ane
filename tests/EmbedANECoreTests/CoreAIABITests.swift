import Foundation
import Testing
@testable import EmbedANECore

@Suite("CoreAI ABI validation contracts")
struct CoreAIABITests {
    @Test func allSixFrozenChunkDescriptions() throws {
        for index in 0..<6 {
            var inputs: [String: CoreAITensorDescription] = [
                "hidden_in": CoreAIABI.hidden,
                "cos": CoreAIABI.rotary,
                "sin": CoreAIABI.rotary
            ]
            if index == 5 {
                inputs["attention_mask"] = CoreAIABI.mask
            }
            let outputs = [
                index == 5 ? "embedding" : "hidden_out": index == 5 ? CoreAIABI.embedding : CoreAIABI.hidden
            ]
            try CoreAIABI.validateChunk(
                chunk: index,
                functionName: "chunk\(index)",
                inputs: inputs,
                outputs: outputs
            )
        }
    }

    @Test func towerDescription() throws {
        let inputs: [String: CoreAITensorDescription] = [
            "patches": CoreAIABI.towerPatches,
            "abs_pos": CoreAIABI.towerAbsPos,
            "cos": CoreAIABI.towerRotary,
            "sin": CoreAIABI.towerRotary,
            "key_mask": CoreAIABI.towerKeyMask
        ]
        let outputs = [
            "visual_tokens": CoreAIABI.towerVisualTokens
        ]
        try CoreAIABI.validateTower(
            functionName: "tower",
            inputs: inputs,
            outputs: outputs
        )
    }

    @Test func rejectsMismatchingChunkFunctionName() throws {
        let inputs: [String: CoreAITensorDescription] = [
            "hidden_in": CoreAIABI.hidden,
            "cos": CoreAIABI.rotary,
            "sin": CoreAIABI.rotary
        ]
        let outputs = ["hidden_out": CoreAIABI.hidden]
        #expect(throws: EmbedANEError.self) {
            try CoreAIABI.validateChunk(chunk: 0, functionName: "chunk1", inputs: inputs, outputs: outputs)
        }
    }

    @Test func rejectsMismatchingTowerFunctionName() throws {
        let inputs: [String: CoreAITensorDescription] = [
            "patches": CoreAIABI.towerPatches,
            "abs_pos": CoreAIABI.towerAbsPos,
            "cos": CoreAIABI.towerRotary,
            "sin": CoreAIABI.towerRotary,
            "key_mask": CoreAIABI.towerKeyMask
        ]
        let outputs = ["visual_tokens": CoreAIABI.towerVisualTokens]
        #expect(throws: EmbedANEError.self) {
            try CoreAIABI.validateTower(functionName: "main", inputs: inputs, outputs: outputs)
        }
    }

    @Test func rejectsChunkMissingOrEarlyMask() throws {
        // Missing mask on chunk 5
        let inputs5: [String: CoreAITensorDescription] = [
            "hidden_in": CoreAIABI.hidden,
            "cos": CoreAIABI.rotary,
            "sin": CoreAIABI.rotary
        ]
        let outputs5 = ["embedding": CoreAIABI.embedding]
        #expect(throws: EmbedANEError.self) {
            try CoreAIABI.validateChunk(chunk: 5, functionName: "chunk5", inputs: inputs5, outputs: outputs5)
        }

        // Early mask on chunk 0
        var inputs0 = inputs5
        inputs0["attention_mask"] = CoreAIABI.mask
        let outputs0 = ["hidden_out": CoreAIABI.hidden]
        #expect(throws: EmbedANEError.self) {
            try CoreAIABI.validateChunk(chunk: 0, functionName: "chunk0", inputs: inputs0, outputs: outputs0)
        }
    }

    @Test func rejectsMismatchingChunkTensorShapeOrElement() throws {
        // Wrong shape
        var inputs = [
            "hidden_in": CoreAITensorDescription(shape: [2, 512, 2048], element: .float16),
            "cos": CoreAIABI.rotary,
            "sin": CoreAIABI.rotary
        ]
        let outputs = ["hidden_out": CoreAIABI.hidden]
        #expect(throws: EmbedANEError.self) {
            try CoreAIABI.validateChunk(chunk: 0, functionName: "chunk0", inputs: inputs, outputs: outputs)
        }

        // Wrong element type
        inputs["hidden_in"] = CoreAITensorDescription(shape: [1, 512, 2048], element: .float32)
        #expect(throws: EmbedANEError.self) {
            try CoreAIABI.validateChunk(chunk: 0, functionName: "chunk0", inputs: inputs, outputs: outputs)
        }
    }

    @Test func rejectsTowerMismatchingInputOrOutput() throws {
        var inputs: [String: CoreAITensorDescription] = [
            "patches": CoreAIABI.towerPatches,
            "abs_pos": CoreAIABI.towerAbsPos,
            "cos": CoreAIABI.towerRotary,
            "sin": CoreAIABI.towerRotary,
            "key_mask": CoreAIABI.towerKeyMask
        ]
        var outputs = ["visual_tokens": CoreAIABI.towerVisualTokens]

        // Missing input
        inputs["patches"] = nil
        #expect(throws: EmbedANEError.self) {
            try CoreAIABI.validateTower(functionName: "tower", inputs: inputs, outputs: outputs)
        }

        // Wrong output name
        inputs["patches"] = CoreAIABI.towerPatches
        outputs = ["var_2722": CoreAIABI.towerVisualTokens]
        #expect(throws: EmbedANEError.self) {
            try CoreAIABI.validateTower(functionName: "tower", inputs: inputs, outputs: outputs)
        }
    }
}
