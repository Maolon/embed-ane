import Foundation

public struct CoreAITensorDescription: Equatable, Sendable {
    public enum Element: String, Sendable { case float16, float32, unsupported }
    public let shape: [Int]
    public let element: Element
    public init(shape: [Int], element: Element) {
        self.shape = shape
        self.element = element
    }
}

public struct CoreAITensor: Sendable {
    public enum Element: Sendable {
        case float16([Float16])
        case float32([Float])
    }
    public let shape: [Int]
    public let element: Element

    public init(shape: [Int], fp16: [Float16]) {
        self.shape = shape
        self.element = .float16(fp16)
    }
    public init(shape: [Int], fp32: [Float]) {
        self.shape = shape
        self.element = .float32(fp32)
    }
    public var fp16Values: [Float16]? {
        if case let .float16(values) = element { return values }
        return nil
    }
    public var fp32Values: [Float]? {
        if case let .float32(values) = element { return values }
        return nil
    }
}

public enum CoreAIComputeUnitKind: Sendable, Equatable {
    case neuralEngine
    case cpu
}

public struct CoreAISpecializationOptions: Sendable, Equatable {
    public let preferredUnit: CoreAIComputeUnitKind
    public init(preferredUnit: CoreAIComputeUnitKind = .neuralEngine) {
        self.preferredUnit = preferredUnit
    }
}

public enum CoreAIABI {
    public static let hidden = CoreAITensorDescription(shape: [1, 512, 2048], element: .float16)
    // The fp16 asset generation made the whole graph fp16:
    // mask values are exact 0/1 and the final embedding is graph-normalized,
    // so fp16 carries both losslessly. The ANE validator rejects F32 tensors.
    public static let mask = CoreAITensorDescription(shape: [1, 512], element: .float16)
    public static let embedding = CoreAITensorDescription(shape: [1, 2048], element: .float16)
    public static let rotary = CoreAITensorDescription(shape: [1, ModelABI.sequenceLength, VisionABIConstants.wemm.rotaryDimension], element: .float16)

    public static let towerPatches = CoreAITensorDescription(shape: [VisionABIConstants.wemm.patchCapacity, VisionABIConstants.wemm.framePatchDimension], element: .float16)
    public static let towerAbsPos = CoreAITensorDescription(shape: [VisionABIConstants.wemm.patchCapacity, VisionABIConstants.wemm.hiddenDimension], element: .float16)
    public static let towerKeyMask = CoreAITensorDescription(shape: [VisionABIConstants.wemm.patchCapacity], element: .float16)
    public static let towerRotary = CoreAITensorDescription(shape: [VisionABIConstants.wemm.patchCapacity, VisionABIConstants.wemm.visionHeadDimension], element: .float16)
    public static let towerVisualTokens = CoreAITensorDescription(shape: [VisionABIConstants.wemm.tokenCapacity, ModelABI.dimension], element: .float16)

    public static func validateChunk(
        chunk: Int,
        functionName: String,
        inputs: [String: CoreAITensorDescription],
        outputs: [String: CoreAITensorDescription]
    ) throws {
        guard (0..<6).contains(chunk) else {
            throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Exactly six chunks (0...5) are required.")
        }
        guard functionName == "chunk\(chunk)" else {
            throw EmbedANEError.abiMismatch(chunk: chunk, reason: "Expected function name chunk\(chunk), got \(functionName).")
        }
        var expectedInputs = [
            "hidden_in": hidden,
            "cos": rotary,
            "sin": rotary
        ]
        let expectedOutputs: [String: CoreAITensorDescription]
        if chunk == 5 {
            expectedInputs["attention_mask"] = mask
            expectedOutputs = ["embedding": embedding]
        } else {
            expectedOutputs = ["hidden_out": hidden]
        }
        guard matches(actual: inputs, expected: expectedInputs),
              matches(actual: outputs, expected: expectedOutputs) else {
            throw EmbedANEError.abiMismatch(chunk: chunk,
                reason: "Expected inputs \(expectedInputs), outputs \(expectedOutputs); got inputs \(inputs), outputs \(outputs).")
        }
    }

    public static func validateTower(
        functionName: String,
        inputs: [String: CoreAITensorDescription],
        outputs: [String: CoreAITensorDescription]
    ) throws {
        guard functionName == "tower" else {
            throw EmbedANEError.verification(path: "vision_tower", reason: "Expected function name tower, got \(functionName).")
        }
        let expectedInputs = [
            "patches": towerPatches,
            "abs_pos": towerAbsPos,
            "cos": towerRotary,
            "sin": towerRotary,
            "key_mask": towerKeyMask
        ]
        let expectedOutputs = [
            "visual_tokens": towerVisualTokens
        ]
        guard matches(actual: inputs, expected: expectedInputs),
              matches(actual: outputs, expected: expectedOutputs) else {
            throw EmbedANEError.verification(path: "vision_tower",
                reason: "Expected tower inputs \(expectedInputs), outputs \(expectedOutputs); got inputs \(inputs), outputs \(outputs).")
        }
    }

    public static func matches(actual: [String: CoreAITensorDescription], expected: [String: CoreAITensorDescription]) -> Bool {
        guard Set(actual.keys) == Set(expected.keys) else { return false }
        for (key, expDesc) in expected {
            guard let actDesc = actual[key] else { return false }
            guard actDesc.element == expDesc.element else { return false }
            if !actDesc.shape.isEmpty && actDesc.shape != expDesc.shape { return false }
        }
        return true
    }
}
