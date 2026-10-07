import CoreML
import Dispatch
import Foundation

public struct VisionTokens: Sendable {
    public let values: [Float16]
    public let count: Int
    public init(values: [Float16], count: Int) throws {
        guard (1...VisionABIConstants.wemm.tokenCapacity).contains(count),
              values.count == count * ModelABI.dimension, values.allSatisfy(\.isFinite) else {
            throw EmbedANEError.verification(path: "vision_tower", reason: "Invalid [V,2048] vision token output.")
        }
        self.values = values; self.count = count
    }
}

struct VisionTowerResult {
    let tokens: VisionTokens
    let predictionNS: UInt64
}

enum VisionTowerABI {
    private static let abi = VisionABIConstants.wemm
    static let inputs: [String: CascadeTensorDescription] = [
        "patches": .init(shape: [abi.patchCapacity, abi.patchDimension], element: .float16),
        "abs_pos": .init(shape: [abi.patchCapacity, abi.hiddenDimension], element: .float16),
        "key_mask": .init(shape: [abi.patchCapacity], element: .float16),
        "cos": .init(shape: [abi.patchCapacity, abi.visionHeadDimension], element: .float16),
        "sin": .init(shape: [abi.patchCapacity, abi.visionHeadDimension], element: .float16)
    ]
    static let output = CascadeTensorDescription(shape: [abi.tokenCapacity, ModelABI.dimension], element: .float16)
    static func validate(inputs: [String: CascadeTensorDescription], outputs: [String: CascadeTensorDescription]) throws {
        guard inputs == Self.inputs, outputs == ["var_2722": output] else {
            throw EmbedANEError.verification(path: "vision_tower", reason: "Expected the two-frame fp16 tower ABI: patches [1536,1536], output var_2722 [384,2048]. Older single-frame towers are not supported.")
        }
    }
}

/// Synchronous and deliberately non-Sendable. MultimodalPredictor constructs,
/// invokes and destroys the tower on the same serial worker as the six chunks.
final class VisionTowerEngine {
    private let model: MLModel
    let loadNS: UInt64
    init(url: URL) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let start = DispatchTime.now().uptimeNanoseconds
        do { model = try MLModel(contentsOf: url, configuration: configuration) }
        catch { throw EmbedANEError.failed("Vision tower load failed: \(error)") }
        try VisionTowerABI.validate(inputs: CoreMLTensor.describe(model.modelDescription.inputDescriptionsByName),
            outputs: CoreMLTensor.describe(model.modelDescription.outputDescriptionsByName))
        loadNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start)
    }

    func predict(_ input: VisionPreparedInput) throws -> VisionTowerResult {
        let abi = VisionABIConstants.wemm
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "patches": CoreMLTensor.half(input.patches, shape: [abi.patchCapacity, abi.patchDimension]),
            "abs_pos": CoreMLTensor.half(input.absolutePositions, shape: [abi.patchCapacity, abi.hiddenDimension]),
            "key_mask": CoreMLTensor.half(input.keyMask, shape: [abi.patchCapacity]),
            "cos": CoreMLTensor.half(input.rotary.cosine, shape: [abi.patchCapacity, abi.visionHeadDimension]),
            "sin": CoreMLTensor.half(input.rotary.sine, shape: [abi.patchCapacity, abi.visionHeadDimension])
        ])
        let start = DispatchTime.now().uptimeNanoseconds
        let output: any MLFeatureProvider
        do { output = try model.prediction(from: features) }
        catch { throw EmbedANEError.failed("Vision tower prediction failed: \(error)") }
        let predictionNS = elapsedNS(DispatchTime.now().uptimeNanoseconds, since: start)
        guard let tensor = output.featureValue(for: "var_2722")?.multiArrayValue else {
            throw EmbedANEError.verification(path: "vision_tower", reason: "Missing var_2722 output.")
        }
        return try .init(tokens: Self.readTokens(tensor, count: input.grid.tokenCount), predictionNS: predictionNS)
    }

    static func readTokens(_ tensor: MLMultiArray, count: Int) throws -> VisionTokens {
        guard tensor.shape.map(\.intValue) == VisionTowerABI.output.shape, tensor.dataType == .float16,
              (1...VisionABIConstants.wemm.tokenCapacity).contains(count), tensor.strides.count == 2,
              tensor.strides.allSatisfy({ $0.intValue > 0 && $0.intValue <= Int.max / (tensor.count * 2) }) else {
            throw EmbedANEError.verification(path: "vision_tower", reason: "Invalid tower output shape/dtype/strides.")
        }
        let pointer = tensor.dataPointer.assumingMemoryBound(to: Float16.self)
        let rowStride = tensor.strides[0].intValue, columnStride = tensor.strides[1].intValue
        // Prediction outputs are not guaranteed contiguous. Keep only the real
        // merged rows; padded query outputs are intentionally discarded.
        var values = [Float16](); values.reserveCapacity(count * ModelABI.dimension)
        for row in 0..<count {
            for column in 0..<ModelABI.dimension { values.append(pointer[row * rowStride + column * columnStride]) }
        }
        return try .init(values: values, count: count)
    }
}
