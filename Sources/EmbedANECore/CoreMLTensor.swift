import CoreML
import Foundation

/// Queue-confined tensor utilities shared by the text cascade and vision tower.
enum CoreMLTensor {
    static func describe(_ features: [String: MLFeatureDescription]) -> [String: CascadeTensorDescription] {
        features.mapValues { feature in
            guard feature.type == .multiArray, let constraint = feature.multiArrayConstraint else {
                return .init(shape: [], element: .unsupported, optional: feature.isOptional)
            }
            let element: CascadeTensorDescription.Element
            switch constraint.dataType {
            case .float16: element = .float16
            case .float32: element = .float32
            default: element = .unsupported
            }
            return .init(shape: constraint.shape.map(\.intValue), element: element,
                optional: feature.isOptional, flexible: constraint.shapeConstraint.type != .unspecified)
        }
    }

    static func half(_ values: [Float16], shape: [Int]) throws -> MLMultiArray {
        guard shape.allSatisfy({ $0 > 0 }), shape.reduce(1, *) == values.count,
              values.allSatisfy(\.isFinite) else {
            throw EmbedANEError.verification(path: "tensor", reason: "Invalid fp16 tensor size or values.")
        }
        let tensor = try MLMultiArray(shape: shape.map(NSNumber.init(value:)), dataType: .float16)
        var stride = 1
        for dimension in shape.indices.reversed() {
            guard tensor.strides[dimension].intValue == stride else {
                throw EmbedANEError.verification(path: "tensor", reason: "Allocated tensor is not contiguous.")
            }
            stride *= shape[dimension]
        }
        values.withUnsafeBytes { bytes in
            if let base = bytes.baseAddress { tensor.dataPointer.copyMemory(from: base, byteCount: bytes.count) }
        }
        return tensor
    }
}

struct CoreMLRotary {
    let cosine: MLMultiArray
    let sine: MLMultiArray
    init(_ values: RotaryValues, shape: [Int]) throws {
        cosine = try CoreMLTensor.half(values.cosine, shape: shape)
        sine = try CoreMLTensor.half(values.sine, shape: shape)
    }
}
