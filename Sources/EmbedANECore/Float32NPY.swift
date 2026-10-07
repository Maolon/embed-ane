import Foundation

/// Small CPU-side arrays only (learned vision positions or reference vectors).
/// The billion-byte fp16 token table continues to use its read-only mmap path.
public struct Float32NPY: Sendable {
    public let shape: [Int]
    public let values: [Float]

    public init(data: Data, expectedShapes: [[Int]]) throws {
        let metadata = try NPYMetadata.parse(prefix: Data(data.prefix(65_545)))
        guard metadata.dtype == "<f4", expectedShapes.contains(metadata.shape) else {
            throw EmbedANEError.invalidNPY("Expected little-endian fp32 with shape in \(expectedShapes), got \(metadata.dtype) \(metadata.shape).")
        }
        var count = 1
        for dimension in metadata.shape {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !product.overflow, product.partialValue <= (Int.max - metadata.payloadOffset) / 4 else {
                throw EmbedANEError.invalidNPY("Invalid array size.")
            }
            count = product.partialValue
        }
        guard data.count == metadata.payloadOffset + count * 4 else {
            throw EmbedANEError.invalidNPY("Truncated fp32 payload or unexpected trailing bytes.")
        }
        values = data.withUnsafeBytes { bytes in
            (0..<count).map { index in
                Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: metadata.payloadOffset + index * 4, as: UInt32.self)))
            }
        }
        guard values.allSatisfy(\.isFinite) else { throw EmbedANEError.invalidNPY("Non-finite fp32 array value.") }
        shape = metadata.shape
    }

    public init(url: URL, expectedShapes: [[Int]]) throws {
        let directory = try SecureDirectory(url.deletingLastPathComponent())
        try self.init(data: directory.read(url.lastPathComponent, limit: 16 * 1_024 * 1_024), expectedShapes: expectedShapes)
    }
}
