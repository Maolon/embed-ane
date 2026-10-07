import Foundation

/// Header-only parser: tests can validate the billion-byte table ABI without
/// allocating a billion-byte fixture. The mapped table checks the real fstat size.
public struct NPYHeader: Sendable, Equatable {
    public let payloadOffset: Int
    public let rows: Int
    public let columns: Int
    public static func parse(prefix: Data, fileByteCount: Int64) throws -> Self {
        let metadata = try NPYMetadata.parse(prefix: prefix)
        guard metadata.dtype == "<f2", metadata.shape == [ModelABI.vocabulary, ModelABI.dimension] else {
            throw EmbedANEError.invalidNPY("dtype/shape must be <f2 [248078, 2048]")
        }
        guard fileByteCount == Int64(metadata.payloadOffset) + ModelABI.tablePayloadBytes else {
            throw EmbedANEError.invalidNPY("truncated payload or unexpected trailing bytes")
        }
        return Self(payloadOffset: metadata.payloadOffset, rows: metadata.shape[0], columns: metadata.shape[1])
    }
    public func validate(ids: [Int]) throws {
        guard ids.allSatisfy({ $0 >= 0 && $0 < rows }) else { throw EmbedANEError.invalidNPY("token id is outside the embedding table") }
    }
}

/// Shared strict v1.0 header grammar; consumers impose their own dtype/shape ABI.
struct NPYMetadata {
    let payloadOffset: Int
    let dtype: String
    let shape: [Int]

    static func parse(prefix: Data) throws -> Self {
        let bytes = [UInt8](prefix)
        guard bytes.count >= 10, Array(bytes.prefix(6)) == [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] else {
            throw EmbedANEError.invalidNPY("missing NPY magic/header")
        }
        guard bytes[6] == 1, bytes[7] == 0 else { throw EmbedANEError.invalidNPY("only NPY v1.0 is accepted") }
        let length = Int(bytes[8]) | (Int(bytes[9]) << 8)
        let offset = 10 + length
        guard length > 0, offset <= bytes.count, offset % 16 == 0, bytes[offset - 1] == 10 else {
            throw EmbedANEError.invalidNPY("truncated, unaligned or unterminated header")
        }
        var scanner = HeaderScanner(bytes: Array(bytes[10..<offset]))
        let (dtype, shape) = try scanner.parse()
        return Self(payloadOffset: offset, dtype: dtype, shape: shape)
    }
}

private struct HeaderScanner {
    let bytes: [UInt8]
    var position = 0
    mutating func whitespace() {
        while position < bytes.count, [9, 10, 13, 32].contains(bytes[position]) { position += 1 }
    }
    mutating func accept(_ byte: UInt8) -> Bool {
        whitespace()
        guard position < bytes.count, bytes[position] == byte else { return false }
        position += 1; return true
    }
    mutating func expect(_ byte: UInt8) throws {
        guard accept(byte) else { throw EmbedANEError.invalidNPY("malformed Python dictionary header") }
    }
    mutating func string() throws -> String {
        whitespace()
        guard position < bytes.count, bytes[position] == 39 || bytes[position] == 34 else { throw EmbedANEError.invalidNPY("expected quoted header string") }
        let quote = bytes[position]; position += 1; let start = position
        while position < bytes.count, bytes[position] != quote {
            guard bytes[position] >= 32, bytes[position] < 127, bytes[position] != 92 else { throw EmbedANEError.invalidNPY("non-ASCII/escaped header string") }
            position += 1
        }
        guard position < bytes.count else { throw EmbedANEError.invalidNPY("unterminated string") }
        let value = String(decoding: bytes[start..<position], as: UTF8.self); position += 1; return value
    }
    mutating func integer() throws -> Int {
        whitespace(); let start = position
        while position < bytes.count, (48...57).contains(bytes[position]) { position += 1 }
        guard start < position, let value = Int(String(decoding: bytes[start..<position], as: UTF8.self)) else { throw EmbedANEError.invalidNPY("invalid shape integer") }
        return value
    }
    mutating func parse() throws -> (String, [Int]) {
        try expect(123)
        var keys = Set<String>(); var shape: [Int]?; var dtype: String?
        while !accept(125) {
            let key = try string()
            guard keys.insert(key).inserted else { throw EmbedANEError.invalidNPY("duplicate header key") }
            try expect(58)
            switch key {
            case "descr":
                dtype = try string()
            case "fortran_order":
                whitespace(); let literal = Array("False".utf8)
                guard position + literal.count <= bytes.count, Array(bytes[position..<(position + literal.count)]) == literal else { throw EmbedANEError.invalidNPY("Fortran order is unsupported") }
                position += literal.count
            case "shape":
                try expect(40)
                var dimensions = [try integer()]
                // A rank-one Python tuple requires its trailing comma.
                try expect(44)
                while !accept(41) {
                    guard dimensions.count < 8 else { throw EmbedANEError.invalidNPY("excessive rank") }
                    dimensions.append(try integer())
                    if accept(41) { break }
                    try expect(44)
                }
                shape = dimensions
            default: throw EmbedANEError.invalidNPY("unknown header key")
            }
            if accept(125) { break }
            try expect(44)
        }
        whitespace()
        guard position == bytes.count, keys == Set(["descr", "fortran_order", "shape"]), let dtype, let shape else { throw EmbedANEError.invalidNPY("incomplete header or trailing code") }
        return (dtype, shape)
    }
}
