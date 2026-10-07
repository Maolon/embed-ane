import Darwin
import Foundation

/// Queue-confined, read-only NPY mapping. No fp32 table or Data-sized table copy
/// is ever created. Callers must retain their installation lease for its lifetime.
final class MappedEmbeddingTable {
    static let filename = "embed_table.fp16.npy"
    static let hiddenByteCount = ModelABI.sequenceLength * ModelABI.dimension * 2
    let header: NPYHeader
    private let mapping: UnsafeMutableRawPointer
    private let mappingSize: Int

    convenience init(url: URL) throws {
        try self.init(directory: SecureDirectory(url.deletingLastPathComponent()), filename: url.lastPathComponent)
    }

    init(directory: SecureDirectory, filename: String = MappedEmbeddingTable.filename) throws {
        let fd = try directory.openFile(filename)
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_size > 0, info.st_size <= Int64(Int.max) else {
            throw EmbedANEError.invalidNPY("Cannot determine the table file length.")
        }
        // A v1 header is at most UInt16.max bytes, plus its ten-byte prefix.
        var prefix = Data(count: Int(min(info.st_size, 65_545)))
        let prefixCount = prefix.count
        let received = prefix.withUnsafeMutableBytes { buffer -> Int in
            guard let address = buffer.baseAddress else { return 0 }
            var total = 0
            while total < prefixCount {
                let n = pread(fd, address.advanced(by: total), prefixCount - total, off_t(total))
                if n < 0, errno == EINTR { continue }
                if n <= 0 { return total }
                total += n
            }
            return total
        }
        prefix.count = received
        header = try NPYHeader.parse(prefix: prefix, fileByteCount: Int64(info.st_size))
        let size = Int(info.st_size)
        guard let pointer = mmap(nil, size, PROT_READ, MAP_PRIVATE, fd, 0), pointer != MAP_FAILED else {
            throw EmbedANEError.io(path: Self.filename, reason: "Read-only mmap failed.")
        }
        mapping = pointer; mappingSize = size
    }
    deinit { munmap(mapping, mappingSize) }

    func copyRows(ids: [Int], into destination: UnsafeMutableRawBufferPointer) throws {
        guard ids.count == ModelABI.sequenceLength, destination.count == Self.hiddenByteCount,
              let output = destination.baseAddress else {
            throw EmbedANEError.invalidNPY("Lookup requires 512 ids and a contiguous fp16 [1,512,2048] destination.")
        }
        // Validate the ENTIRE id vector before any destination write.
        try header.validate(ids: ids)
        let rowBytes = ModelABI.dimension * 2
        for (position, id) in ids.enumerated() {
            memcpy(output.advanced(by: position * rowBytes),
                   mapping.advanced(by: header.payloadOffset + id * rowBytes), rowBytes)
        }
    }
}
