import CoreML
import Darwin
import Foundation
import Testing
@testable import EmbedANECore

private struct SparseTableFixture {
    let root: URL
    let file: URL
    let offset: Int
    init(dictionary: String = "{'descr': '<f2', 'fortran_order': False, 'shape': (248078, 2048), }") throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        file = root.appendingPathComponent(MappedEmbeddingTable.filename)
        var body = dictionary
        body += String(repeating: " ", count: (16 - ((10 + body.utf8.count + 1) % 16)) % 16) + "\n"
        var header = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0,
                           UInt8(body.utf8.count & 255), UInt8(body.utf8.count >> 8)])
        header.append(Data(body.utf8)); offset = header.count
        try header.write(to: file)
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        try handle.truncate(atOffset: UInt64(offset) + UInt64(ModelABI.tablePayloadBytes))
        for (id, bits) in [(0, UInt16(0x3c00)), (7, UInt16(0x3555)), (248077, UInt16(0x4000))] {
            try handle.seek(toOffset: UInt64(offset + id * 4096))
            var row = Data()
            for _ in 0..<2048 { row.append(UInt8(bits & 255)); row.append(UInt8(bits >> 8)) }
            try handle.write(contentsOf: row)
        }
    }
    func remove() { try? FileManager.default.removeItem(at: root) }
}

struct MappedEmbeddingTableTests {
    @Test func injectedTableURLAndVisionSplicePreserveTextAndPaddingRows() throws {
        let fixture = try SparseTableFixture(); defer { fixture.remove() }
        let url = fixture.root.appendingPathComponent("injected-table.npy")
        try FileManager.default.moveItem(at: fixture.file, to: url)
        let table = try MappedEmbeddingTable(url: url)
        let engine = CoreMLCascadeEngine(models: [], table: table, variant: .extrope)
        let input = try MultimodalTokenization.prepare(text: "fixture", grid: VisionGrid(height: 2, width: 2), tokenizer: VisionFixtureTokenizer(ids: [7]))
        let hidden = try engine.lookup(input) // CPU-only mmap/lookup, no graph call.
        let tokens = try VisionTokens(values: Array(repeating: Float16(3), count: 2048), count: 1)
        try VisionTokenSplicer.splice(tokens, input: input, into: hidden)
        let pointer = hidden.dataPointer.assumingMemoryBound(to: UInt16.self)
        // <|im_start|> user <vision_start> image <vision_end> 7 <|im_end|> <embedding> pad...
        #expect(pointer[3 * 2048] == Float16(3).bitPattern) // image
        #expect(pointer[5 * 2048] == 0x3555) // text id 7, exact fp16 bits
        #expect(pointer[7 * 2048] == 0x4000) // <embedding>
        #expect(pointer[8 * 2048] == 0x3c00) // id-0 table row, not zero hidden
        #expect(pointer[hidden.count - 1] == 0x3c00)
    }
    @Test func copiesRawFP16BitsIncludingPaddingRowZero() throws {
        let fixture = try SparseTableFixture(); defer { fixture.remove() }
        let table = try MappedEmbeddingTable(directory: SecureDirectory(fixture.root))
        let input = try TokenizedInput(contentIDs: [7])
        var output = [UInt8](repeating: 0xaa, count: MappedEmbeddingTable.hiddenByteCount)
        try output.withUnsafeMutableBytes { try table.copyRows(ids: input.ids, into: $0) }
        #expect(Array(output[0..<4]) == [0x55, 0x35, 0x55, 0x35])
        #expect(Array(output[4096..<4100]) == [0x00, 0x40, 0x00, 0x40])
        #expect(Array(output[8192..<8196]) == [0x00, 0x3c, 0x00, 0x3c])
        #expect(Array(output.suffix(4)) == [0x00, 0x3c, 0x00, 0x3c])
        #expect(table.header.payloadOffset == fixture.offset)
    }
    @Test(arguments: [-1, 248078, Int.max])
    func invalidIDCannotPartiallyWrite(_ invalidID: Int) throws {
        let fixture = try SparseTableFixture(); defer { fixture.remove() }
        let table = try MappedEmbeddingTable(directory: SecureDirectory(fixture.root))
        var ids = Array(repeating: 7, count: 512); ids[511] = invalidID
        var output = [UInt8](repeating: 0x55, count: MappedEmbeddingTable.hiddenByteCount)
        #expect(throws: EmbedANEError.self) {
            try output.withUnsafeMutableBytes { try table.copyRows(ids: ids, into: $0) }
        }
        #expect(output.allSatisfy { $0 == 0x55 })
    }
    @Test func rejectsWrongDestinationAndSequenceLengths() throws {
        let fixture = try SparseTableFixture(); defer { fixture.remove() }
        let table = try MappedEmbeddingTable(directory: SecureDirectory(fixture.root))
        var output = [UInt8](repeating: 0, count: 32)
        #expect(throws: EmbedANEError.self) {
            try output.withUnsafeMutableBytes { try table.copyRows(ids: Array(repeating: 0, count: 512), into: $0) }
        }
        output = Array(repeating: 0, count: MappedEmbeddingTable.hiddenByteCount)
        #expect(throws: EmbedANEError.self) {
            try output.withUnsafeMutableBytes { try table.copyRows(ids: [0], into: $0) }
        }
    }
    @Test(arguments: ["truncated", "trailing", "v2", "symlink"])
    func rejectsInvalidFile(_ mutation: String) throws {
        let fixture = try SparseTableFixture(); defer { fixture.remove() }
        if mutation == "symlink" {
            let other = fixture.root.appendingPathComponent("other.npy")
            try FileManager.default.moveItem(at: fixture.file, to: other)
            try FileManager.default.createSymbolicLink(at: fixture.file, withDestinationURL: other)
        } else {
            let handle = try FileHandle(forWritingTo: fixture.file); defer { try? handle.close() }
            if mutation == "v2" { try handle.seek(toOffset: 6); try handle.write(contentsOf: Data([2])) }
            else {
                let size = UInt64(fixture.offset) + UInt64(ModelABI.tablePayloadBytes)
                try handle.truncate(atOffset: mutation == "trailing" ? size + 1 : size - 1)
            }
        }
        #expect(throws: EmbedANEError.self) { try MappedEmbeddingTable(directory: SecureDirectory(fixture.root)) }
    }
    @Test(arguments: ["{'descr': '<f4', 'fortran_order': False, 'shape': (248078, 2048), }",
                      "{'descr': '<f2', 'fortran_order': True, 'shape': (248078, 2048), }",
                      "{'descr': '<f2', 'fortran_order': False, 'shape': (248077, 2048), }"])
    func rejectsWrongNPYABI(_ dictionary: String) throws {
        let fixture = try SparseTableFixture(dictionary: dictionary); defer { fixture.remove() }
        #expect(throws: EmbedANEError.self) { try MappedEmbeddingTable(directory: SecureDirectory(fixture.root)) }
    }
}
