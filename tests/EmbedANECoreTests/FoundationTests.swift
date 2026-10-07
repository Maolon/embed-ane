import Foundation
import Testing
@testable import EmbedANECore

private struct TinyTokenizer: ContentTokenizer {
    func encodeContent(_ text: String) throws -> [Int] { text.unicodeScalars.map { Int($0.value) % 200_000 } }
}

private final class TemporaryDirectory {
    let url: URL
    init() throws {
        url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }
    deinit { try? FileManager.default.removeItem(at: url) }
}

private func modelYAML() -> String {
    let digest = String(repeating: "a", count: 64)
    let paths = ["tokenizer.json", "tokenizer_config.json", "embed_table.fp16.npy"] + (0..<6).map { "chunks/chunk\($0).mlmodelc/manifest.txt" }
    return """
    spec_version: 1
    model: {id: wemm-embedding-2b-ane, dim: 2048, max_seq: 512, normalize: l2}
    source: {repo: owner/model, revision: v1}
    files:
    \(paths.map { "  - {path: \($0), sha256: \(digest), size: 1}" }.joined(separator: "\n"))
    runtime: {compute_units: cpu_and_ne, pad_side: right, mask_dtype: fp32}
    """
}

@Suite("Binding schema and provenance") struct SchemaTests {
    @Test func validSpecAndRevisionRoundTrip() throws {
        let spec = try ModelSpec.parse(modelYAML())
        #expect(spec.specVersion == 1)
        #expect(spec.source.endpoint == "https://huggingface.co")
        #expect(spec.source.revision == .tag("v1"))
        let decoded = try JSONDecoder().decode(ModelSpec.self, from: JSONEncoder().encode(spec))
        #expect(decoded == spec)
        #expect(try Revision(String(repeating: "a", count: 40)) == .commit(String(repeating: "a", count: 40)))
    }
    @Test(arguments: [
        ("spec_version: 1", "spec_version: 2"),
        ("dim: 2048", "dim: 2047"),
        ("max_seq: 512", "max_seq: 513"),
        ("normalize: l2", "normalize: none"),
        ("id: wemm-embedding-2b-ane", "id: ../escape"),
        ("id: wemm-embedding-2b-ane", "id: Uppercase"),
        ("revision: v1", "revision: ../main"),
        ("repo: owner/model", "repo: owner/model/extra"),
        ("pad_side: right", "pad_side: left"),
        ("mask_dtype: fp32", "mask_dtype: fp16"),
        ("compute_units: cpu_and_ne", "compute_units: all"),
        ("dim: 2048", "dim: 2048, typo: true"),
        ("revision: v1", "revision: v1, branch: main"),
        ("size: 1", "size: -1"),
    ]) func rejectsBadSchema(change: (String, String)) {
        #expect(throws: (any Error).self) { try ModelSpec.parse(modelYAML().replacingOccurrences(of: change.0, with: change.1)) }
    }
    @Test func rejectsUnknownAndDuplicateRootKeys() {
        #expect(throws: (any Error).self) { try ModelSpec.parse(modelYAML() + "\nunknown: 1\n") }
        #expect(throws: (any Error).self) { try ModelSpec.parse(modelYAML() + "\nspec_version: 1\n") }
    }
    @Test func provenanceDetectsSpecMutation() throws {
        let spec = try ModelSpec.parse(modelYAML())
        let manifest = try InstallManifest(specYAML: modelYAML(), spec: spec, resolvedCommit: String(repeating: "b", count: 40))
        let encoded = try manifest.yaml()
        #expect(try StrictYAML.decode(InstallManifest.self, from: encoded).specDigest == manifest.specDigest)
        #expect(throws: (any Error).self) { try StrictYAML.decode(InstallManifest.self, from: encoded.replacingOccurrences(of: "owner/model", with: "owner/other")) }
    }
}

@Suite("Safe artifact verification") struct VerificationTests {
    @Test(arguments: ["../x", "/absolute", "a/../b", "a//b", "a/./b", "a\\b", "x\n", "x y", "x:y"])
    func rejectsTraversal(path: String) { #expect(throws: EmbedANEError.self) { try SafePath.validate(path) } }

    @Test func sha256KnownVectorAndCorruptedByte() throws {
        #expect(FileDigest.sha256(Data("abc".utf8)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        let temp = try TemporaryDirectory(); let dir = try SecureDirectory(temp.url)
        let data = Data("abc".utf8)
        try dir.atomicWrite(data, to: "nested/value")
        let file = try ArtifactFile(path: "nested/value", sha256: FileDigest.sha256(data), size: 3)
        try FileDigest.verify(file, in: dir)
        try dir.atomicWrite(Data("abd".utf8), to: "nested/value")
        #expect(throws: EmbedANEError.self) { try FileDigest.verify(file, in: dir) }
    }
    @Test func rejectsLeafAndIntermediateSymlinks() throws {
        let temp = try TemporaryDirectory(); let outside = try TemporaryDirectory()
        try Data("secret".utf8).write(to: outside.url.appendingPathComponent("data"))
        try FileManager.default.createSymbolicLink(at: temp.url.appendingPathComponent("link"), withDestinationURL: outside.url)
        try FileManager.default.createSymbolicLink(at: temp.url.appendingPathComponent("leaf"), withDestinationURL: outside.url.appendingPathComponent("data"))
        let dir = try SecureDirectory(temp.url)
        #expect(throws: (any Error).self) { try dir.read("link/data") }
        #expect(throws: (any Error).self) { try dir.read("leaf") }
    }
    @Test func innerManifestNestedVerificationAndUnexpectedFile() throws {
        let temp = try TemporaryDirectory(); let dir = try SecureDirectory(temp.url)
        let data = Data("weights".utf8)
        let text = "sub/weights.bin \(data.count) \(FileDigest.sha256(data))\n"
        try dir.atomicWrite(data, to: "sub/weights.bin")
        try dir.atomicWrite(Data(text.utf8), to: "manifest.txt")
        let manifest = try InnerManifest(data: Data(text.utf8))
        try manifest.verify(in: temp.url)
        try dir.atomicWrite(Data(), to: "unexpected")
        #expect(throws: EmbedANEError.self) { try manifest.verify(in: temp.url) }
    }
    @Test func innerManifestRejectsDuplicateTraversalAndUnsortedPaths() throws {
        let digest = FileDigest.sha256(Data())
        for text in ["x 0 \(digest)\nx 0 \(digest)\n", "../x 0 \(digest)\n", "z 0 \(digest)\na 0 \(digest)\n", "x 0 \(digest)\r\n", "manifest.txt 0 \(digest)\n", "x 0 \(digest)\nX 0 \(digest)\n"] {
            #expect(throws: (any Error).self) { try InnerManifest(data: Data(text.utf8)) }
        }
    }
}

private func npyPrefix(_ dictionary: String = "{'descr': '<f2', 'fortran_order': False, 'shape': (248078, 2048), }") -> Data {
    var header = Array(dictionary.utf8)
    while (10 + header.count + 1) % 16 != 0 { header.append(32) }
    header.append(10)
    return Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, UInt8(header.count & 255), UInt8(header.count >> 8)] + header)
}

@Suite("Frozen NPY ABI") struct NPYTests {
    @Test func validHeaderAndIDRange() throws {
        let data = npyPrefix(); let header = try NPYHeader.parse(prefix: data, fileByteCount: Int64(data.count) + ModelABI.tablePayloadBytes)
        #expect(header.rows == 248_078); #expect(header.columns == 2_048)
        try header.validate(ids: [0, 248_077])
        #expect(throws: EmbedANEError.self) { try header.validate(ids: [248_078]) }
        #expect(throws: EmbedANEError.self) { try header.validate(ids: [-1]) }
    }
    @Test(arguments: [
        "{'descr': '<f4', 'fortran_order': False, 'shape': (248078, 2048)}",
        "{'descr': '>f2', 'fortran_order': False, 'shape': (248078, 2048)}",
        "{'descr': '<f2', 'fortran_order': True, 'shape': (248078, 2048)}",
        "{'descr': '<f2', 'fortran_order': False, 'shape': (248077, 2048)}",
        "{'descr': '<f2', 'fortran_order': False, 'shape': (1, 248078, 2048)}",
        "{'descr': '<f2', 'fortran_order': False, 'shape': (248078, 2048), 'evil': 1}",
    ]) func rejectsWrongABI(dictionary: String) {
        let data = npyPrefix(dictionary)
        #expect(throws: EmbedANEError.self) { try NPYHeader.parse(prefix: data, fileByteCount: Int64(data.count) + ModelABI.tablePayloadBytes) }
    }
    @Test func rejectsVersionTruncationAndTrailingBytes() {
        var data = npyPrefix(); let size = Int64(data.count) + ModelABI.tablePayloadBytes
        #expect(throws: EmbedANEError.self) { try NPYHeader.parse(prefix: Data(data.dropLast()), fileByteCount: size) }
        #expect(throws: EmbedANEError.self) { try NPYHeader.parse(prefix: data, fileByteCount: size - 1) }
        #expect(throws: EmbedANEError.self) { try NPYHeader.parse(prefix: data, fileByteCount: size + 1) }
        data[6] = 2
        #expect(throws: EmbedANEError.self) { try NPYHeader.parse(prefix: data, fileByteCount: size) }
        data[6] = 3
        #expect(throws: EmbedANEError.self) { try NPYHeader.parse(prefix: data, fileByteCount: size) }
    }
}

@Suite("Tokenizer policy and differential harness") struct TokenizerTests {
    @Test func exactSpecialZeroPaddingAndMask() throws {
        let value = try TokenizedInput(contentIDs: [7, 9])
        #expect(value.nTokens == 3); #expect(value.ids.count == 512)
        #expect(Array(value.ids.prefix(4)) == [7, 9, 248077, 0])
        #expect(Array(value.mask.prefix(4)) == [1, 1, 1, 0])
        #expect(!value.ids.contains(248044))
        #expect(try TokenizedInput(contentIDs: []).nTokens == 1)
    }
    @Test(arguments: [510, 511, 512, 513]) func contentBoundary(count: Int) throws {
        if count <= 511 { #expect(try TokenizedInput(contentIDs: Array(repeating: 1, count: count)).nTokens == count + 1) }
        else { #expect(throws: EmbedANEError.self) { try TokenizedInput(contentIDs: Array(repeating: 1, count: count)) } }
    }
    @Test func syntheticGoldenGateAndMismatchDetection() throws {
        let tokenizer = TinyTokenizer(); var jsonl = Data()
        for index in 0..<1_000 {
            let text = ["中文", "hello", " \t\n", "é🧪", ""][index % 5]
            let value = try TokenizedInput(contentIDs: tokenizer.encodeContent(text))
            let golden = TokenizerGoldenCase(text: text, ids: value.ids, mask: value.mask, nTokens: value.nTokens)
            jsonl.append(try JSONEncoder().encode(golden)); jsonl.append(10)
        }
        let report = try TokenizerDifferential.verify(jsonl: jsonl, tokenizer: tokenizer)
        #expect(report.gatePassed); #expect(report.cases == 1_000); #expect(report.mismatches == 0)
        let bad = TokenizerGoldenCase(text: "x", ids: Array(repeating: 0, count: 512), mask: Array(repeating: 0, count: 512), nTokens: 1)
        jsonl.append(try JSONEncoder().encode(bad)); jsonl.append(10)
        #expect(try !TokenizerDifferential.verify(jsonl: jsonl, tokenizer: tokenizer).gatePassed)
        #expect(try !TokenizerDifferential.verify(jsonl: Data(), tokenizer: tokenizer).gatePassed)
    }
    @Test func promptTemplateMatchesUpstreamSequences() throws {
        // "hello world" content ids from the WeMM tokenizer.
        let content = [14556, 1814]
        #expect(PromptTemplate.wrap(content, lead: .text) == [248045, 846, 198, 14556, 1814, 248046])
        #expect(PromptTemplate.wrap(content, lead: .image) == [248045, 846, 14556, 1814, 248046]) // no role newline
        #expect(PromptTemplate.overhead(.text) == 4); #expect(PromptTemplate.overhead(.image) == 3)
        let input = try TokenizedInput(contentIDs: PromptTemplate.wrap(content, lead: .text))
        #expect(Array(input.ids.prefix(input.nTokens)) == [248045, 846, 198, 14556, 1814, 248046, 248077])
        #expect(input.ids[input.nTokens] == 0)
        struct Fixed: ContentTokenizer { func encodeContent(_ text: String) throws -> [Int] { [14556, 1814] } }
        #expect(try TokenizedInput.prepare(["hello world"], tokenizer: Fixed()) == [input])
    }
    @Test func overlengthGoldenMustMatchRawTokensAndReject() throws {
        let text = String(repeating: "a", count: 512)
        let raw = try TinyTokenizer().encodeContent(text) + [ModelABI.embeddingToken]
        let golden = TokenizerGoldenCase(text: text, ids: raw, mask: Array(repeating: 1, count: raw.count), nTokens: raw.count)
        let report = try TokenizerDifferential.verify(jsonl: JSONEncoder().encode(golden), tokenizer: TinyTokenizer(), minimumCases: 1)
        #expect(report.gatePassed); #expect(report.rejectedOverlengthCases == 1)
    }
}
