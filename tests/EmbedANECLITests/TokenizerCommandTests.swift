import EmbedANECore
import Foundation
import Testing
@testable import embed_ane

struct TokenizerCommandTests {
    /// Synthetic tiny tokenizer. Besides `<embedding>` it carries the WeMM
    /// chat-template ids the loaders verify: `<|im_start|>`=248045,
    /// `<|im_end|>`=248046 and "user\n" → [846, 198] (via BPE merges); the
    /// byte-level newline spelling "Ċ" (U+010A) is also 198, as the loaders
    /// check the vocabulary ids of "user" and "Ċ".
    private func assets(at directory: URL, specialID: Int = 248077, imEndID: Int = 248046) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let configuration: [String: Any] = ["tokenizer_class": "GPT2TokenizerFast", "unk_token": "<unk>",
            "clean_up_tokenization_spaces": false, "add_bos_token": false, "add_eos_token": false]
        func special(_ id: Int, _ content: String) -> [String: Any] {
            ["id": id, "content": content, "special": true, "single_word": false, "lstrip": false, "rstrip": false, "normalized": false]
        }
        let data: [String: Any] = ["version": "1.0", "added_tokens": [
            special(248045, "<|im_start|>"), special(imEndID, "<|im_end|>"), special(specialID, "<embedding>")],
            "model": ["type": "BPE", "unk_token": "<unk>",
                      "vocab": ["a": 0, "b": 1, "<unk>": 2, "u": 3, "s": 4, "e": 5, "r": 6, "us": 7, "er": 8,
                                "user": 846, "\n": 198, "\u{010A}": 198, "<|im_start|>": 248045, "<|im_end|>": imEndID, "<embedding>": specialID],
                      "merges": ["u s", "e r", "us er"]]]
        try JSONSerialization.data(withJSONObject: configuration).write(to: directory.appendingPathComponent("tokenizer_config.json"))
        try JSONSerialization.data(withJSONObject: data).write(to: directory.appendingPathComponent("tokenizer.json"))
    }
    @Test func synchronousLoaderMatchesPinnedLocalFolderLoader() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try assets(at: directory)
        let sync = try LocalAssetTokenizer(assets: directory)
        let original = try await SwiftTransformersTokenizer(assets: directory)
        for text in ["", "a", "b", "ab", "user\n", "<embedding>", String(repeating: "a", count: 511)] {
            #expect(try sync.encodeContent(text) == original.encodeContent(text))
        }
        #expect(try sync.encodeContent("ab") == [0, 1])
        #expect(try sync.encodeContent("user\n") == [846, 198])
    }
    @Test func rejectsWrongSpecialID() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try assets(at: directory, specialID: 3)
        #expect(throws: EmbedANEError.self) { try LocalAssetTokenizer(assets: directory) }
    }
    @Test func rejectsWrongChatTemplateID() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        try assets(at: directory, imEndID: 248047)
        #expect(throws: EmbedANEError.self) { try LocalAssetTokenizer(assets: directory) }
        await #expect(throws: EmbedANEError.self) { try await SwiftTransformersTokenizer(assets: directory) }
    }
    @Test func commandNeedsNoModelAndPersistsPassingThenFailingGate() async throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: home) }
        let directory = home.appendingPathComponent("assets")
        try assets(at: directory)
        let input = try TokenizedInput(contentIDs: [0])
        let record = try JSONEncoder().encode(TokenizerGoldenCase(text: "a", ids: input.ids, mask: input.mask, nTokens: input.nTokens))
        var jsonl = Data()
        for _ in 0..<1000 { jsonl.append(record); jsonl.append(10) }
        let golden = home.appendingPathComponent("golden.jsonl"); try jsonl.write(to: golden)
        let store = ConfigurationStore(home: home)
        let args = ["verify-tokenizer", golden.path, "--assets", directory.path]
        let success = await CLIApplication.run(arguments: args, store: store, workingDirectory: home,
                                               environment: [:], executor: ProductionCommands())
        #expect(success.exitCode == 0)
        let receipt = try TokenizerGateReceipt.read(store: store)
        #expect(receipt.report.gatePassed); #expect(receipt.report.cases == 1000)
        #expect(receipt.report.assetDigests.count == 2)
        #expect(!FileManager.default.fileExists(atPath: home.appendingPathComponent(".embed-ane/models").path))
        // A later failed run replaces the old pass; it cannot leave stale trust.
        try record.write(to: golden)
        let failure = await CLIApplication.run(arguments: args, store: store, workingDirectory: home,
                                               environment: [:], executor: ProductionCommands())
        #expect(failure.exitCode == 2)
        #expect(try !TokenizerGateReceipt.read(store: store).report.gatePassed)
    }
}
