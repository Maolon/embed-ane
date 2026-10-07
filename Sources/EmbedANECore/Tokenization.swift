import Foundation
import Tokenizers

public protocol ContentTokenizer: Sendable {
    func encodeContent(_ text: String) throws -> [Int]
}

/// The local-folder overload is intentionally the only dependency API used:
/// it reads tokenizer assets and does not fetch a model from the Hub.
public struct SwiftTransformersTokenizer: ContentTokenizer {
    private let tokenizer: any Tokenizers.Tokenizer
    public init(assets: URL) async throws {
        let directory = try SecureDirectory(assets)
        _ = try directory.read("tokenizer_config.json")
        let tokenizerFD = try directory.openFile("tokenizer.json")
        let handle = FileHandle(fileDescriptor: tokenizerFD, closeOnDealloc: true)
        try handle.close()
        tokenizer = try await AutoTokenizer.from(modelFolder: assets, strict: true)
        guard tokenizer.convertTokenToId("<embedding>") == ModelABI.embeddingToken else {
            throw EmbedANEError.verification(path: "tokenizer.json", reason: "<embedding> must have id 248077")
        }
        try PromptTemplate.verify(tokenizer)
    }
    public func encodeContent(_ text: String) throws -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }
}

/// The upstream WeMM embedding prompt for one user turn: the Qwen chat template
/// `<|im_start|>user\n … <|im_end|>`, after the reference processor's
/// post-processing (no role newline before a leading image), followed by
/// `<embedding>` (appended by `TokenizedInput`). Without it, fp32 reference
/// embeddings move to cosine 0.86–0.96 of the upstream ones.
public enum PromptTemplate {
    public enum Lead: Sendable { case text, image, video }

    /// Text in the upstream prompt. Upstream tokenizes `"user\n" + text` as one
    /// segment. Encoding the text on its own gives the same ids unless the text
    /// opens with whitespace containing a newline: that run then merges with the
    /// role newline (e.g. "\n\n" is one token), so it is encoded jointly.
    public static func text(_ text: String, tokenizer: any ContentTokenizer, index: Int = 0) throws -> TokenizedInput {
        guard text.prefix(while: \.isWhitespace).contains(where: \.isNewline) else {
            return try textInput(tokenizer.encodeContent(text), index: index)
        }
        let joined = try tokenizer.encodeContent("user\n" + text)
        guard joined.starts(with: ModelABI.userRoleTokens), joined.count > ModelABI.userRoleTokens.count else {
            throw EmbedANEError.verification(path: "tokenizer.json", reason: "The role line did not tokenize as expected.")
        }
        // Role "user" and the newline run are prompt tokens; the rest is content.
        let content = joined.count - ModelABI.userRoleTokens.count - 1
        guard content <= ModelABI.textContentLimit else {
            throw EmbedANEError.inputTooLong(index: index, contentTokens: content)
        }
        return try TokenizedInput(contentIDs: [ModelABI.imStartToken] + joined + [ModelABI.imEndToken], index: index)
    }

    /// A text or token-array input: checks the user's own token count against
    /// `ModelABI.textContentLimit`, then wraps it.
    public static func textInput(_ content: [Int], index: Int = 0) throws -> TokenizedInput {
        guard content.count <= ModelABI.textContentLimit else {
            throw EmbedANEError.inputTooLong(index: index, contentTokens: content.count)
        }
        return try TokenizedInput(contentIDs: wrap(content, lead: .text), index: index)
    }

    public static func wrap(_ content: [Int], lead: Lead) -> [Int] {
        prefix(lead) + content + [ModelABI.imEndToken]
    }
    /// Tokens the wrapper adds around content, excluding `<embedding>`.
    public static func overhead(_ lead: Lead) -> Int { prefix(lead).count + 1 }

    static func prefix(_ lead: Lead) -> [Int] {
        [ModelABI.imStartToken] + ModelABI.userRoleTokens + (lead == .image ? [] : [ModelABI.newlineToken])
    }
    /// Fails loading when the vocabulary does not hold the wrapper ids. Checked
    /// by vocabulary lookup: encoding "user\n" alone drops the newline, because
    /// the tokenizer's normalizer removes one trailing "\n" per segment.
    static func verify(_ tokenizer: any Tokenizers.Tokenizer) throws {
        guard tokenizer.convertTokenToId("<|im_start|>") == ModelABI.imStartToken,
              tokenizer.convertTokenToId("<|im_end|>") == ModelABI.imEndToken,
              tokenizer.convertTokenToId("user") == ModelABI.userRoleTokens.first,
              tokenizer.convertTokenToId("\u{010A}") == ModelABI.newlineToken else {
            throw EmbedANEError.verification(path: "tokenizer.json", reason: "Chat-template tokens do not match the WeMM tokenizer.")
        }
    }
}

public struct TokenizedInput: Codable, Sendable, Equatable {
    public let ids: [Int]
    public let mask: [Float]
    public let nTokens: Int
    enum CodingKeys: String, CodingKey { case ids, mask, nTokens = "n_tokens" }
    public init(contentIDs: [Int], index: Int = 0) throws {
        guard contentIDs.count <= 511 else { throw EmbedANEError.inputTooLong(index: index, contentTokens: contentIDs.count) }
        guard contentIDs.allSatisfy({ $0 >= 0 && $0 < ModelABI.vocabulary }) else { throw EmbedANEError.invalidRequest("Tokenizer returned an out-of-vocabulary id.", param: "input[\(index)]") }
        nTokens = contentIDs.count + 1
        ids = contentIDs + [ModelABI.embeddingToken] + Array(repeating: 0, count: 512 - nTokens)
        mask = Array(repeating: 1, count: nTokens) + Array(repeating: 0, count: 512 - nTokens)
    }
    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let ids = try c.decode([Int].self, forKey: .ids); let mask = try c.decode([Float].self, forKey: .mask)
        let n = try c.decode(Int.self, forKey: .nTokens)
        guard (1...512).contains(n), ids.count == 512, mask.count == 512 else { throw EmbedANEError.invalidRequest("Malformed tokenized input.", param: "input") }
        try self.init(contentIDs: Array(ids.prefix(n - 1)))
        guard self.ids == ids, self.mask == mask else { throw EmbedANEError.invalidRequest("Noncanonical token padding, mask or special token.", param: "input") }
    }
    /// Text requests in the upstream prompt format.
    public static func prepare(_ texts: [String], tokenizer: any ContentTokenizer) throws -> [Self] {
        try texts.enumerated().map { try PromptTemplate.text($0.element, tokenizer: tokenizer, index: $0.offset) }
    }
}

public struct TokenizerGoldenCase: Codable, Sendable {
    public let text: String
    public let ids: [Int]
    public let mask: [Float]
    public let nTokens: Int
    enum CodingKeys: String, CodingKey { case text, ids, mask, nTokens = "n_tokens" }
    public init(text: String, ids: [Int], mask: [Float], nTokens: Int) {
        self.text = text; self.ids = ids; self.mask = mask; self.nTokens = nTokens
    }
}

public struct TokenizerVerificationReport: Codable, Sendable {
    public struct Mismatch: Codable, Sendable { public let line: Int; public let reason: String }
    public let cases: Int
    public let mismatches: Int
    public let rejectedOverlengthCases: Int
    public let minimumCases: Int
    public let gatePassed: Bool
    public let examples: [Mismatch]
    public let assetDigests: [String: String]
}

public enum TokenizerDifferential {
    /// Overlength records use the reference's untruncated ids/mask/count. They
    /// must match raw tokenization AND be rejected by the 511-content-token policy.
    public static func verify(jsonl: Data, tokenizer: any ContentTokenizer, minimumCases: Int = 1_000, assetDigests: [String: String] = [:]) throws -> TokenizerVerificationReport {
        guard minimumCases > 0, let text = String(data: jsonl, encoding: .utf8) else { throw EmbedANEError.invalidRequest("Golden JSONL must be UTF-8.", param: nil) }
        var count = 0; var failures = 0; var rejections = 0; var examples: [TokenizerVerificationReport.Mismatch] = []
        for (lineNumber, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
            count += 1
            var reason: String?
            do {
                let expected = try JSONDecoder().decode(TokenizerGoldenCase.self, from: Data(line.utf8))
                let content = try tokenizer.encodeContent(expected.text)
                if content.count > 511 {
                    let raw = content + [ModelABI.embeddingToken]
                    if expected.ids != raw || expected.mask != Array(repeating: 1, count: raw.count) || expected.nTokens != raw.count {
                        reason = "overlength raw ids/mask/count mismatch"
                    } else {
                        do { _ = try TokenizedInput(contentIDs: content); reason = "overlength input was accepted" }
                        catch EmbedANEError.inputTooLong { rejections += 1 }
                        catch { reason = "wrong overlength error: \(error)" }
                    }
                } else {
                    let actual = try TokenizedInput(contentIDs: content)
                    if actual.ids != expected.ids { reason = "ids mismatch" }
                    else if actual.mask != expected.mask { reason = "mask mismatch" }
                    else if actual.nTokens != expected.nTokens { reason = "n_tokens mismatch" }
                }
            } catch { reason = "invalid record or tokenizer error: \(error)" }
            if let reason {
                failures += 1
                if examples.count < 100 { examples.append(.init(line: lineNumber + 1, reason: reason)) }
            }
        }
        return .init(cases: count, mismatches: failures, rejectedOverlengthCases: rejections, minimumCases: minimumCases,
                     gatePassed: count >= minimumCases && failures == 0, examples: examples, assetDigests: assetDigests)
    }
}
