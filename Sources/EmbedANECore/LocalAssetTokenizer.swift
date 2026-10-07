import Foundation
import Tokenizers
import Hub

/// Synchronous counterpart of the pinned dependency's local-folder loader.
/// Uses its public PreTrainedTokenizer initializer (the same class as the
/// local-folder overload), not the Hub/network loader or tokenizer-class routing.
/// Construction and use belong on a dedicated blocking queue.
public struct LocalAssetTokenizer: ContentTokenizer {
    private let tokenizer: any Tokenizers.Tokenizer

    public init(assets: URL) throws {
        let directory = try SecureDirectory(assets)
        let configBytes = try directory.read("tokenizer_config.json")
        let tokenizerBytes = try directory.read("tokenizer.json", limit: 128 * 1_024 * 1_024)
        guard let config = try JSONSerialization.jsonObject(with: configBytes) as? [String: Any],
              let data = try JSONSerialization.jsonObject(with: tokenizerBytes) as? [String: Any] else {
            throw EmbedANEError.verification(path: "tokenizer.json", reason: "Tokenizer assets must be JSON objects.")
        }
        tokenizer = try PreTrainedTokenizer(tokenizerConfig: Config(config as [NSString: Any]), tokenizerData: Config(data as [NSString: Any]), strict: true)
        guard tokenizer.convertTokenToId("<embedding>") == ModelABI.embeddingToken else {
            throw EmbedANEError.verification(path: "tokenizer.json", reason: "<embedding> must have id 248077.")
        }
        try PromptTemplate.verify(tokenizer)
    }
    public func encodeContent(_ text: String) throws -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false)
    }
}
