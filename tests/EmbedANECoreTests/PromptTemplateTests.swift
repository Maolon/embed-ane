import Foundation
import Testing
@testable import EmbedANECore

/// Encodes from a fixed table, so joint and separate encodings are distinguishable.
private struct TableTokenizer: ContentTokenizer {
    let table: [String: [Int]]
    func encodeContent(_ text: String) throws -> [Int] {
        guard let ids = table[text] else { throw EmbedANEError.invalidRequest("unexpected text \(text.debugDescription)", param: nil) }
        return ids
    }
}

struct PromptTemplateTests {
    private let imStart = ModelABI.imStartToken, user = ModelABI.userRoleTokens[0]
    private let newline = ModelABI.newlineToken, imEnd = ModelABI.imEndToken, embedding = ModelABI.embeddingToken

    @Test func normalTextUsesTheFixedRolePrefix() throws {
        let tokenizer = TableTokenizer(table: ["hi": [10, 11], "user\nhi": [999]])
        let input = try PromptTemplate.text("hi", tokenizer: tokenizer)
        #expect(Array(input.ids.prefix(input.nTokens)) == [imStart, user, newline, 10, 11, imEnd, embedding])
        #expect(try TokenizedInput.prepare(["hi"], tokenizer: tokenizer) == [input])
        // Leading whitespace without a newline keeps the fixed prefix.
        let spaced = try PromptTemplate.text(" hi", tokenizer: TableTokenizer(table: [" hi": [12]]))
        #expect(Array(spaced.ids.prefix(spaced.nTokens)) == [imStart, user, newline, 12, imEnd, embedding])
    }

    @Test func leadingNewlineIsEncodedJointlyWithTheRoleLine() throws {
        // Upstream merges the role newline with the text's leading "\n" (271 = "\n\n").
        let tokenizer = TableTokenizer(table: ["user\n\nhi": [user, 271, 10], "\nhi": [198, 10]])
        let input = try PromptTemplate.text("\nhi", tokenizer: tokenizer)
        #expect(Array(input.ids.prefix(input.nTokens)) == [imStart, user, 271, 10, imEnd, embedding])
        #expect(try TokenizedInput.prepare(["\nhi"], tokenizer: tokenizer) == [input])
        // Whitespace run that contains a newline, not just a leading "\n".
        let mixed = try PromptTemplate.text(" \nhi", tokenizer: TableTokenizer(table: ["user\n \nhi": [user, 300, 10]]))
        #expect(Array(mixed.ids.prefix(mixed.nTokens)) == [imStart, user, 300, 10, imEnd, embedding])
        // A joint encoding that does not start with the role token is rejected.
        #expect(throws: EmbedANEError.self) {
            try PromptTemplate.text("\nhi", tokenizer: TableTokenizer(table: ["user\n\nhi": [5, 271, 10]]))
        }
    }

    @Test func inputTooLongCountsOnlyUserContentTokens() throws {
        let limit = ModelABI.textContentLimit
        // Fixed-prefix path: content is exactly the text's tokens.
        let long = TableTokenizer(table: ["x": Array(repeating: 10, count: limit + 1)])
        #expect(throws: EmbedANEError.inputTooLong(index: 3, contentTokens: limit + 1)) {
            try PromptTemplate.text("x", tokenizer: long, index: 3)
        }
        let atLimit = try PromptTemplate.text("x", tokenizer: TableTokenizer(table: ["x": Array(repeating: 10, count: limit)]))
        #expect(atLimit.nTokens == limit + 5)
        // Joint path: "user" and the merged newline run are prompt tokens, not content.
        let joint = TableTokenizer(table: ["user\n\nx": [user, 271] + Array(repeating: 10, count: limit + 1)])
        #expect(throws: EmbedANEError.inputTooLong(index: 1, contentTokens: limit + 1)) {
            try PromptTemplate.text("\nx", tokenizer: joint, index: 1)
        }
        let jointAtLimit = try PromptTemplate.text("\nx", tokenizer: TableTokenizer(table: ["user\n\nx": [user, 271] + Array(repeating: 10, count: limit)]))
        #expect(jointAtLimit.nTokens == limit + 5)
    }
}
