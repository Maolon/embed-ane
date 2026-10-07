import CoreML
import Foundation
import Testing
@testable import EmbedANECore

/// Upstream WeMM processor output for a sampled video (no text), recorded with
/// HF `get_rope_index`; see `source` in the fixture.
private struct VideoPromptGolden: Decodable {
    let source: String
    let inputIDs: [Int]
    let pairs: Int
    let patchGrid: [Int]
    let positions: [[Int]]
    enum CodingKeys: String, CodingKey {
        case source, inputIDs = "input_ids", pairs, patchGrid = "patch_grid", positions
    }

    static func load() throws -> Self {
        let url = try #require(Bundle.module.url(forResource: "video-prompt-golden", withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
    }

    /// Timestamp text ids before each vision_start: after the role newline for
    /// the first pair, after the previous vision_end for the others.
    func timestampPrefixes() throws -> [[Int]] {
        let abi = VisionABIConstants.wemm
        let lead = [ModelABI.imStartToken] + ModelABI.userRoleTokens + [ModelABI.newlineToken]
        try #require(Array(inputIDs.prefix(lead.count)) == lead)
        var prefixes: [[Int]] = [], start = lead.count
        for (index, id) in inputIDs.enumerated() {
            if id == abi.visionStartToken { prefixes.append(Array(inputIDs[start..<index])) }
            if id == abi.visionEndToken { start = index + 1 }
        }
        return prefixes
    }
}

struct VideoPromptTests {
    @Test func videoPromptIDsAndPositionsMatchUpstreamGolden() throws {
        let golden = try VideoPromptGolden.load()
        let abi = VisionABIConstants.wemm
        #expect(golden.inputIDs.count == 488 && golden.positions.count == 488 && golden.pairs == 4)
        // The golden prompt already ends with <|im_end|> <embedding>.
        #expect(Array(golden.inputIDs.suffix(2)) == [ModelABI.imEndToken, ModelABI.embeddingToken])

        let grid = try VisionGrid(height: golden.patchGrid[0], width: golden.patchGrid[1])
        let prefixes = try golden.timestampPrefixes()
        #expect(prefixes.count == golden.pairs)
        #expect(prefixes.allSatisfy { !$0.isEmpty })
        let spans = prefixes.map { VisualSpan(grid: grid, padToken: abi.videoToken, prefixIDs: $0) }
        let input = try MultimodalTokenization.prepare(text: "", spans: spans, lead: .video,
                                                       tokenizer: VisionFixtureTokenizer(ids: []))

        // TokenizedInput appends <embedding> itself, so the whole golden prompt
        // (including its trailing <embedding>) is the active prefix.
        #expect(input.nTokens == golden.inputIDs.count)
        #expect(Array(input.ids.prefix(input.nTokens)) == golden.inputIDs)
        #expect(input.ids.dropFirst(input.nTokens).allSatisfy { $0 == 0 })
        #expect(input.ids.filter { $0 == abi.videoToken }.count == golden.pairs * grid.tokenCount)

        let positions = try Mrope.positions(input: input, grids: Array(repeating: grid, count: golden.pairs))
        #expect(positions.count == ModelABI.sequenceLength)
        let actual = positions.prefix(golden.positions.count).map { [$0.temporal, $0.height, $0.width] }
        #expect(actual == golden.positions)
        if let mismatch = zip(actual, golden.positions).enumerated().first(where: { $0.element.0 != $0.element.1 }) {
            Issue.record("First position mismatch at \(mismatch.offset): \(mismatch.element.0) vs \(mismatch.element.1)")
        }
        // A grid count that disagrees with the spans is rejected.
        #expect(throws: EmbedANEError.self) { try Mrope.positions(input: input, grids: Array(repeating: grid, count: 3)) }
    }

    @Test func multiSpanSpliceWritesEachBlockAtItsOwnRowsAndRejectsBeforeWriting() throws {
        let abi = VisionABIConstants.wemm
        let small = try VisionGrid(height: 2, width: 2) // 1 token
        let large = try VisionGrid(height: 2, width: 4) // 2 tokens
        let spans = [VisualSpan(grid: small, padToken: abi.videoToken, prefixIDs: [11]),
                     VisualSpan(grid: large, padToken: abi.videoToken, prefixIDs: [12, 13])]
        let input = try MultimodalTokenization.prepare(text: "", spans: spans, lead: .video, tokenizer: VisionFixtureTokenizer(ids: []))
        // Rows: 0 im_start, 1 user, 2 \n, 3 ts, 4 vision_start, 5 pad, 6 vision_end,
        // 7-8 ts, 9 vision_start, 10-11 pad, 12 vision_end, 13 im_end, 14 <embedding>
        #expect(Array(input.ids.prefix(input.nTokens)) == [ModelABI.imStartToken] + ModelABI.userRoleTokens + [ModelABI.newlineToken,
            11, abi.visionStartToken, abi.videoToken, abi.visionEndToken,
            12, 13, abi.visionStartToken, abi.videoToken, abi.videoToken, abi.visionEndToken,
            ModelABI.imEndToken, ModelABI.embeddingToken])
        #expect(VisionTokenSplicer.spans(input) == [5..<6, 10..<12])

        let hidden = try MLMultiArray(shape: [1, 512, 2048], dataType: .float16)
        let pointer = hidden.dataPointer.assumingMemoryBound(to: Float16.self)
        pointer.initialize(repeating: -2, count: hidden.count)
        let d = ModelABI.dimension
        let first = try VisionTokens(values: Array(repeating: 3, count: d), count: 1)
        let second = try VisionTokens(values: Array(repeating: 4, count: d) + Array(repeating: 5, count: d), count: 2)

        // Mismatched counts: wrong number of blocks, and blocks in the wrong order.
        #expect(throws: EmbedANEError.self) { try VisionTokenSplicer.splice([first], input: input, into: hidden) }
        #expect(throws: EmbedANEError.self) { try VisionTokenSplicer.splice([second, first], input: input, into: hidden) }
        #expect(throws: EmbedANEError.self) { try VisionTokenSplicer.splice([first, second, first], input: input, into: hidden) }
        #expect((0..<hidden.count).allSatisfy { pointer[$0] == -2 })

        try VisionTokenSplicer.splice([first, second], input: input, into: hidden)
        func row(_ index: Int) -> [Float16] { Array(UnsafeBufferPointer(start: pointer + index * d, count: d)) }
        #expect(row(4).allSatisfy { $0 == -2 } && row(6).allSatisfy { $0 == -2 })
        #expect(row(5).allSatisfy { $0 == 3 })
        #expect(row(9).allSatisfy { $0 == -2 } && row(12).allSatisfy { $0 == -2 })
        #expect(row(10).allSatisfy { $0 == 4 })
        #expect(row(11).allSatisfy { $0 == 5 })
        #expect((0..<5).allSatisfy { row($0).allSatisfy { $0 == -2 } })
        #expect((13..<512).allSatisfy { row($0).allSatisfy { $0 == -2 } })

        // Positions: each span starts after the text before it and advances by max(rows, cols).
        let positions = try Mrope.positions(input: input, grids: [small, large])
        #expect(positions[5] == .init(temporal: 5, height: 5, width: 5))
        #expect(positions[6] == .init(temporal: 6, height: 6, width: 6))
        #expect(positions[9] == .init(temporal: 9, height: 9, width: 9))
        #expect(positions[10] == .init(temporal: 10, height: 10, width: 10))
        #expect(positions[11] == .init(temporal: 10, height: 10, width: 11))
        #expect(positions[12] == .init(temporal: 12, height: 12, width: 12))
    }

    @Test func coreAIPredictorRejectsVideoBeforeTouchingTheModel() async throws {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let paths = try MultimodalModelPaths(tower: missing, chunks: Array(repeating: missing, count: 6),
            tokenizerDirectory: missing, embeddingTable: missing, positionTable: missing)
        let predictor = CoreAIMultimodalPredictor(paths: paths)
        let video = try ImageEmbeddingRequest(videoURL: URL(fileURLWithPath: "/tmp/clip.mp4"), text: "a")
        #expect(video.videoURL?.path == "/tmp/clip.mp4" && video.imageData == nil)
        await #expect(throws: EmbedANEError.invalidRequest("Video embedding requires engine_backend: coreml.", param: "input")) {
            try await predictor.predictImage(video)
        }
        // An image is not rejected by that rule; it reaches the (unloaded) model.
        let image = try ImageEmbeddingRequest(imageData: Data([1]), text: "a")
        #expect(image.imageData == Data([1]) && image.videoURL == nil)
        do { _ = try await predictor.predictImage(image); Issue.record("Unloaded predictor embedded an image") }
        catch { #expect(error as? EmbedANEError != .invalidRequest("Video embedding requires engine_backend: coreml.", param: "input")) }
        // Video requests must name local files.
        #expect(throws: EmbedANEError.self) { try ImageEmbeddingRequest(videoURL: URL(string: "https://example.com/v.mp4")!, text: "") }
    }
}
