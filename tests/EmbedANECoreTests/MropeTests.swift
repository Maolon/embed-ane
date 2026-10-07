import Foundation
import Testing
@testable import EmbedANECore

struct MropeTests {
    @Test func textPositionsAndTailContinueTo512() throws {
        let input = try TokenizedInput(contentIDs: [7, 8])
        let positions = try Mrope.positions(input: input)
        #expect(positions.count == 512)
        for index in [0, 1, 2, 3, 511] {
            #expect(positions[index] == .init(temporal: index, height: index, width: index))
        }
        let rotary = try Mrope.language(input: input)
        #expect(rotary.cosine.count == 512 * 64 && rotary.sine.count == 512 * 64)
        #expect(rotary.cosine.prefix(64).allSatisfy { $0 == 1 })
        #expect(rotary.sine.prefix(64).allSatisfy { $0 == 0 })
        for row in [1, 2, 511] {
            #expect(rotary.cosine[row * 64] == Float16(cosf(Float(row))))
            #expect(Array(rotary.sine[(row * 64)..<(row * 64 + 32)]) == Array(rotary.sine[(row * 64 + 32)..<(row * 64 + 64)]))
        }
    }
    @Test func imageCoordinatesAndSuffixUseMaximumGridPosition() throws {
        let grid = try VisionGrid(height: 4, width: 6)
        let input = try MultimodalTokenization.prepare(text: "fixture", grid: grid, tokenizer: VisionFixtureTokenizer(ids: [7]))
        let positions = try Mrope.positions(input: input, imageGrid: grid)
        // <|im_start|> user <vision_start> image×6 <vision_end> 7 <|im_end|> <embedding>
        #expect(input.nTokens == 13)
        #expect(positions[0] == .init(temporal: 0, height: 0, width: 0)) // im_start
        #expect(positions[1] == .init(temporal: 1, height: 1, width: 1)) // user
        #expect(positions[2] == .init(temporal: 2, height: 2, width: 2)) // vision_start
        #expect(positions[3] == .init(temporal: 3, height: 3, width: 3)) // first image token
        #expect(positions[5] == .init(temporal: 3, height: 3, width: 5)) // end of first merged row
        #expect(positions[8] == .init(temporal: 3, height: 4, width: 5)) // last image token
        #expect(positions[9] == .init(temporal: 6, height: 6, width: 6)) // vision_end
        #expect(positions[11] == .init(temporal: 8, height: 8, width: 8)) // im_end
        #expect(positions[12] == .init(temporal: 9, height: 9, width: 9)) // embedding
        #expect(positions[13] == .init(temporal: 10, height: 10, width: 10)) // padding
        #expect(positions[511] == .init(temporal: 508, height: 508, width: 508))
    }
    @Test func languageRotaryMatchesIndependentQwen35CPUFixture() throws {
        let grid = try VisionGrid(height: 4, width: 6)
        // The independent fixture is at (T,H,W)=(1,2,3): row 6, the last image token
        // of an image span starting at index 1. The chat-template prefix moves real
        // prompts' span to index 3, so build the bare span directly: this test pins
        // the rotary math, not the prompt layout (see imageCoordinatesAndSuffix...).
        let abi = VisionABIConstants.wemm
        let input = try TokenizedInput(contentIDs: [abi.visionStartToken]
            + Array(repeating: abi.imageToken, count: grid.tokenCount) + [abi.visionEndToken])
        let rotary = try Mrope.language(input: input, imageGrid: grid)
        // Numeric output only from Qwen3_5TextRotaryEmbedding on CPU, position
        // (T,H,W)=(1,2,3), theta=1e7, partial rotary 256*0.25, sections[11,11,10].
        let cosine: [Float] = [0.54052734375,0.354248046875,0.45751953125,0.9755859375,0.96484375,0.970703125,0.9990234375,0.998046875,0.99853515625] + Array(repeating: 1, count: 23)
        let sine: [Float] = [0.84130859375,0.93505859375,0.88916015625,0.2188720703125,0.263671875,0.2393798828125,0.048675537109375,0.058807373046875,0.053314208984375,0.0107421875,0.0129852294921875,0.01177215576171875,0.0023708343505859375,0.0028667449951171875,0.002597808837890625,0.0005230903625488281,0.0006322860717773438,0.0005731582641601562,0.00011545419692993164,0.00013959407806396484,0.0001264810562133789,0.000025510787963867188,0.00003081560134887695,0.000027894973754882812,0.000005602836608886719,0.000006794929504394531,0.000006139278411865234,0.0000012516975402832031,0.0000014901161193847656,0.0000013709068298339844,0.0000002980232238769531,0.00000035762786865234375]
        for index in 0..<64 {
            #expect(abs(Float(rotary.cosine[6 * 64 + index]) - cosine[index % 32]) <= 0.0005)
            #expect(abs(Float(rotary.sine[6 * 64 + index]) - sine[index % 32]) <= max(0.00000006, sine[index % 32] * 0.001))
        }
    }
    @Test func towerRotaryUsesPatchOrderAndIndependentAxes() throws {
        let grid = try VisionGrid(height: 2, width: 4)
        let rotary = Mrope.vision(grid: grid)
        // In merger-group order the second patch is (row=0,col=1), not (1,0).
        #expect(rotary.cosine[64] == 1)
        #expect(rotary.sine[64] == 0)
        #expect(rotary.cosine[64 + 16] == Float16(cosf(1)))
        #expect(rotary.sine[64 + 16] == Float16(sinf(1)))
        #expect(rotary.sine[2 * 64] == Float16(sinf(1)))
        #expect(rotary.sine[2 * 64 + 16] == 0)
        for index in 0..<(8 * 64) { #expect(rotary.cosine[index].isFinite && rotary.sine[index].isFinite) }
    }
    @Test func malformedImageSpansAreRejected() throws {
        let image = VisionABIConstants.wemm.imageToken
        let grid = try VisionGrid(height: 2, width: 4)
        let noncontiguous = try TokenizedInput(contentIDs: [1, image, 2, image])
        let wrongCount = try TokenizedInput(contentIDs: [1, image])
        #expect(throws: EmbedANEError.self) { try Mrope.positions(input: noncontiguous, imageGrid: grid) }
        #expect(throws: EmbedANEError.self) { try Mrope.positions(input: wrongCount, imageGrid: grid) }
        #expect(throws: EmbedANEError.self) { try Mrope.positions(input: wrongCount) }
        #expect(throws: EmbedANEError.self) { try Mrope.positions(input: TokenizedInput(contentIDs: [7]), imageGrid: grid) }
    }
    @Test func exactMultimodalBudgetAndCanonicalSpecials() throws {
        let grid = try VisionGrid(height: 32, width: 48) // 384 tokens
        let ids = Array(repeating: 7, count: 122) // 384+122+im_start/user/start/end/im_end/embedding=512
        let input = try MultimodalTokenization.prepare(text: "fixture", grid: grid, tokenizer: VisionFixtureTokenizer(ids: ids))
        #expect(input.nTokens == 512 && input.mask.allSatisfy { $0 == 1 })
        #expect(Array(input.ids.prefix(3)) == [248045, 846, 248053]) // no role newline before a leading image
        #expect(input.ids[387] == 248054); #expect(input.ids[510] == 248046); #expect(input.ids[511] == 248077)
        #expect(input.ids.filter { $0 == 248056 }.count == 384)
        #expect(try Mrope.positions(input: input, imageGrid: grid).count == 512)
        #expect(throws: EmbedANEError.self) {
            try MultimodalTokenization.prepare(text: "long", grid: grid, tokenizer: VisionFixtureTokenizer(ids: ids + [7]))
        }
        let text = try MultimodalTokenization.prepare(text: "text", grid: nil, tokenizer: VisionFixtureTokenizer(ids: [7, 8]))
        #expect(text == (try TokenizedInput(contentIDs: [248045, 846, 198, 7, 8, 248046])))
    }
    @Test(arguments: [248053, 248054, 248056, 248057, 248045, 248046, 248077])
    func rejectsUserPlaceholders(_ token: Int) throws {
        #expect(throws: EmbedANEError.self) {
            try MultimodalTokenization.prepare(text: "", grid: VisionGrid(height: 2, width: 2), tokenizer: VisionFixtureTokenizer(ids: [token]))
        }
    }
}
