import EmbedANECore
import Testing

/// Expected values come from the fp32 reference (upstream processor) on a
/// 1709-frame, 30 fps, 360×640 clip: 8 frames, 4 pairs, 448×256 pixels.
@Suite("Video sampling policy") struct VideoSamplingTests {
    @Test func referenceClipSamplesEightFramesUniformly() throws {
        let plan = try VideoSampling.plan(totalFrames: 1709, fps: 30)
        #expect(plan.frameIndices == [0, 244, 488, 732, 976, 1220, 1464, 1708])
        #expect(plan.pairs == 4)
        #expect(plan.timestamps.map(VideoSampling.timestampText) == ["<4.1 seconds>", "<20.3 seconds>", "<36.6 seconds>", "<52.9 seconds>"])
    }
    @Test func shortClipsKeepTheFourFrameMinimumAndEvenCounts() throws {
        #expect(try VideoSampling.plan(totalFrames: 30, fps: 30).frameIndices.count == 4)  // 1 s -> 2 -> min 4
        #expect(try VideoSampling.plan(totalFrames: 75, fps: 30).frameIndices.count == 4)  // 2.5 s -> 5 -> even 4
        #expect(try VideoSampling.plan(totalFrames: 3, fps: 30).frameIndices == [0, 2])     // fewer frames than minimum
        #expect(throws: EmbedANEError.self) { try VideoSampling.plan(totalFrames: 1, fps: 30) }
    }
    @Test func linspaceRoundsHalfToEvenLikeNumpy() {
        // numpy.linspace(0, 9, 4).round() == [0, 3, 6, 9]; (0, 5, 3) -> [0, 2 (2.5), 5]
        #expect(VideoSampling.linspace(stop: 9, count: 4) == [0, 3, 6, 9])
        #expect(VideoSampling.linspace(stop: 5, count: 3) == [0, 2, 5])
    }
    @Test func smartResizeMatchesTheReference() throws {
        let size = try VideoSampling.resizedSize(frames: 8, width: 360, height: 640, tokensPerPair: 117)
        #expect(size == VisionImageSize(width: 256, height: 448))
        #expect((size.width / 32) * (size.height / 32) == 112)
    }
    @Test func budgetLeavesRoomForPromptTimestampsAndText() throws {
        // 511 usable - 4 wrapper - 20 timestamp - 8 vision boundaries = 479 -> 119 per pair
        #expect(try VideoSampling.tokensPerPair(pairs: 4, textTokens: 0, timestampTokens: 20) == 119)
        #expect(try VideoSampling.tokensPerPair(pairs: 1, textTokens: 0, timestampTokens: 5) == 384)
        #expect(throws: EmbedANEError.self) { try VideoSampling.tokensPerPair(pairs: 4, textTokens: 500, timestampTokens: 20) }
    }
}
