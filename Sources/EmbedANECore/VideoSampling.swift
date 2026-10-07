import Foundation

/// How a video becomes frame-pairs that fit the 512-token decoder.
///
/// Upstream WeMM samples at 2 fps (at least 4, at most 768 frames). The decoder
/// here holds 512 tokens, so the frame count is capped at `maximumFrames` and the
/// resolution is chosen so every frame-pair, its timestamp and the prompt fit.
/// Rounding follows the upstream processor: `linspace(...).round()` for frame
/// indices (round half to even) and `smart_resize` with a 32-pixel factor.
public enum VideoSampling {
    public static let samplingFPS = 2.0
    public static let minimumFrames = 4
    public static let maximumFrames = 8
    public static let minimumPixels = 4_096

    public struct Plan: Sendable, Equatable {
        /// Source frame indices, two per pair, in order.
        public let frameIndices: [Int]
        /// Mean presentation time of each pair, in seconds.
        public let timestamps: [Double]
        public var pairs: Int { frameIndices.count / 2 }
    }

    public static func plan(totalFrames: Int, fps: Double) throws -> Plan {
        guard totalFrames >= 2, fps.isFinite, fps > 0 else {
            throw EmbedANEError.invalidRequest("Video must have at least two frames and a valid frame rate.", param: "video")
        }
        let duration = Double(totalFrames) / fps
        var count = min(max(Int(duration * samplingFPS), minimumFrames), maximumFrames, totalFrames)
        count -= count % 2
        let indices = linspace(stop: totalFrames - 1, count: count)
        let timestamps = stride(from: 0, to: indices.count, by: 2).map {
            (Double(indices[$0]) / fps + Double(indices[$0 + 1]) / fps) / 2
        }
        return .init(frameIndices: indices, timestamps: timestamps)
    }

    /// The text placed before each frame-pair, e.g. `<4.1 seconds>`.
    public static func timestampText(_ seconds: Double) -> String { String(format: "<%.1f seconds>", seconds) }

    /// Visual tokens each pair may use, given the tokens everything else needs.
    public static func tokensPerPair(pairs: Int, textTokens: Int, timestampTokens: Int) throws -> Int {
        let wrapper = PromptTemplate.overhead(.video)
        let free = ModelABI.sequenceLength - 1 - wrapper - textTokens - timestampTokens - 2 * pairs
        let perPair = min(VisionABIConstants.wemm.tokenCapacity, free / max(pairs, 1))
        guard perPair >= 1 else {
            throw EmbedANEError.invalidRequest("Text is too long to embed together with a video in 512 tokens.", param: "input")
        }
        return perPair
    }

    /// Upstream `smart_resize` for video (temporal factor 2, spatial factor 32),
    /// with the pixel budget `frames × tokensPerPair × 1024`.
    public static func resizedSize(frames: Int, width: Int, height: Int, tokensPerPair: Int) throws -> VisionImageSize {
        let factor = VisionABIConstants.wemm.resizeFactor
        guard frames >= 2, width > 0, height > 0,
              Double(max(width, height)) / Double(min(width, height)) <= 200 else {
            throw EmbedANEError.invalidRequest("Invalid video dimensions or aspect ratio greater than 200:1.", param: "video")
        }
        let maxPixels = frames * tokensPerPair * factor * factor
        let f = Double(factor), t = Double((frames + 1) / 2 * 2)
        var h = Int((Double(height) / f).rounded(.toNearestOrEven)) * factor
        var w = Int((Double(width) / f).rounded(.toNearestOrEven)) * factor
        if t * Double(h) * Double(w) > Double(maxPixels) {
            let beta = (Double(frames) * Double(height) * Double(width) / Double(maxPixels)).squareRoot()
            h = max(factor, Int(floor(Double(height) / beta / f)) * factor)
            w = max(factor, Int(floor(Double(width) / beta / f)) * factor)
        } else if t * Double(h) * Double(w) < Double(minimumPixels) {
            let beta = (Double(minimumPixels) / (Double(frames) * Double(height) * Double(width))).squareRoot()
            h = Int(ceil(Double(height) * beta / f)) * factor
            w = Int(ceil(Double(width) * beta / f)) * factor
        }
        guard (h / 32) * (w / 32) <= tokensPerPair, h >= factor, w >= factor else {
            throw EmbedANEError.invalidRequest("Video frames cannot be resized to fit the token budget.", param: "video")
        }
        return .init(width: w, height: h)
    }

    /// numpy `linspace(0, stop, count).round()`: `i * step`, last element exact.
    public static func linspace(stop: Int, count: Int) -> [Int] {
        guard count > 1 else { return [0] }
        let step = Double(stop) / Double(count - 1)
        return (0..<count).map { $0 == count - 1 ? stop : Int((Double($0) * step).rounded(.toNearestOrEven)) }
    }
}
