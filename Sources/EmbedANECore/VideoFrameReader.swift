import AVFoundation
import CoreGraphics
import Foundation

/// Sampled, upright RGB frames of a local video file, ready for preprocessing.
public struct DecodedVideo: Sendable {
    public let plan: VideoSampling.Plan
    public let frames: [RGBImage]
    public let width: Int
    public let height: Int
}

/// Reads only the frames `VideoSampling` selects. Seeks with zero tolerance to
/// the middle of each frame's display interval and applies the track transform
/// (phone videos stay upright). Frames keep the video's own RGB space instead of
/// being color-matched to sRGB: ffmpeg-based reference decoders convert YUV with
/// the tagged matrix only, and sRGB matching shifted pixels by ~7 levels.
public enum VideoFrameReader {
    public static let maximumBytes = 2 * 1_024 * 1_024 * 1_024

    public static func read(_ url: URL) async throws -> DecodedVideo {
        guard url.isFileURL else {
            throw EmbedANEError.invalidRequest("Videos must be local file URLs.", param: "video")
        }
        let values = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values?.isRegularFile == true, let size = values?.fileSize, size > 0, size <= maximumBytes else {
            throw EmbedANEError.invalidRequest("Video must be a regular file of at most 2 GiB.", param: "video")
        }
        let asset = AVURLAsset(url: url)
        let tracks: [AVAssetTrack]
        let duration: CMTime
        do {
            tracks = try await asset.loadTracks(withMediaType: .video)
            duration = try await asset.load(.duration)
        } catch {
            throw EmbedANEError.invalidRequest("Cannot read the video file.", param: "video")
        }
        guard let track = tracks.first else {
            throw EmbedANEError.invalidRequest("The file has no video track.", param: "video")
        }
        let (rate, natural, transform) = try await track.load(.nominalFrameRate, .naturalSize, .preferredTransform)
        let fps = Double(rate)
        let seconds = duration.seconds
        guard fps > 0, seconds.isFinite, seconds > 0 else {
            throw EmbedANEError.invalidRequest("The video has no usable frame rate or duration.", param: "video")
        }
        let upright = natural.applying(transform)
        let width = Int(abs(upright.width).rounded()), height = Int(abs(upright.height).rounded())
        let plan = try VideoSampling.plan(totalFrames: Int((seconds * fps).rounded()), fps: fps)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var frames: [RGBImage] = []
        frames.reserveCapacity(plan.frameIndices.count)
        for index in plan.frameIndices {
            let time = CMTime(seconds: (Double(index) + 0.5) / fps, preferredTimescale: 600_000)
            let image: CGImage
            do { image = try await generator.image(at: time).image }
            catch { throw EmbedANEError.invalidRequest("Cannot decode video frame \(index).", param: "video") }
            frames.append(try VisionPreprocessor.rgb(image, colorMatched: false))
        }
        return .init(plan: plan, frames: frames, width: width, height: height)
    }
}
