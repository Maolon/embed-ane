import Foundation

/// The verified WeMM Qwen3.5 vision ABI, not generic Qwen2-VL defaults. Only this
/// exact ABI is accepted; no model assets are embedded in the product.
public struct VisionABIConstants: Codable, Sendable, Equatable {
    public let patchSize: Int
    public let mergeSize: Int
    public let temporalPatchSize: Int
    public let channels: Int
    public let patchCapacity: Int
    public let hiddenDimension: Int
    public let positionGridSide: Int
    public let visionHeadDimension: Int
    public let visionRopeTheta: Float
    public let textHeadDimension: Int
    public let partialRotaryFactor: Float
    public let textRopeTheta: Float
    public let mropeSections: [Int]
    public let imageToken: Int
    public let visionStartToken: Int
    public let visionEndToken: Int
    public let videoToken: Int
    public let minimumPixels: Int
    public let pixelMean: Float
    public let pixelStandardDeviation: Float

    public init() {
        patchSize = 16; mergeSize = 2; temporalPatchSize = 2; channels = 3
        patchCapacity = 1536; hiddenDimension = 1024; positionGridSide = 48
        visionHeadDimension = 64; visionRopeTheta = 10_000
        textHeadDimension = 256; partialRotaryFactor = 0.25; textRopeTheta = 10_000_000
        mropeSections = [11, 11, 10]
        imageToken = 248_056; visionStartToken = 248_053; visionEndToken = 248_054; videoToken = 248_057
        minimumPixels = 65_536; pixelMean = 0.5; pixelStandardDeviation = 0.5
    }
    public static let wemm = Self()
    /// Packed (C, T, 16, 16) frame-pair patch fed to the two-frame tower.
    public var patchDimension: Int { channels * temporalPatchSize * patchSize * patchSize }
    /// One frame of a patch, for the CoreAI tower whose kernel is folded over T.
    public var framePatchDimension: Int { channels * patchSize * patchSize }
    public var tokenCapacity: Int { patchCapacity / (mergeSize * mergeSize) }
    public var resizeFactor: Int { patchSize * mergeSize }
    public var maximumPixels: Int { patchCapacity * patchSize * patchSize }
    public var rotaryDimension: Int { Int(Float(textHeadDimension) * partialRotaryFactor) }
}

/// One frame-pair's spatial grid: a still image, or one pair of a video. Patch and
/// merged-token grids must not be confused: the tower rotates patches; the
/// decoder rotates merged tokens.
public struct VisionGrid: Sendable, Equatable, Codable {
    public let height: Int
    public let width: Int
    public var patchCount: Int { height * width }
    public var tokenCount: Int { mergedHeight * mergedWidth }
    public var mergedHeight: Int { height / VisionABIConstants.wemm.mergeSize }
    public var mergedWidth: Int { width / VisionABIConstants.wemm.mergeSize }

    public init(height: Int, width: Int) throws {
        let abi = VisionABIConstants.wemm
        guard height >= abi.mergeSize, width >= abi.mergeSize,
              height <= abi.patchCapacity, width <= abi.patchCapacity,
              height.isMultiple(of: abi.mergeSize), width.isMultiple(of: abi.mergeSize),
              height * width <= abi.patchCapacity else {
            throw EmbedANEError.invalidRequest("Image grid must be even and fit 1536 patches (384 image tokens).", param: "image")
        }
        self.height = height; self.width = width
    }

    enum CodingKeys: String, CodingKey { case height, width }
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(height: container.decode(Int.self, forKey: .height), width: container.decode(Int.self, forKey: .width))
    }

    /// Order expected by the tower: each 2×2 merger group is contiguous.
    var patchCoordinates: [(row: Int, column: Int)] {
        let merge = VisionABIConstants.wemm.mergeSize
        return (0..<mergedHeight).flatMap { blockRow in
            (0..<mergedWidth).flatMap { blockColumn in
                (0..<merge).flatMap { row in
                    (0..<merge).map { column in (blockRow * merge + row, blockColumn * merge + column) }
                }
            }
        }
    }
}
