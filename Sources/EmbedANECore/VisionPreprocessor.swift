import CoreGraphics
import Foundation
import ImageIO

public struct RGBImage: Sendable {
    public let width: Int
    public let height: Int
    /// Top-to-bottom rows, interleaved RGB bytes, no alpha or row padding.
    public let pixels: [UInt8]
    public init(width: Int, height: Int, pixels: [UInt8]) throws {
        guard width > 0, height > 0, width <= 16_384, height <= 16_384,
              width * height <= 64 * 1_024 * 1_024, pixels.count == width * height * 3 else {
            throw EmbedANEError.invalidRequest("Expected RGB pixels for a nonempty image of at most 64 megapixels (16384 per side).", param: "image")
        }
        self.width = width; self.height = height; self.pixels = pixels
    }
}

public enum VisionResizeMode: String, Codable, Sendable {
    case smart
    case originBucket = "origin-bucket"
}

public struct VisionImageSize: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public init(width: Int, height: Int) { self.width = width; self.height = height }
}

public struct VisionPreparedInput: Sendable {
    public let grid: VisionGrid
    /// [patchCapacity, 1536]: each patch packs (C, T=2, 16, 16), like the
    /// reference processor's pixel values.
    public let patches: [Float16]
    public let absolutePositions: [Float16]
    public let keyMask: [Float16]
    public let rotary: RotaryValues

    /// [patchCapacity, 768]: the first frame of each patch, for the CoreAI tower
    /// whose kernel is folded over T. Exact only for stills (identical frames).
    public var firstFramePatches: [Float16] {
        let abi = VisionABIConstants.wemm
        let plane = abi.patchSize * abi.patchSize
        var result = [Float16](repeating: 0, count: abi.patchCapacity * abi.framePatchDimension)
        for patch in 0..<abi.patchCapacity {
            for channel in 0..<abi.channels {
                let source = patch * abi.patchDimension + channel * abi.temporalPatchSize * plane
                let target = patch * abi.framePatchDimension + channel * plane
                for offset in 0..<plane { result[target + offset] = patches[source + offset] }
            }
        }
        return result
    }
}

/// Pure CPU preparation; no AppKit, network, Python or model inference.
/// Patch/position ordering adapts our wemm_vision.py and ane_embedding_bridge.py.
/// abs_pos is the INTERPOLATED LEARNED [48,48,1024] table, not coordinate IDs.
public struct VisionPreprocessor: Sendable {
    private let positionTable: [Float]
    public let resizeMode: VisionResizeMode

    public init(positionTable: [Float], resizeMode: VisionResizeMode = .smart) throws {
        let abi = VisionABIConstants.wemm
        guard positionTable.count == abi.positionGridSide * abi.positionGridSide * abi.hiddenDimension,
              positionTable.allSatisfy(\.isFinite) else {
            throw EmbedANEError.verification(path: "vision_position_table", reason: "Expected finite fp32 [2304,1024] learned position weights.")
        }
        self.positionTable = positionTable; self.resizeMode = resizeMode
    }
    public init(positionTableURL: URL, resizeMode: VisionResizeMode = .smart) throws {
        let abi = VisionABIConstants.wemm
        let table = try Float32NPY(url: positionTableURL,
            expectedShapes: [[abi.positionGridSide * abi.positionGridSide, abi.hiddenDimension]])
        try self.init(positionTable: table.values, resizeMode: resizeMode)
    }

    public func prepare(imageData: Data) throws -> VisionPreparedInput { try prepare(image: Self.decode(imageData)) }
    public func prepare(image: RGBImage) throws -> VisionPreparedInput {
        let size = try resizedSize(width: image.width, height: image.height)
        return try prepareResized(Self.resize(image, to: size))
    }

    /// Pixels already resized to the exact 32-aligned grid. A still repeats its
    /// frame in both temporal slots, as the reference processor does.
    public func prepareResized(_ image: RGBImage) throws -> VisionPreparedInput {
        try prepareResized(pair: (image, image))
    }

    /// One video frame-pair (or a still twice), both already resized to the same
    /// 32-aligned size.
    public func prepareResized(pair: (RGBImage, RGBImage)) throws -> VisionPreparedInput {
        let abi = VisionABIConstants.wemm
        let image = pair.0
        guard image.width.isMultiple(of: abi.resizeFactor), image.height.isMultiple(of: abi.resizeFactor),
              pair.1.width == image.width, pair.1.height == image.height else {
            throw EmbedANEError.invalidRequest("Resized frames must share one size with sides that are multiples of 32.", param: "image")
        }
        let frames = [pair.0.pixels, pair.1.pixels]
        let plane = abi.patchSize * abi.patchSize
        let grid = try VisionGrid(height: image.height / abi.patchSize, width: image.width / abi.patchSize)
        var patches = [Float16](repeating: 0, count: abi.patchCapacity * abi.patchDimension)
        var positions = [Float16](repeating: 0, count: abi.patchCapacity * abi.hiddenDimension)
        let coordinates = grid.patchCoordinates
        for (patchIndex, coordinate) in coordinates.enumerated() {
            // (C, T, H, W) inside each patch; blockRow, blockCol, row, col outside it.
            for channel in 0..<abi.channels {
                for (slot, pixels) in frames.enumerated() {
                    for y in 0..<abi.patchSize {
                        for x in 0..<abi.patchSize {
                            let source = ((coordinate.row * abi.patchSize + y) * image.width + coordinate.column * abi.patchSize + x) * abi.channels + channel
                            let target = patchIndex * abi.patchDimension + (channel * abi.temporalPatchSize + slot) * plane + y * abi.patchSize + x
                            let scaled = Float(pixels[source]) / 255
                            patches[target] = Float16((scaled - abi.pixelMean) / abi.pixelStandardDeviation)
                        }
                    }
                }
            }
            // Endpoint-aligned bilinear interpolation in fp32, THEN fp16. This
            // reproduces the learned-table math in our VisionTower.abs_pos_embed.
            let h = Float(coordinate.row) * Float(abi.positionGridSide - 1) / Float(grid.height - 1)
            let w = Float(coordinate.column) * Float(abi.positionGridSide - 1) / Float(grid.width - 1)
            let h0 = Int(h), w0 = Int(w)
            let h1 = min(h0 + 1, abi.positionGridSide - 1), w1 = min(w0 + 1, abi.positionGridSide - 1)
            let dh = h - Float(h0), dw = w - Float(w0)
            let offsets = [(h0 * abi.positionGridSide + w0) * abi.hiddenDimension,
                           (h0 * abi.positionGridSide + w1) * abi.hiddenDimension,
                           (h1 * abi.positionGridSide + w0) * abi.hiddenDimension,
                           (h1 * abi.positionGridSide + w1) * abi.hiddenDimension]
            for column in 0..<abi.hiddenDimension {
                let a = positionTable[offsets[0] + column] * (1 - dh) * (1 - dw)
                let b = positionTable[offsets[1] + column] * (1 - dh) * dw
                let c = positionTable[offsets[2] + column] * dh * (1 - dw)
                let d = positionTable[offsets[3] + column] * dh * dw
                positions[patchIndex * abi.hiddenDimension + column] = Float16(((a + b) + c) + d)
            }
        }
        var mask = [Float16](repeating: 0, count: abi.patchCapacity)
        for index in 0..<grid.patchCount { mask[index] = 1 }
        guard positions.allSatisfy(\.isFinite) else {
            throw EmbedANEError.verification(path: "vision_position_table", reason: "Interpolated positions overflow fp16.")
        }
        return .init(grid: grid, patches: patches, absolutePositions: positions, keyMask: mask, rotary: Mrope.vision(grid: grid))
    }

    public func resizedSize(width: Int, height: Int) throws -> VisionImageSize {
        let abi = VisionABIConstants.wemm
        guard width > 0, height > 0, width <= 16_384, height <= 16_384,
              Double(max(width, height)) / Double(min(width, height)) <= 200 else {
            throw EmbedANEError.invalidRequest("Invalid image dimensions or aspect ratio greater than 200:1.", param: "image")
        }
        let factor = Double(abi.resizeFactor)
        let area = Double(width) * Double(height)
        func aligned(_ value: Double) -> Int { max(abi.resizeFactor, Int((value / factor).rounded(.toNearestOrEven)) * abi.resizeFactor) }
        var w: Int, h: Int
        if resizeMode == .originBucket {
            // Opt-in legacy policy: always fill the 1536-patch capacity, even for
            // tiny inputs (upscaling). Upstream keeps the native size, as `.smart`
            // does; on a 672×448 test image the upscale alone costs fp32 cosine 0.986.
            let scale = sqrt(Double(abi.maximumPixels) / area)
            w = aligned(Double(width) * scale); h = aligned(Double(height) * scale)
        } else {
            w = aligned(Double(width)); h = aligned(Double(height))
            if w * h > abi.maximumPixels {
                let scale = sqrt(area / Double(abi.maximumPixels))
                w = max(abi.resizeFactor, Int(floor(Double(width) / scale / factor)) * abi.resizeFactor)
                h = max(abi.resizeFactor, Int(floor(Double(height) / scale / factor)) * abi.resizeFactor)
            } else if w * h < abi.minimumPixels {
                let scale = sqrt(Double(abi.minimumPixels) / area)
                w = max(abi.resizeFactor, Int(ceil(Double(width) * scale / factor)) * abi.resizeFactor)
                h = max(abi.resizeFactor, Int(ceil(Double(height) * scale / factor)) * abi.resizeFactor)
            }
        }
        // Clamp very narrow inputs too: the minimum 32-pixel side must not push
        // the other side beyond CAP after rounding.
        while w * h > abi.maximumPixels {
            // The legacy service passes PIL (width,height) to bucket_resize;
            // ties therefore shrink width first. Preserve that parity detail.
            if w >= h { w -= abi.resizeFactor } else { h -= abi.resizeFactor }
        }
        return .init(width: w, height: h)
    }

    /// ImageIO handles EXIF orientation; an explicit sRGB bitmap fixes byte
    /// order. Animated/multipage input uses only the first still image.
    static func decode(_ data: Data) throws -> RGBImage {
        guard !data.isEmpty, data.count <= 64 * 1_024 * 1_024,
              let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
              let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue,
              width > 0, height > 0, width <= 16_384, height <= 16_384, width * height <= 64 * 1_024 * 1_024 else {
            throw EmbedANEError.invalidRequest("Cannot decode image (64 MiB encoded / 64 megapixel limit).", param: "image")
        }
        let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: max(width, height)]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw EmbedANEError.invalidRequest("Cannot decode image pixels.", param: "image")
        }
        return try rgb(image)
    }

    /// Draws a decoded image into 8-bit RGB and drops alpha. Shared by the image
    /// decoder (`colorMatched`: converted to sRGB) and `VideoFrameReader`
    /// (not matched: drawn in the frame's own RGB space, which reproduces the
    /// plain YUV-to-RGB conversion the reference video decoders use).
    static func rgb(_ image: CGImage, colorMatched: Bool = true) throws -> RGBImage {
        let own = colorMatched ? nil : image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil }
        guard let colorSpace = own ?? CGColorSpace(name: CGColorSpace.sRGB) else {
            throw EmbedANEError.invalidRequest("Cannot decode image pixels.", param: "image")
        }
        var rgba = [UInt8](repeating: 0, count: image.width * image.height * 4)
        try rgba.withUnsafeMutableBytes { buffer in
            guard let context = CGContext(data: buffer.baseAddress, width: image.width, height: image.height,
                bitsPerComponent: 8, bytesPerRow: image.width * 4, space: colorSpace,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.noneSkipLast.rawValue) else {
                throw EmbedANEError.invalidRequest("Cannot allocate RGB image bitmap.", param: "image")
            }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        var rgb = [UInt8](repeating: 0, count: image.width * image.height * 3)
        for index in 0..<(image.width * image.height) {
            for channel in 0..<3 { rgb[index * 3 + channel] = rgba[index * 4 + channel] }
        }
        return try RGBImage(width: image.width, height: image.height, pixels: rgb)
    }

    /// Separable, antialiased bicubic sampling (cubic coefficient -0.5). Work
    /// in 8-bit sRGB like the reference PIL image path, not linear-light floats.
    /// Independent mathematical implementation; no Pillow/transformers code used.
    static func resize(_ image: RGBImage, to size: VisionImageSize) throws -> RGBImage {
        if image.width == size.width, image.height == size.height { return image }
        struct Tap { let index: Int; let weight: Double }
        func taps(source: Int, target: Int) -> [[Tap]] {
            let ratio = Double(source) / Double(target), scale = max(1, ratio)
            return (0..<target).map { output in
                let center = (Double(output) + 0.5) * ratio
                let lower = max(0, Int(floor(center - 2 * scale)))
                let upper = min(source - 1, Int(ceil(center + 2 * scale)))
                var values: [Tap] = []
                var sum: Double = 0
                for input in lower...upper {
                    let x = abs((Double(input) + 0.5 - center) / scale)
                    let weight: Double
                    if x < 1 { weight = 1 + x * x * (1.5 * x - 2.5) }
                    else if x < 2 { weight = 2 + x * (-4 + x * (2.5 - 0.5 * x)) }
                    else { continue }
                    values.append(.init(index: input, weight: weight)); sum += weight
                }
                return values.map { .init(index: $0.index, weight: $0.weight / sum) }
            }
        }
        func byte(_ value: Double) -> UInt8 { UInt8(min(255, max(0, value.rounded()))) }
        let horizontal = taps(source: image.width, target: size.width)
        let vertical = taps(source: image.height, target: size.height)
        var intermediate = [UInt8](repeating: 0, count: size.width * image.height * 3)
        for y in 0..<image.height {
            for x in 0..<size.width {
                for channel in 0..<3 {
                    var sum: Double = 0
                    for tap in horizontal[x] { sum += Double(image.pixels[(y * image.width + tap.index) * 3 + channel]) * tap.weight }
                    intermediate[(y * size.width + x) * 3 + channel] = byte(sum)
                }
            }
        }
        var result = [UInt8](repeating: 0, count: size.width * size.height * 3)
        for y in 0..<size.height {
            for x in 0..<size.width {
                for channel in 0..<3 {
                    var sum: Double = 0
                    for tap in vertical[y] { sum += Double(intermediate[(tap.index * size.width + x) * 3 + channel]) * tap.weight }
                    result[(y * size.width + x) * 3 + channel] = byte(sum)
                }
            }
        }
        return try RGBImage(width: size.width, height: size.height, pixels: result)
    }
}
