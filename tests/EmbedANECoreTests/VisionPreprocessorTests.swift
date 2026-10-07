import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import EmbedANECore

struct VisionFixtureTokenizer: ContentTokenizer {
    let ids: [Int]
    func encodeContent(_ text: String) throws -> [Int] { ids }
}

enum VisionFixtures {
    static func table() -> [Float] {
        let abi = VisionABIConstants.wemm
        return (0..<(abi.positionGridSide * abi.positionGridSide * abi.hiddenDimension)).map { index in
            let position = index / abi.hiddenDimension
            return Float(position / abi.positionGridSide) * 2 + Float(position % abi.positionGridSide) / 2 + Float(index % 8) / 8
        }
    }
    static func npy(_ values: [Float], shape: [Int], dtype: String = "<f4", fortran: Bool = false) -> Data {
        let dimensions = shape.map(String.init).joined(separator: ",") + ","
        var header = "{'descr': '\(dtype)', 'fortran_order': \(fortran ? "True" : "False"), 'shape': (\(dimensions)), }"
        header += String(repeating: " ", count: (16 - ((10 + header.utf8.count + 1) % 16)) % 16) + "\n"
        var data = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0, UInt8(header.utf8.count & 255), UInt8(header.utf8.count >> 8)])
        data.append(contentsOf: header.utf8)
        for value in values {
            var bits = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bits) { data.append(contentsOf: $0) }
        }
        return data
    }
}

@Suite(.serialized)
struct VisionPreprocessorTests {
    @Test func verifiedConstantsAndCodableRoundTrip() throws {
        let abi = VisionABIConstants.wemm
        #expect(abi.patchDimension == 1536); #expect(abi.framePatchDimension == 768); #expect(abi.maximumPixels == 393_216)
        #expect(abi.rotaryDimension == 64); #expect(abi.tokenCapacity == 384)
        #expect(try JSONDecoder().decode(VisionABIConstants.self, from: JSONEncoder().encode(abi)) == abi)
        #expect(throws: EmbedANEError.self) { try JSONDecoder().decode(VisionGrid.self, from: Data("{\"height\":99999,\"width\":99999}".utf8)) }
    }
    @Test func framePairKeepsEachFrameInItsOwnTemporalSlot() throws {
        let preprocessor = try VisionPreprocessor(positionTable: VisionFixtures.table())
        let dark = try RGBImage(width: 32, height: 32, pixels: Array(repeating: 0, count: 32 * 32 * 3))
        let light = try RGBImage(width: 32, height: 32, pixels: Array(repeating: 255, count: 32 * 32 * 3))
        let result = try preprocessor.prepareResized(pair: (dark, light))
        for patch in 0..<4 {
            for channel in 0..<3 {
                #expect(result.patches[patch * 1536 + (channel * 2) * 256 + 5] == -1)
                #expect(result.patches[patch * 1536 + (channel * 2 + 1) * 256 + 5] == 1)
            }
        }
        #expect(throws: EmbedANEError.self) {
            try preprocessor.prepareResized(pair: (dark, try RGBImage(width: 64, height: 32, pixels: Array(repeating: 0, count: 64 * 32 * 3))))
        }
    }
    @Test func patchCHWMergeOrderingNormalizationAndTailPadding() throws {
        let preprocessor = try VisionPreprocessor(positionTable: VisionFixtures.table())
        var pixels: [UInt8] = []
        for y in 0..<32 { for x in 0..<64 { pixels += [UInt8(y), UInt8(x), 128] } }
        let result = try preprocessor.prepareResized(RGBImage(width: 64, height: 32, pixels: pixels))
        #expect(result.grid == (try VisionGrid(height: 2, width: 4)))
        let coordinates = [(0,0),(0,1),(1,0),(1,1),(0,2),(0,3),(1,2),(1,3)]
        // Packed (C, T, 16, 16): channel c, frame slot s at offset (c * 2 + s) * 256.
        // A still fills both slots; the CoreAI view keeps only slot 0 as (C, 16, 16).
        let folded = result.firstFramePatches
        for (patch, coordinate) in coordinates.enumerated() {
            for local in [0, 17, 255] {
                let r = UInt8(coordinate.0 * 16 + local / 16), g = UInt8(coordinate.1 * 16 + local % 16)
                let expected = [r, g, 128].map { Float16((Float($0) / 255 - 0.5) / 0.5) }
                for channel in 0..<3 {
                    for slot in 0..<2 {
                        #expect(result.patches[patch * 1536 + (channel * 2 + slot) * 256 + local] == expected[channel])
                    }
                    #expect(folded[patch * 768 + channel * 256 + local] == expected[channel])
                }
            }
        }
        #expect(result.patches.dropFirst(8 * 1536).allSatisfy { $0 == 0 })
        #expect(folded.dropFirst(8 * 768).allSatisfy { $0 == 0 })
        #expect(result.absolutePositions.dropFirst(8 * 1024).allSatisfy { $0 == 0 })
        #expect(result.keyMask.prefix(8).allSatisfy { $0 == 1 })
        #expect(result.keyMask.dropFirst(8).allSatisfy { $0 == 0 })
        #expect(result.rotary.cosine.dropFirst(8 * 64).allSatisfy { $0 == 1 })
        #expect(result.rotary.sine.dropFirst(8 * 64).allSatisfy { $0 == 0 })
    }
    @Test func learnedAbsolutePositionsAreInterpolatedOnPatchGrid() throws {
        let preprocessor = try VisionPreprocessor(positionTable: VisionFixtures.table())
        let result = try preprocessor.prepareResized(RGBImage(width: 96, height: 64, pixels: [UInt8](repeating: 255, count: 96 * 64 * 3)))
        for (index, coordinate) in result.grid.patchCoordinates.enumerated() {
            for channel in [0, 1, 7, 1023] {
                let h = Float(coordinate.row) * 47 / 3
                let w = Float(coordinate.column) * 47 / 5
                let expected = Float16(h * 2 + w / 2 + Float(channel % 8) / 8)
                #expect(abs(Float(result.absolutePositions[index * 1024 + channel]) - Float(expected)) <= 0.0625)
            }
        }
        #expect(result.patches.prefix(24 * 768).allSatisfy { $0 == 1 })
    }
    @Test func smartResizeDoesNotFillBucketUnnecessarily() throws {
        let table = VisionFixtures.table()
        let smart = try VisionPreprocessor(positionTable: table)
        let legacy = try VisionPreprocessor(positionTable: table, resizeMode: .originBucket)
        #expect(try smart.resizedSize(width: 640, height: 480) == .init(width: 640, height: 480))
        #expect(try smart.resizedSize(width: 32, height: 32) == .init(width: 256, height: 256))
        #expect(try smart.resizedSize(width: 1024, height: 1024) == .init(width: 608, height: 608))
        // Legacy MMService tie-breaking shrinks WIDTH first (PIL size order).
        #expect(try legacy.resizedSize(width: 32, height: 32) == .init(width: 608, height: 640))
        for (w, h) in [(16_384, 128), (128, 16_384), (6000, 4000), (200, 1), (17, 17)] {
            for preprocessor in [smart, legacy] {
                let size = try preprocessor.resizedSize(width: w, height: h)
                #expect(size.width.isMultiple(of: 32) && size.height.isMultiple(of: 32))
                #expect(size.width * size.height <= 393_216)
                #expect(size.width > 0 && size.height > 0)
            }
        }
        #expect(throws: EmbedANEError.self) { try smart.resizedSize(width: 201, height: 1) }
        #expect(throws: EmbedANEError.self) { try smart.resizedSize(width: 0, height: 1) }
    }
    @Test func bicubicMatchesIndependentCPUNumericFixture() throws {
        var pixels: [UInt8] = []
        for y in 0..<5 { for x in 0..<7 { pixels += [UInt8(x * 25 + y * 3), UInt8(x * 31 + y * 7), UInt8(x * 15 + y * 6)] } }
        let image = try RGBImage(width: 7, height: 5, pixels: pixels)
        // Numerical output of Pillow BICUBIC on the generated RGB ramp above;
        // fixture values only, no dependency implementation copied.
        let expected: [UInt8] = [11,14,8,53,67,33,99,123,61,141,176,86,16,26,18,58,79,43,104,135,71,146,188,96,21,38,28,63,91,53,109,147,81,151,200,106]
        let resized = try VisionPreprocessor.resize(image, to: .init(width: 4, height: 3))
        #expect(resized.pixels == expected)
        let enlarged = try VisionPreprocessor.resize(image, to: .init(width: 10, height: 8))
        #expect(Array(enlarged.pixels.prefix(12)) == [0,0,0,12,14,7,31,38,19,49,59,29])
        #expect(Array(enlarged.pixels.suffix(12)) == [113,155,85,131,176,95,150,200,107,164,217,115])
    }
    @Test func imageIODecodePreservesRGBAndRowOrder() throws {
        let pixels: [UInt8] = [255,0,0,255, 0,255,0,255, 0,0,255,255, 255,255,255,255]
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        let color = try #require(CGColorSpace(name: CGColorSpace.sRGB))
        let image = try #require(CGImage(width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8,
            space: color, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let bytes = NSMutableData()
        let destination = try #require(CGImageDestinationCreateWithData(bytes, "public.png" as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
        let decoded = try VisionPreprocessor.decode(bytes as Data)
        #expect(decoded.width == 2 && decoded.height == 2)
        #expect(decoded.pixels == [255,0,0,0,255,0,0,0,255,255,255,255])
    }
    @Test func invalidInputsFailWithoutModelLoading() throws {
        #expect(throws: EmbedANEError.self) { try VisionPreprocessor(positionTable: [0]) }
        var badTable = VisionFixtures.table(); badTable[0] = .nan
        #expect(throws: EmbedANEError.self) { try VisionPreprocessor(positionTable: badTable) }
        #expect(throws: EmbedANEError.self) { try VisionPreprocessor.decode(Data("not an image".utf8)) }
        #expect(throws: EmbedANEError.self) { try RGBImage(width: Int.max, height: Int.max, pixels: []) }
        #expect(throws: EmbedANEError.self) { try VisionGrid(height: 3, width: 2) }
        #expect(throws: EmbedANEError.self) { try VisionGrid(height: 40, width: 40) }
        let prep = try VisionPreprocessor(positionTable: VisionFixtures.table())
        #expect(throws: EmbedANEError.self) { try prep.prepareResized(RGBImage(width: 33, height: 32, pixels: .init(repeating: 0, count: 33 * 32 * 3))) }
    }
}

struct VisionNPYTests {
    @Test func readsStrictFloat32AndPositionTableShape() throws {
        let bytes = VisionFixtures.npy([1, 2, -3.25, 4], shape: [2, 2])
        let array = try Float32NPY(data: bytes, expectedShapes: [[2, 2]])
        #expect(array.values == [1, 2, -3.25, 4]); #expect(array.shape == [2, 2])
        let vector = try Float32NPY(data: VisionFixtures.npy([1, 2], shape: [2]), expectedShapes: [[2]])
        #expect(vector.values == [1, 2])
        #expect(throws: EmbedANEError.self) { try Float32NPY(data: bytes, expectedShapes: [[2304, 1024]]) }
    }
    @Test(arguments: ["dtype", "fortran", "v2", "truncated", "trailing", "nonfinite"])
    func rejectsInvalidNPY(_ mutation: String) throws {
        var data = VisionFixtures.npy([mutation == "nonfinite" ? .infinity : 1], shape: [1],
            dtype: mutation == "dtype" ? "<f2" : "<f4", fortran: mutation == "fortran")
        if mutation == "v2" { data[6] = 2 }
        if mutation == "truncated" { data.removeLast() }
        if mutation == "trailing" { data.append(0) }
        #expect(throws: EmbedANEError.self) { try Float32NPY(data: data, expectedShapes: [[1]]) }
    }
}
