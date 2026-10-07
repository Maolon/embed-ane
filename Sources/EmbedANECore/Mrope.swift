import Foundation

public struct RotaryValues: Sendable {
    public let cosine: [Float16]
    public let sine: [Float16]
    init(cosine: [Float16], sine: [Float16]) { self.cosine = cosine; self.sine = sine }
}

public struct MropePosition: Sendable, Equatable {
    public let temporal: Int
    public let height: Int
    public let width: Int
}

/// Position semantics adapted from our prior ane_embedding_bridge.py
/// (_mrope_cos_sin) and wemm_vision.py (rope_cos_sin). No dependency code copied.
/// Qwen3.5 uses 256 * 0.25 = 64 rotary dimensions, with 32 interleaved
/// frequencies duplicated (NOT 128 dimensions or concatenated T/H/W sections).
public enum Mrope {
    public static func positions(input: TokenizedInput, imageGrid: VisionGrid? = nil) throws -> [MropePosition] {
        try positions(input: input, grids: imageGrid.map { [$0] } ?? [])
    }

    /// Qwen3.5 multimodal positions. Text advances one scalar position; each
    /// visual span (a still, or one video frame-pair, which upstream also places
    /// with temporal index 0) gets (t, h, w) = (start, start+row, start+col) and
    /// the next position is start + max(rows, cols).
    public static func positions(input: TokenizedInput, grids: [VisionGrid]) throws -> [MropePosition] {
        let runs = VisionTokenSplicer.spans(input)
        guard runs.count == grids.count, zip(runs, grids).allSatisfy({ $0.count == $1.tokenCount }) else {
            throw EmbedANEError.invalidRequest("Expected one contiguous image or video token span per visual grid.", param: "input")
        }
        var result: [MropePosition] = []
        result.reserveCapacity(ModelABI.sequenceLength)
        var index = 0, next = 0, span = 0
        while index < input.nTokens {
            if span < runs.count, runs[span].lowerBound == index {
                let grid = grids[span], start = next
                for row in 0..<grid.mergedHeight {
                    for column in 0..<grid.mergedWidth {
                        result.append(.init(temporal: start, height: start + row, width: start + column))
                    }
                }
                index += grid.tokenCount; next = start + max(grid.mergedHeight, grid.mergedWidth); span += 1
            } else {
                result.append(scalar(next)); next += 1; index += 1
            }
        }
        guard let last = result.last else { throw EmbedANEError.emptyInput(index: nil) }
        for offset in 1..<(ModelABI.sequenceLength - input.nTokens + 1) {
            result.append(.init(temporal: last.temporal + offset, height: last.height + offset, width: last.width + offset))
        }
        return result
    }

    public static func language(input: TokenizedInput, imageGrid: VisionGrid? = nil) throws -> RotaryValues {
        try language(input: input, grids: imageGrid.map { [$0] } ?? [])
    }

    public static func language(input: TokenizedInput, grids: [VisionGrid]) throws -> RotaryValues {
        let abi = VisionABIConstants.wemm
        let positions = try positions(input: input, grids: grids)
        let half = abi.rotaryDimension / 2
        let frequencies = (0..<half).map { 1 / powf(abi.textRopeTheta, Float(2 * $0) / Float(abi.rotaryDimension)) }
        var cosine = [Float16](repeating: 0, count: positions.count * abi.rotaryDimension)
        var sine = cosine
        for (row, position) in positions.enumerated() {
            for index in 0..<half {
                let coordinate: Int
                if index % 3 == 1, index < abi.mropeSections[1] * 3 { coordinate = position.height }
                else if index % 3 == 2, index < abi.mropeSections[2] * 3 { coordinate = position.width }
                else { coordinate = position.temporal }
                let angle = Float(coordinate) * frequencies[index]
                let c = Float16(cosf(angle)), s = Float16(sinf(angle))
                let offset = row * abi.rotaryDimension + index
                cosine[offset] = c; cosine[offset + half] = c
                sine[offset] = s; sine[offset + half] = s
            }
        }
        return .init(cosine: cosine, sine: sine)
    }

    /// Tower positions use unmerged PATCH coordinates in merger-group order.
    /// Padded rows are the identity rotation (cos=1, sin=0), not zero cosine.
    public static func vision(grid: VisionGrid) -> RotaryValues {
        let abi = VisionABIConstants.wemm
        let perAxis = abi.visionHeadDimension / 4
        let half = perAxis * 2
        let frequencies = (0..<perAxis).map { 1 / powf(abi.visionRopeTheta, Float($0) / Float(perAxis)) }
        var cosine = [Float16](repeating: 1, count: abi.patchCapacity * abi.visionHeadDimension)
        var sine = [Float16](repeating: 0, count: cosine.count)
        for (row, coordinates) in grid.patchCoordinates.enumerated() {
            for index in 0..<half {
                let position = index < perAxis ? coordinates.row : coordinates.column
                let angle = Float(position) * frequencies[index % perAxis]
                let offset = row * abi.visionHeadDimension + index
                let c = Float16(cosf(angle)), s = Float16(sinf(angle))
                cosine[offset] = c; cosine[offset + half] = c
                sine[offset] = s; sine[offset + half] = s
            }
        }
        return .init(cosine: cosine, sine: sine)
    }

    private static func scalar(_ index: Int) -> MropePosition { .init(temporal: index, height: index, width: index) }
}
