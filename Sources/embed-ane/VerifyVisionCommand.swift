import EmbedANECore
import Foundation
import UniformTypeIdentifiers

struct VerifyVisionOptions: Sendable {
    /// Image or video file; video is detected from the file type.
    let image: String
    let text: String
    let positionTable: String
    let tower: String?
    /// Defaults to the installed model folder (model root / model id).
    let chunksDirectory: String?
    let embeddingTable: String?
    let resize: VisionResizeMode

    static func parse(_ arguments: [String]) throws -> CLIInvocation {
        var values: [String: String] = [:], positionals: [String] = []
        var seen = Set<String>(), index = 0, positionalOnly = false
        let options = Set(["--position-table", "--tower", "--chunks-dir", "--assets", "--embedding-table",
                           "--resize", "--reference", "--model-root", "--model-id"])
        while index < arguments.count {
            let argument = arguments[index]; index += 1
            if argument == "--", !positionalOnly { positionalOnly = true; continue }
            if positionalOnly || !argument.hasPrefix("-") { positionals.append(argument); continue }
            guard seen.insert(argument).inserted else { throw CLIUsageError("Duplicate option \(argument).") }
            if argument == "--extrope" { continue } // This command never loads plain chunks.
            guard options.contains(argument), index < arguments.count,
                  !arguments[index].hasPrefix("--"), !arguments[index].isEmpty else {
                throw CLIUsageError("Unknown option or missing value: \(argument).")
            }
            values[argument] = arguments[index]; index += 1
        }
        guard positionals.count == 2, !positionals[0].isEmpty, let positions = values["--position-table"] else {
            throw CLIUsageError("verify-vision requires <image-or-video> <text> and --position-table <fp32.npy> (learned [2304,1024] weights).")
        }
        guard let resize = VisionResizeMode(rawValue: values["--resize"] ?? "smart") else {
            throw CLIUsageError("--resize must be smart or origin-bucket.")
        }
        var overrides = ConfigurationOverrides()
        overrides.modelRoot = values["--model-root"]; overrides.modelID = values["--model-id"]
        return .init(command: .verifyVision, operand: positionals[0], overrides: overrides, offline: false, replace: false,
            assets: values["--assets"], reference: values["--reference"], outputDirectory: nil,
            vision: .init(image: positionals[0], text: positionals[1], positionTable: positions,
                tower: values["--tower"], chunksDirectory: values["--chunks-dir"],
                embeddingTable: values["--embedding-table"], resize: resize))
    }
}

struct VisionReferenceComparison: Encodable, Sendable {
    let cosine: Double
    let threshold: Double = 0.99
    let gatePassed: Bool
    init(embedding: [Float], reference: [Float]) throws {
        guard embedding.count == ModelABI.dimension, reference.count == ModelABI.dimension,
              embedding.allSatisfy(\.isFinite), reference.allSatisfy(\.isFinite) else {
            throw EmbedANEError.verification(path: "reference", reason: "Expected finite 2048-dimensional vectors.")
        }
        var dot: Double = 0, left: Double = 0, right: Double = 0
        for (a, b) in zip(embedding, reference) { dot += Double(a) * Double(b); left += Double(a) * Double(a); right += Double(b) * Double(b) }
        guard left > 0, right > 0 else { throw EmbedANEError.verification(path: "reference", reason: "Zero-norm vector.") }
        cosine = min(1, max(-1, dot / sqrt(left * right)))
        gatePassed = cosine >= threshold
    }
}

private struct VisionSmokeReport: Encodable {
    let dimension: Int
    let norm: Double
    let vectorHead: [Float]
    let promptTokens: Int
    let imageTokens: Int
    let grid: VisionGrid?
    let media: String
    let resize: VisionResizeMode
    let load: MultimodalLoadReport
    let preprocessNS: UInt64
    let tokenizeNS: UInt64
    let towerNS: UInt64
    let tableLookupNS: UInt64
    let perChunkNS: [UInt64]
    let totalNS: UInt64
    let comparison: VisionReferenceComparison?
    let note = "Injected local artifacts; CPU + Neural Engine configured, residency not audited. Chunk 5 normalization retained unchanged."
}

enum VerifyVisionCommand {
    static func run(_ invocation: CLIInvocation, context: CLIContext, io: CLIFileWork) async throws -> CLIResult {
        guard let options = invocation.vision else { throw CLIUsageError("Missing vision options.") }
        let bundle = URL(fileURLWithPath: context.configuration.modelRoot).appendingPathComponent(context.configuration.modelID)
        let chunks = options.chunksDirectory.map(context.path) ?? bundle.appendingPathComponent("chunks", isDirectory: true)
        let paths = try MultimodalModelPaths(
            tower: options.tower.map(context.path) ?? bundle.appendingPathComponent("vision_tower.mlmodelc"),
            chunks: (0..<6).map { chunks.appendingPathComponent("chunk\($0).mlmodelc") },
            tokenizerDirectory: invocation.assets.map(context.path) ?? bundle,
            embeddingTable: options.embeddingTable.map(context.path) ?? bundle.appendingPathComponent("embed_table.fp16.npy"),
            positionTable: context.path(options.positionTable))
        let mediaURL = context.path(options.image)
        let isVideo = UTType(filenameExtension: mediaURL.pathExtension)?.conforms(to: .movie) == true
        // Fail malformed references before incurring any model load.
        let (image, reference) = try await io.run {
            let url = mediaURL
            let image = isVideo ? Data() : try SecureDirectory(url.deletingLastPathComponent()).read(url.lastPathComponent, limit: 64 * 1_024 * 1_024)
            let reference = try invocation.reference.map { try Float32NPY(url: context.path($0), expectedShapes: [[2048], [1, 2048]]).values }
            if let reference { _ = try VisionReferenceComparison(embedding: reference, reference: reference) }
            return (image, reference)
        }
        let runtime = MultimodalPredictor(paths: paths, resizeMode: options.resize)
        do {
            let load = try await runtime.loadVision()
            let prediction = isVideo
                ? try await runtime.embed(videoURL: mediaURL, text: options.text)
                : try await runtime.embed(imageData: image, text: options.text)
            _ = try await runtime.unload()
            let comparison = try reference.map { try VisionReferenceComparison(embedding: prediction.embedding, reference: $0) }
            let report = VisionSmokeReport(dimension: prediction.embedding.count,
                norm: sqrt(prediction.embedding.reduce(0.0) { $0 + Double($1) * Double($1) }),
                vectorHead: Array(prediction.embedding.prefix(4)), promptTokens: prediction.promptTokens,
                imageTokens: prediction.imageTokens, grid: prediction.grid, media: isVideo ? "video" : "image",
                resize: options.resize, load: load,
                preprocessNS: prediction.preprocessNS, tokenizeNS: prediction.tokenizeNS, towerNS: prediction.towerNS,
                tableLookupNS: prediction.tableLookupNS, perChunkNS: prediction.perChunkNS, totalNS: prediction.totalNS, comparison: comparison)
            return .init(exitCode: comparison?.gatePassed == false ? 2 : 0, stdout: try CLIJSON.encode(report))
        } catch {
            _ = try? await runtime.unload()
            throw error
        }
    }
}
