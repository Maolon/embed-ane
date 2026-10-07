import EmbedANECore
import Foundation
import Testing
@testable import embed_ane

struct VerifyVisionCommandTests {
    @Test func parsesImageTextAndExplicitInjectedPaths() throws {
        let invocation = try CLIInvocation.parse(["verify-vision", "picture.png", "a cat", "--position-table", "positions.npy",
            "--chunks-dir", "/models", "--tower", "/tower.mlmodelc", "--assets", "/tokenizer", "--embedding-table", "/table.npy",
            "--resize", "origin-bucket", "--reference", "ref.npy", "--extrope"])
        #expect(invocation.command == .verifyVision && invocation.reference == "ref.npy")
        let options = try #require(invocation.vision)
        #expect(options.image == "picture.png" && options.text == "a cat")
        #expect(options.positionTable == "positions.npy" && options.chunksDirectory == "/models")
        #expect(options.resize == .originBucket && options.embeddingTable == "/table.npy")
        #expect(invocation.assets == "/tokenizer")
        let defaultOptions = try #require(CLIInvocation.parse(["verify-vision", "picture.png", "", "--position-table", "p.npy"]).vision)
        #expect(defaultOptions.resize == .smart)
        #expect(defaultOptions.chunksDirectory == nil) // falls back to the installed model folder
    }
    @Test(arguments: [
        ["verify-vision", "pic.png", "text"],
        ["verify-vision", "pic.png", "--position-table", "p.npy"],
        ["verify-vision", "pic.png", "text", "--position-table", "p.npy", "--resize", "unknown"],
        ["verify-vision", "pic.png", "text", "--position-table", "p.npy", "--tower"],
        ["verify-vision", "pic.png", "text", "--position-table", "p.npy", "--port", "8080"],
        ["verify-vision", "pic.png", "text", "--position-table", "p.npy", "--extrope", "--extrope"],
        ["verify-vision", "pic.png", "text", "--position-table", "p.npy", "--position-table", "other.npy"]
    ])
    func rejectsInvalidOptions(_ arguments: [String]) {
        #expect(throws: CLIUsageError.self) { try CLIInvocation.parse(arguments) }
    }
    @Test func referenceComparisonDoesNotConfuseNormWithCosine() throws {
        let a: [Float] = [2] + Array(repeating: 0, count: 2047)
        let same: [Float] = [0.5] + Array(repeating: 0, count: 2047)
        let other: [Float] = [0, 1] + Array(repeating: 0, count: 2046)
        let equal = try VisionReferenceComparison(embedding: a, reference: same)
        #expect(equal.gatePassed && equal.cosine == 1 && equal.threshold == 0.99)
        let different = try VisionReferenceComparison(embedding: a, reference: other)
        #expect(!different.gatePassed && different.cosine == 0)
        #expect(throws: EmbedANEError.self) { try VisionReferenceComparison(embedding: a, reference: Array(repeating: 0, count: 2048)) }
        #expect(throws: EmbedANEError.self) { try VisionReferenceComparison(embedding: a, reference: [.nan]) }
    }
    @Test func malformedReferenceReturnsVerificationExitBeforeAnyModelLoad() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data([1,2,3]).write(to: root.appendingPathComponent("image.png"))
        try Data("not an npy".utf8).write(to: root.appendingPathComponent("ref.npy"))
        let result = await CLIApplication.run(arguments: ["verify-vision", "image.png", "text", "--position-table", "missing.npy", "--reference", "ref.npy"],
            store: ConfigurationStore(home: root), workingDirectory: root, environment: [:], executor: ProductionCommands())
        #expect(result.exitCode == 2)
        #expect(String(decoding: result.stderr, as: UTF8.self).contains("NPY"))
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent(".embed-ane/config.yaml").path))
    }
}
