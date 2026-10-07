import EmbedANECore
import EmbedANEDownload
import EmbedANEHTTP
import EmbedANETestSupport
import Foundation
import Testing
@testable import embed_ane

private struct CLIFixture: Sendable {
    let home: URL
    let work: URL
    let models: URL
    var store: ConfigurationStore { .init(home: home) }
    init() throws {
        home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        work = home.appendingPathComponent("work")
        models = home.appendingPathComponent("models")
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)
        try store.save(.init(modelRoot: models.path))
    }
    func remove() { try? FileManager.default.removeItem(at: home) }
    func run(_ args: [String], executor: any CLICommandExecuting = ProductionCommands()) async -> CLIResult {
        await CLIApplication.run(arguments: args, store: store, workingDirectory: work, environment: [:], executor: executor)
    }
    /// Digest/manifest fixture only, deliberately NOT a usable CoreML bundle.
    func installFixture() throws -> URL {
        let root = models.appendingPathComponent(ModelABI.defaultModelID)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let directory = try SecureDirectory(root)
        var files: [ArtifactFile] = []
        for name in ["tokenizer.json", "tokenizer_config.json", "embed_table.fp16.npy"] {
            let bytes = Data("fixture".utf8)
            try directory.atomicWrite(bytes, to: name)
            files.append(try .init(path: name, sha256: FileDigest.sha256(bytes), size: Int64(bytes.count)))
        }
        for index in 0..<6 {
            let prefix = "chunks/chunk\(index).mlmodelc/"
            let bytes = Data("fixture-\(index)".utf8)
            try directory.atomicWrite(bytes, to: prefix + "weights.bin")
            let manifest = Data("weights.bin \(bytes.count) \(FileDigest.sha256(bytes))\n".utf8)
            try directory.atomicWrite(manifest, to: prefix + "manifest.txt")
            files.append(try .init(path: prefix + "manifest.txt", sha256: FileDigest.sha256(manifest), size: Int64(manifest.count)))
        }
        let yaml = """
        spec_version: 1
        model: {id: \(ModelABI.defaultModelID), dim: 2048, max_seq: 512, normalize: l2}
        source: {repo: fixtures/test, revision: \(String(repeating: "a", count: 40))}
        runtime: {compute_units: cpu_and_ne, pad_side: right, mask_dtype: fp32}
        files:
        \(files.map { "  - {path: \($0.path), sha256: \($0.sha256), size: \($0.size)}" }.joined(separator: "\n"))
        """
        let spec = try ModelSpec.parse(yaml)
        let manifest = try InstallManifest(specYAML: yaml, spec: spec, resolvedCommit: String(repeating: "a", count: 40))
        try directory.atomicWrite(Data(manifest.yaml().utf8), to: "manifest.yaml")
        let specURL = work.appendingPathComponent("spec.yaml")
        try Data(yaml.utf8).write(to: specURL)
        return specURL
    }
    func provenance() throws -> VerificationReport {
        try JSONDecoder().decode(VerificationReport.self, from: JSONSerialization.data(withJSONObject: [
            "modelID": ModelABI.defaultModelID, "resolvedCommit": String(repeating: "a", count: 40),
            "specDigest": String(repeating: "b", count: 64),
            "artifactDigests": ["tokenizer.json": String(repeating: "c", count: 64),
                                "tokenizer_config.json": String(repeating: "d", count: 64)],
        ]))
    }
    func gate(for verification: VerificationReport, pass: Bool = true) throws {
        let report = try JSONDecoder().decode(TokenizerVerificationReport.self, from: JSONSerialization.data(withJSONObject: [
            "cases": 1000, "minimumCases": 1000, "mismatches": pass ? 0 : 1,
            "rejectedOverlengthCases": 0, "gatePassed": pass, "examples": [], "assetDigests": verification.artifactDigests,
        ]))
        let receipt = TokenizerGateReceipt(receiptVersion: 1, createdAt: "fixture",
            goldenSHA256: String(repeating: "e", count: 64), report: report)
        try StateFileStorage(directory: home.appendingPathComponent(".embed-ane"))
            .write("tokenizer-gate.json", data: CLIJSON.encode(receipt))
    }
}

struct CLIProductionTests {
    @Test func verifyAndOfflineFetchUseTheRealDownloaderWithoutModelsOrNetwork() async throws {
        let fixture = try CLIFixture(); defer { fixture.remove() }
        let spec = try fixture.installFixture()
        let verified = await fixture.run(["verify", ModelABI.defaultModelID])
        #expect(verified.exitCode == 0)
        #expect(try JSONDecoder().decode(VerificationReport.self, from: verified.stdout).modelID == ModelABI.defaultModelID)
        let fetched = await fixture.run(["fetch", spec.path, "--offline"])
        #expect(fetched.exitCode == 0)
        #expect(try JSONDecoder().decode(FetchReport.self, from: fetched.stdout).disposition == .verifiedOffline)
        try Data("corrupt".utf8).write(to: fixture.models.appendingPathComponent(ModelABI.defaultModelID + "/tokenizer.json"))
        let invalid = await fixture.run(["verify", ModelABI.defaultModelID])
        #expect(invalid.exitCode == 2)
        #expect(invalid.stdout.isEmpty)
    }
    @Test func benchWritesInspectableMockSamplesReloadAndProvenance() async throws {
        let fixture = try CLIFixture(); defer { fixture.remove() }
        let predictor = MockPredictor()
        let verification = try fixture.provenance()
        let executor = ProductionCommands(makeRuntime: { _ in
            .init(predictor: predictor, preparer: CountingPreparer(), verification: { verification }, audit: { [] })
        })
        try Data("a\nb\n".utf8).write(to: fixture.work.appendingPathComponent("corpus.txt"))
        let result = await fixture.run(["bench", "corpus.txt"], executor: executor)
        #expect(result.exitCode == 0)
        let summary = try JSONDecoder().decode(BenchmarkSummary.self, from: result.stdout)
        #expect(summary.samples == 2); #expect(summary.parityGatePassed == nil)
        let artifact = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: summary.report))) as? [String: Any])
        #expect(artifact["report_version"] as? Int == 1)
        #expect(artifact["same_process_reload"] != nil); #expect(artifact["reload_prediction"] != nil)
        #expect((artifact["process_rss"] as? [String: Any])?["scope"] as? String == "process")
        let measurement = try #require(artifact["measurements"] as? [String: Any])
        let samples = try #require(measurement["samples"] as? [[String: Any]])
        #expect(samples.map { $0["text"] as? String } == ["a", "b"])
        #expect((samples[0]["embedding"] as? [Double])?.first == 97)
        #expect(await predictor.loadCount == 2); #expect(await predictor.unloadCount == 2)
        #expect(await predictor.predictCount == 3)
    }
    @Test func parityFailureStillWritesRawResultAndExitsTwo() async throws {
        let fixture = try CLIFixture(); defer { fixture.remove() }
        let predictor = MockPredictor(); let verification = try fixture.provenance()
        try fixture.gate(for: verification)
        try Data("a\n".utf8).write(to: fixture.work.appendingPathComponent("corpus.txt"))
        // The mock row for "a" is [97, n_tokens 6] (1 byte + 4 chat-template ids + <embedding>).
        let opposite = [-97.0, -6.0] + Array(repeating: 0.0, count: 2046)
        try JSONSerialization.data(withJSONObject: ["text": "a", "embedding": opposite])
            .write(to: fixture.work.appendingPathComponent("ref.jsonl"))
        let executor = ProductionCommands(makeRuntime: { _ in
            .init(predictor: predictor, preparer: CountingPreparer(), verification: { verification }, audit: { [] })
        })
        let result = await fixture.run(["bench", "corpus.txt", "--reference", "ref.jsonl"], executor: executor)
        #expect(result.exitCode == 2)
        let summary = try JSONDecoder().decode(BenchmarkSummary.self, from: result.stdout)
        #expect(summary.parityGatePassed == false)
        #expect(abs((summary.minCos ?? 0) + 1) < 1e-12)
        #expect(FileManager.default.fileExists(atPath: summary.report))
    }
    @Test func missingAndMismatchedGateNeverPredict() async throws {
        let fixture = try CLIFixture(); defer { fixture.remove() }
        let predictor = MockPredictor(); let verification = try fixture.provenance()
        try Data("a\n".utf8).write(to: fixture.work.appendingPathComponent("corpus.txt"))
        try JSONSerialization.data(withJSONObject: ["text": "a", "embedding": [97.0, 2.0] + Array(repeating: 0.0, count: 2046)])
            .write(to: fixture.work.appendingPathComponent("ref.jsonl"))
        let executor = ProductionCommands(makeRuntime: { _ in
            .init(predictor: predictor, preparer: CountingPreparer(), verification: { verification }, audit: { [] })
        })
        let args = ["bench", "corpus.txt", "--reference", "ref.jsonl"]
        #expect(await fixture.run(args, executor: executor).exitCode == 2)
        #expect(await predictor.loadCount == 0)
        try fixture.gate(for: verification, pass: false)
        #expect(await fixture.run(args, executor: executor).exitCode == 2)
        #expect(await predictor.predictCount == 0)
        #expect(await predictor.unloadCount == 1)
    }
    @Test func serveCompositionLoadsBeforeListenerAndUnloadsOnFailure() async throws {
        let fixture = try CLIFixture(); defer { fixture.remove() }
        let predictor = MockPredictor()
        let executor = ProductionCommands(makeRuntime: { _ in
            .init(predictor: predictor, preparer: CountingPreparer(), verification: { nil }, audit: { [] })
        }, runServer: { server in
            #expect(await server.lifecycle.snapshot().state == .ready)
            throw EmbedANEError.io(path: "127.0.0.1:8080", reason: "fixture bind failure")
        })
        #expect(await fixture.run(["serve"], executor: executor).exitCode == 1)
        #expect(await predictor.loadCount == 1); #expect(await predictor.unloadCount == 1)
    }
    @Test func successfulGateWithStaleAssetDigestIsRejected() throws {
        let fixture = try CLIFixture(); defer { fixture.remove() }
        let verification = try fixture.provenance()
        try fixture.gate(for: verification)
        let receipt = try TokenizerGateReceipt.read(store: fixture.store)
        try receipt.validate(artifactDigests: verification.artifactDigests)
        var changed = verification.artifactDigests
        changed["tokenizer.json"] = String(repeating: "f", count: 64)
        #expect(throws: EmbedANEError.self) { try receipt.validate(artifactDigests: changed) }
    }

}
