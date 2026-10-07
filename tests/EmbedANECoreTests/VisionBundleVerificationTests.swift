import CryptoKit
import Dispatch
import EmbedANECore
import EmbedANETestSupport
import Foundation
import Testing

private final class TempDir {
    let url: URL
    init() throws {
        url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }
    deinit { try? FileManager.default.removeItem(at: url) }
}

private func sha256Hex(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func createMockChunk(at dir: URL) throws -> (manifestData: Data, files: [(path: String, sha256: String, size: Int)]) {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let metaData = Data("{\"model\": \"test\"}\n".utf8)
    let coreData = Data("weights-data-binary\n".utf8)
    try metaData.write(to: dir.appendingPathComponent("metadata.json"))
    try coreData.write(to: dir.appendingPathComponent("coremldata.bin"))

    let entries = [
        ("coremldata.bin", sha256Hex(coreData), coreData.count),
        ("metadata.json", sha256Hex(metaData), metaData.count)
    ]
    let manifestText = entries.map { "\($0.0) \($0.2) \($0.1)\n" }.joined()
    let manifestData = Data(manifestText.utf8)
    try manifestData.write(to: dir.appendingPathComponent("manifest.txt"))
    return (manifestData, entries)
}

/// The release multimodal layout: the text bundle files, whose
/// `chunks/chunk{0..5}.mlmodelc` ARE the position-input (extrope) decoder, plus
/// the root-level two-frame `vision_tower.mlmodelc` and `vision_abs_pos.fp32.npy`.
private func buildTestMultimodalBundle(at root: URL, id: String = "wemm-embedding-2b") throws {
    let bundle = root.appendingPathComponent(id)
    try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)

    var specFiles: [ArtifactFile] = []

    // 1. Tokenizer files + table
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        let data = Data("{\"name\": \"\(name)\"}\n".utf8)
        try data.write(to: bundle.appendingPathComponent(name))
        specFiles.append(try ArtifactFile(path: name, sha256: sha256Hex(data), size: Int64(data.count)))
    }
    let tableData = Data("embedding-table-fp16-data\n".utf8)
    try tableData.write(to: bundle.appendingPathComponent("embed_table.fp16.npy"))
    specFiles.append(try ArtifactFile(path: "embed_table.fp16.npy", sha256: sha256Hex(tableData), size: Int64(tableData.count)))

    // 2. Decoder chunks (0..<6); in a multimodal bundle these are the extrope graphs.
    let chunksDir = bundle.appendingPathComponent("chunks")
    for c in 0..<6 {
        let chunkDir = chunksDir.appendingPathComponent("chunk\(c).mlmodelc")
        let (manifestData, _) = try createMockChunk(at: chunkDir)
        let path = "chunks/chunk\(c).mlmodelc/manifest.txt"
        specFiles.append(try ArtifactFile(path: path, sha256: sha256Hex(manifestData), size: Int64(manifestData.count)))
    }

    // 3. Vision tower
    let towerDir = bundle.appendingPathComponent("vision_tower.mlmodelc")
    let (towerManifest, _) = try createMockChunk(at: towerDir)
    specFiles.append(try ArtifactFile(path: ModelSpec.visionTowerManifest, sha256: sha256Hex(towerManifest), size: Int64(towerManifest.count)))

    // 4. Position table
    let posData = Data("vision-position-table-data\n".utf8)
    try posData.write(to: bundle.appendingPathComponent("vision_abs_pos.fp32.npy"))
    specFiles.append(try ArtifactFile(path: ModelSpec.visionPositionTable, sha256: sha256Hex(posData), size: Int64(posData.count)))

    specFiles.sort(by: { $0.path.utf8.lexicographicallyPrecedes($1.path.utf8) })

    let resolvedCommit = String(repeating: "c", count: 40)
    let fileLines = specFiles.map { "  - {path: \($0.path), sha256: \($0.sha256), size: \($0.size)}" }.joined(separator: "\n")
    let specYAML = """
    spec_version: 1
    model: {id: \(id), dim: 2048, max_seq: 512, normalize: l2}
    source: {repo: maolon/WeMM-Embedding-2B-CoreML-ANE, revision: \(resolvedCommit)}
    files:
    \(fileLines)
    runtime: {compute_units: cpu_and_ne, pad_side: right, mask_dtype: fp32}
    """
    try Data(specYAML.utf8).write(to: bundle.appendingPathComponent("spec.yaml"))

    let parsedSpec = try ModelSpec.parse(specYAML)
    let installManifest = try InstallManifest(specYAML: specYAML, spec: parsedSpec, resolvedCommit: resolvedCommit)
    let manifestYAML = try installManifest.yaml()
    try Data(manifestYAML.utf8).write(to: bundle.appendingPathComponent("manifest.yaml"))
}

private func specYAML(paths: [String]) -> String {
    let digest = String(repeating: "a", count: 64)
    return """
    spec_version: 1
    model: {id: wemm-embedding-2b, dim: 2048, max_seq: 512, normalize: l2}
    source: {repo: maolon/WeMM-Embedding-2B-CoreML-ANE, revision: v1}
    files:
    \(paths.map { "  - {path: \($0), sha256: \(digest), size: 1}" }.joined(separator: "\n"))
    runtime: {compute_units: cpu_and_ne, pad_side: right, mask_dtype: fp32}
    """
}

private let textBundlePaths = ["tokenizer.json", "tokenizer_config.json", "embed_table.fp16.npy"]
    + (0..<6).map { "chunks/chunk\($0).mlmodelc/manifest.txt" }

private func errorMessage(_ body: () throws -> Void) -> String? {
    do { try body(); return nil }
    catch let error as EmbedANEError { return error.errorDescription ?? String(describing: error) }
    catch { return String(describing: error) }
}

/// Leases record acquisition and release, like the installation lease.
private final class LeaseLog: @unchecked Sendable {
    private let lock = NSLock()
    private var acquired = 0, released = 0
    var gate: DispatchSemaphore?
    var entered = false
    var counts: (acquired: Int, released: Int) { lock.withLock { (acquired, released) } }
    var hasEntered: Bool { lock.withLock { entered } }
    final class Lease: Sendable {
        let log: LeaseLog
        init(_ log: LeaseLog) { self.log = log }
        deinit { log.lock.withLock { log.released += 1 } }
    }
    func acquire() -> any Sendable {
        lock.withLock { acquired += 1; entered = true }
        gate?.wait()
        return Lease(self)
    }
}

@Suite("Vision bundle and verifier contracts")
struct VisionBundleVerificationTests {
    @Test func multimodalBundleVerificationCoversVisionArtifactsAndMetadataJSON() throws {
        let temp = try TempDir()
        try buildTestMultimodalBundle(at: temp.url)
        let realBundle = temp.url.appendingPathComponent("wemm-embedding-2b")

        let report = try LocalAssetVerifier.verify(at: realBundle, expectedID: "wemm-embedding-2b")
        #expect(report.modelID == "wemm-embedding-2b")

        // Check that vision tower and its inner files (metadata.json) are in artifactDigests
        #expect(report.artifactDigests["vision_tower.mlmodelc/manifest.txt"] != nil)
        #expect(report.artifactDigests["vision_tower.mlmodelc/metadata.json"] != nil)
        #expect(report.artifactDigests["vision_tower.mlmodelc/coremldata.bin"] != nil)

        // The position-input decoder chunks and their inner files
        for c in 0..<6 {
            #expect(report.artifactDigests["chunks/chunk\(c).mlmodelc/manifest.txt"] != nil)
            #expect(report.artifactDigests["chunks/chunk\(c).mlmodelc/metadata.json"] != nil)
        }

        // Check position table
        #expect(report.artifactDigests["vision_abs_pos.fp32.npy"] != nil)
        // No legacy names are part of a release bundle.
        #expect(!report.artifactDigests.keys.contains { $0.contains("vision_tower_base") || $0.contains("extrope") })
    }

    @Test func tamperedVisionMetadataJSONFailsVerification() throws {
        let temp = try TempDir()
        try buildTestMultimodalBundle(at: temp.url)
        let bundle = temp.url.appendingPathComponent("wemm-embedding-2b")

        let metaURL = bundle.appendingPathComponent("vision_tower.mlmodelc/metadata.json")
        try Data("{\"tampered\": true}\n".utf8).write(to: metaURL)

        #expect(throws: EmbedANEError.self) {
            try LocalAssetVerifier.verify(at: bundle, expectedID: "wemm-embedding-2b")
        }
    }

    @Test func tamperedVisionPositionTableFailsVerification() throws {
        let temp = try TempDir()
        try buildTestMultimodalBundle(at: temp.url)
        let bundle = temp.url.appendingPathComponent("wemm-embedding-2b")

        let posURL = bundle.appendingPathComponent("vision_abs_pos.fp32.npy")
        try Data("tampered-position-data".utf8).write(to: posURL)

        #expect(throws: EmbedANEError.self) {
            try LocalAssetVerifier.verify(at: bundle, expectedID: "wemm-embedding-2b")
        }
    }

    @Test func missingVisionPositionTableFailsVerification() throws {
        let temp = try TempDir()
        try buildTestMultimodalBundle(at: temp.url)
        let bundle = temp.url.appendingPathComponent("wemm-embedding-2b")

        let posURL = bundle.appendingPathComponent("vision_abs_pos.fp32.npy")
        try FileManager.default.removeItem(at: posURL)

        #expect(throws: (any Error).self) {
            try LocalAssetVerifier.verify(at: bundle, expectedID: "wemm-embedding-2b")
        }
    }

    @Test func multimodalSpecRequiresTowerAndPositionTableTogether() throws {
        let text = try ModelSpec.parse(specYAML(paths: textBundlePaths))
        #expect(!text.isMultimodal)
        let multimodal = try ModelSpec.parse(specYAML(paths: textBundlePaths + [ModelSpec.visionTowerManifest, ModelSpec.visionPositionTable]))
        #expect(multimodal.isMultimodal)
        #expect(ModelSpec.visionTowerManifest == "vision_tower.mlmodelc/manifest.txt")
        #expect(ModelSpec.visionPositionTable == "vision_abs_pos.fp32.npy")

        for incomplete in [[ModelSpec.visionTowerManifest], [ModelSpec.visionPositionTable]] {
            let message = errorMessage { _ = try ModelSpec.parse(specYAML(paths: textBundlePaths + incomplete)) }
            #expect(message?.contains("a multimodal bundle must include both vision_tower.mlmodelc and vision_abs_pos.fp32.npy") == true, "\(incomplete)")
        }
    }

    @Test func incompleteVisionBundleInSpecYAMLIsRejected() throws {
        // Legacy single-frame tower name: a tower must be vision_tower.mlmodelc,
        // so the legacy name alongside the table is an incomplete bundle.
        #expect(throws: EmbedANEError.self) {
            try ModelSpec.parse(specYAML(paths: textBundlePaths + ["vision_tower_base.mlmodelc/manifest.txt", ModelSpec.visionPositionTable]))
        }
        // Legacy extrope chunk names inside chunks/ are not one of the six required manifests.
        let legacyChunk = errorMessage {
            _ = try ModelSpec.parse(specYAML(paths: textBundlePaths + [ModelSpec.visionTowerManifest, ModelSpec.visionPositionTable,
                                                                       "chunks/wemm_chunk0_of6_seq512_extrope.mlmodelc/manifest.txt"]))
        }
        #expect(legacyChunk?.contains("chunk members must be listed by their inner manifest: chunks/wemm_chunk0_of6_seq512_extrope.mlmodelc/manifest.txt") == true)
        // Model package members must still be listed by their inner manifest.
        #expect(throws: EmbedANEError.self) {
            try ModelSpec.parse(specYAML(paths: textBundlePaths + ["vision_tower.mlmodelc/metadata.json", ModelSpec.visionPositionTable]))
        }
    }

    @Test func relativeVisionDefaultsResolveWhenFilesExistAndRemainNilWhenAbsent() throws {
        let temp = try TempDir()
        let store = ConfigurationStore(home: temp.url)

        // 1. Without vision files (empty model directory)
        let emptyConfig = ServiceConfiguration(modelID: "wemm-embedding-2b", modelRoot: temp.url.path)
        let expandedEmpty = store.expandingPaths(emptyConfig)
        #expect(expandedEmpty.visionTowerPath == nil)
        #expect(expandedEmpty.visionExtropeChunksDirectory == nil)
        #expect(expandedEmpty.visionPositionTablePath == nil)

        // 2. With vision files
        try buildTestMultimodalBundle(at: temp.url)
        let bundleConfig = ServiceConfiguration(modelID: "wemm-embedding-2b", modelRoot: temp.url.path)
        let expandedWithVision = store.expandingPaths(bundleConfig)

        let expectedModelDir = temp.url.appendingPathComponent("wemm-embedding-2b")
        #expect(expandedWithVision.visionTowerPath == expectedModelDir.appendingPathComponent("vision_tower.mlmodelc").path)
        #expect(expandedWithVision.visionExtropeChunksDirectory == expectedModelDir.appendingPathComponent("chunks").path)
        #expect(expandedWithVision.visionPositionTablePath == expectedModelDir.appendingPathComponent("vision_abs_pos.fp32.npy").path)

        // 3. Explicit absolute paths are preserved over auto-detection
        let explicitConfig = ServiceConfiguration(
            modelID: "wemm-embedding-2b", modelRoot: temp.url.path,
            visionTowerPath: "/custom/tower",
            visionExtropeChunksDirectory: "/custom/chunks",
            visionPositionTablePath: "/custom/pos.npy"
        )
        let expandedExplicit = store.expandingPaths(explicitConfig)
        #expect(expandedExplicit.visionTowerPath == "/custom/tower")
        #expect(expandedExplicit.visionExtropeChunksDirectory == "/custom/chunks")
        #expect(expandedExplicit.visionPositionTablePath == "/custom/pos.npy")

        // 4. Each release component is required: without the table, or with a
        // missing decoder chunk, nothing is detected.
        let model = expectedModelDir
        try FileManager.default.moveItem(at: model.appendingPathComponent("vision_abs_pos.fp32.npy"), to: temp.url.appendingPathComponent("pos.npy"))
        #expect(ConfigurationStore.detectVisionArtifacts(in: model) == nil)
        try FileManager.default.moveItem(at: temp.url.appendingPathComponent("pos.npy"), to: model.appendingPathComponent("vision_abs_pos.fp32.npy"))
        #expect(ConfigurationStore.detectVisionArtifacts(in: model) != nil)
        try FileManager.default.removeItem(at: model.appendingPathComponent("chunks/chunk5.mlmodelc"))
        #expect(ConfigurationStore.detectVisionArtifacts(in: model) == nil)
    }

    @Test func legacyVisionLayoutsAreNotDetected() throws {
        let temp = try TempDir()
        let fm = FileManager.default
        func makeDirectory(_ path: String, in base: URL) throws {
            try fm.createDirectory(at: base.appendingPathComponent(path), withIntermediateDirectories: true)
        }
        // Each legacy layout: text chunks/ plus the old tower name and old
        // extrope chunk names (root, chunks/, or extrope_chunks/).
        for (index, extropeDirectory) in ["", "chunks/", "extrope_chunks/"].enumerated() {
            let model = temp.url.appendingPathComponent("legacy-\(index)")
            try makeDirectory("vision_tower_base.mlmodelc", in: model)
            for c in 0..<6 {
                try makeDirectory("chunks/chunk\(c).mlmodelc", in: model)
                try makeDirectory("\(extropeDirectory)wemm_chunk\(c)_of6_seq512_extrope.mlmodelc", in: model)
            }
            try Data("table".utf8).write(to: model.appendingPathComponent("vision_abs_pos.fp32.npy"))
            #expect(ConfigurationStore.detectVisionArtifacts(in: model) == nil, "\(extropeDirectory)")
            // A legacy tower name with legacy root chunks but no chunks/ is not detected either.
            try makeDirectory("vision_tower.mlmodelc", in: model)
            try fm.removeItem(at: model.appendingPathComponent("chunks/chunk0.mlmodelc"))
            #expect(ConfigurationStore.detectVisionArtifacts(in: model) == nil, "\(extropeDirectory)")
        }
        // Legacy files do not leak into configuration either.
        let store = ConfigurationStore(home: temp.url)
        let expanded = store.expandingPaths(ServiceConfiguration(modelID: "legacy-0", modelRoot: temp.url.path))
        #expect(expanded.visionTowerPath == nil && expanded.visionExtropeChunksDirectory == nil && expanded.visionPositionTablePath == nil)
    }

    @Test func servingRuntimeFactorySelectsMultimodalWhenConfiguredAndTextWhenNot() throws {
        let temp = try TempDir()
        try buildTestMultimodalBundle(at: temp.url)
        let store = ConfigurationStore(home: temp.url)

        final class DummyLease: @unchecked Sendable {}
        let dummyLease: @Sendable () throws -> any Sendable = { DummyLease() }

        // Mock text predictor & preparer
        final class MockTextAndPreparer: EmbeddingPredictor, EmbeddingPreparer, @unchecked Sendable {
            func load() async throws -> LoadReport {
                LoadReport(perChunkNS: Array(repeating: 1, count: 6), residentBytes: 100, computePlanChecked: false)
            }
            func predict(_ request: PredictRequest) async throws -> PredictResult {
                PredictResult(embeddings: [[Float](repeating: 0, count: 2048)], tableLookupNS: 1, perChunkNS: Array(repeating: 1, count: 6))
            }
            func unload() async throws -> UnloadReport { UnloadReport(residentBytes: 0) }
            func prepare(_ texts: [String]) async throws -> PredictRequest {
                try PredictRequest(inputs: texts.map { _ in try TokenizedInput(contentIDs: [1, 2]) })
            }
        }

        final class MockVision: MultimodalEmbeddingPredictor, @unchecked Sendable {
            func load() async throws -> LoadReport {
                LoadReport(perChunkNS: Array(repeating: 2, count: 6), residentBytes: 200, computePlanChecked: false)
            }
            func predict(_ request: PredictRequest) async throws -> PredictResult {
                PredictResult(embeddings: [[Float](repeating: 0, count: 2048)], tableLookupNS: 1, perChunkNS: Array(repeating: 1, count: 6))
            }
            func unload() async throws -> UnloadReport { UnloadReport(residentBytes: 0) }
            func predictImage(_ request: ImageEmbeddingRequest) async throws -> ImageEmbeddingResult {
                ImageEmbeddingResult(embedding: [Float](repeating: 0, count: 2048), promptTokens: 66, tokenizeNS: 10)
            }
        }

        // Test with vision: injected makers keep the two-runtime composition.
        let configWithVision = store.expandingPaths(ServiceConfiguration(modelID: "wemm-embedding-2b", modelRoot: temp.url.path))
        let componentsVision = try ServingRuntimeFactory.make(
            configuration: configWithVision,
            acquireLease: dummyLease,
            makeTextPredictor: { _, _ in MockTextAndPreparer() },
            makeVisionPredictor: { _ in MockVision() }
        )
        #expect(componentsVision.isMultimodal)
        #expect(componentsVision.predictor is any MultimodalEmbeddingPredictor)
        #expect(componentsVision.predictor is ServingMultimodalRuntime)
        #expect(!(componentsVision.predictor is MultimodalPredictor))

        // Test without vision:
        let configNoVision = ServiceConfiguration(modelID: "wemm-embedding-2b", modelRoot: temp.url.path)
        // ensure paths are nil
        #expect(configNoVision.visionTowerPath == nil)
        let componentsText = try ServingRuntimeFactory.make(
            configuration: configNoVision,
            acquireLease: dummyLease,
            makeTextPredictor: { _, _ in MockTextAndPreparer() },
            makeVisionPredictor: { _ in MockVision() }
        )
        #expect(!componentsText.isMultimodal)
        #expect(!(componentsText.predictor is any MultimodalEmbeddingPredictor))
    }

    @Test func unifiedCoreMLRuntimeServesAMultimodalBundleWithOneLeasedPredictor() async throws {
        let temp = try TempDir()
        try buildTestMultimodalBundle(at: temp.url)
        let store = ConfigurationStore(home: temp.url)
        let configuration = store.expandingPaths(ServiceConfiguration(modelID: "wemm-embedding-2b", modelRoot: temp.url.path))
        #expect(configuration.engineBackend == .coreml)
        let log = LeaseLog()

        // No injected makers: one MultimodalPredictor is both predictor and
        // preparer. Construction does not load (or lease) anything.
        let components = try ServingRuntimeFactory.make(configuration: configuration, acquireLease: { log.acquire() })
        #expect(components.isMultimodal)
        let unified = try #require(components.predictor as? MultimodalPredictor)
        #expect(components.preparer as? MultimodalPredictor === unified)
        #expect(!(components.predictor is ServingMultimodalRuntime))
        #expect(components.verification?() == nil)
        #expect(components.audit?().isEmpty == true)
        #expect(log.counts.acquired == 0)

        // The (fixture) bundle verifies, then the fixture tokenizer cannot load:
        // the lease was taken for the attempt and released by the failure cleanup;
        // no verification is published for a failed load.
        await #expect(throws: (any Error).self) { try await unified.load() }
        #expect(log.counts.acquired == 1 && log.counts.released == 1)
        #expect(unified.verificationReport() == nil)

        // A tampered bundle fails BundleVerifier only after the lease is held.
        let bundle = temp.url.appendingPathComponent("wemm-embedding-2b")
        try Data("{\"tampered\": true}\n".utf8).write(to: bundle.appendingPathComponent("vision_tower.mlmodelc/metadata.json"))
        await #expect(throws: EmbedANEError.self) { try await unified.load() }
        #expect(log.counts.acquired == 2 && log.counts.released == 2)

        // Text preparation is not queued behind an in-progress load.
        let gate = DispatchSemaphore(value: 0)
        log.gate = gate
        let loading = Task { try await unified.load() }
        try await eventually { log.counts.acquired == 3 }
        await #expect(throws: EmbedANEError.loading) { try await unified.prepare(["a"]) }
        gate.signal()
        _ = try? await loading.value
        #expect(log.counts.released == 3)
        _ = try await unified.unload()
    }

    @Test func servingRuntimeFactoryPassesVisionResizeModeThrough() throws {
        let temp = try TempDir()
        try buildTestMultimodalBundle(at: temp.url)
        let store = ConfigurationStore(home: temp.url)

        final class DummyLease: @unchecked Sendable {}
        let dummyLease: @Sendable () throws -> any Sendable = { DummyLease() }

        final class MockTextAndPreparer: EmbeddingPredictor, EmbeddingPreparer, @unchecked Sendable {
            func load() async throws -> LoadReport {
                LoadReport(perChunkNS: Array(repeating: 1, count: 6), residentBytes: 100, computePlanChecked: false)
            }
            func predict(_ request: PredictRequest) async throws -> PredictResult {
                PredictResult(embeddings: [[Float](repeating: 0, count: 2048)], tableLookupNS: 1, perChunkNS: Array(repeating: 1, count: 6))
            }
            func unload() async throws -> UnloadReport { UnloadReport(residentBytes: 0) }
            func prepare(_ texts: [String]) async throws -> PredictRequest {
                try PredictRequest(inputs: texts.map { _ in try TokenizedInput(contentIDs: [1, 2]) })
            }
        }

        final class MockVision: MultimodalEmbeddingPredictor, @unchecked Sendable {
            func load() async throws -> LoadReport {
                LoadReport(perChunkNS: Array(repeating: 2, count: 6), residentBytes: 200, computePlanChecked: false)
            }
            func predict(_ request: PredictRequest) async throws -> PredictResult {
                PredictResult(embeddings: [[Float](repeating: 0, count: 2048)], tableLookupNS: 1, perChunkNS: Array(repeating: 1, count: 6))
            }
            func unload() async throws -> UnloadReport { UnloadReport(residentBytes: 0) }
            func predictImage(_ request: ImageEmbeddingRequest) async throws -> ImageEmbeddingResult {
                ImageEmbeddingResult(embedding: [Float](repeating: 0, count: 2048), promptTokens: 66, tokenizeNS: 10)
            }
        }

        final class ModeCapture: @unchecked Sendable {
            var captured: VisionResizeMode?
        }

        let capture = ModeCapture()

        // 1. Explicit .smart resize mode passes .smart
        let configSmart = store.expandingPaths(ServiceConfiguration(
            modelID: "wemm-embedding-2b",
            modelRoot: temp.url.path,
            visionResizeMode: .smart
        ))
        _ = try ServingRuntimeFactory.make(
            configuration: configSmart,
            acquireLease: dummyLease,
            makeTextPredictor: { _, _ in MockTextAndPreparer() },
            makeVisionPredictorWithMode: { _, mode in
                capture.captured = mode
                return MockVision()
            }
        )
        #expect(capture.captured == .smart)

        // 2. Default configuration passes .originBucket
        let configDefault = store.expandingPaths(ServiceConfiguration(
            modelID: "wemm-embedding-2b",
            modelRoot: temp.url.path
        ))
        _ = try ServingRuntimeFactory.make(
            configuration: configDefault,
            acquireLease: dummyLease,
            makeTextPredictor: { _, _ in MockTextAndPreparer() },
            makeVisionPredictorWithMode: { _, mode in
                capture.captured = mode
                return MockVision()
            }
        )
        #expect(capture.captured == .smart)
    }
}
