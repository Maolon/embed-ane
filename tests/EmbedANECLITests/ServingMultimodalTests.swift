import EmbedANECore
import EmbedANETestSupport
import Foundation
import Testing
@testable import embed_ane

struct ServingMultimodalTests {
    @Test func eagerLoadAndTeardownOwnBothRuntimesAndKeepPlainPredictionsUnchanged() async throws {
        let text = MockPredictor(), vision = MockMultimodalPredictor()
        let runtime = ServingMultimodalRuntime(text: text, preparer: CountingPreparer(), vision: vision)
        let lifecycle = try LifecycleActor(predictor: runtime)
        await vision.text.loadGate.close()
        let loading = Task { try await lifecycle.load() }
        try await eventually { await vision.text.loadCount == 1 }
        #expect(await text.loadCount == 1)
        #expect(await lifecycle.snapshot().state == .loading)
        do { _ = try await runtime.prepare(["a"]); Issue.record("Preparation waited behind vision load") }
        catch { #expect(error as? EmbedANEError == .loading) }
        await vision.text.loadGate.open()
        let report = try await loading.value
        #expect(report.perChunkNS == Array(repeating: 2, count: 6)); #expect(!report.computePlanChecked)
        let result = try await lifecycle.embed(["a"], using: runtime)
        #expect(result.embeddings == [[97, 6] + Array(repeating: 0, count: 2046)]) // 1 byte + 4 template ids + <embedding>
        let image = try await lifecycle.submitImage(.init(imageData: Data([1]), text: "a"))
        #expect(image.promptTokens == 66)
        #expect(await text.predictCount == 1); #expect(await vision.text.predictCount == 0)
        await vision.text.unloadGate.close()
        let unloading = Task { try await lifecycle.unload() }
        try await eventually { await vision.text.unloadCount == 2 }
        #expect(await text.unloadCount == 0) // Keep the plain bundle lease until vision releases shared assets.
        await vision.text.unloadGate.open()
        _ = try await unloading.value
        #expect(await text.unloadCount == 1)
    }

    @Test func failedVisionLoadCleansUpBothAdapters() async throws {
        let text = MockPredictor(), vision = MockMultimodalPredictor()
        await vision.text.setLoadFailures(1)
        let runtime = ServingMultimodalRuntime(text: text, preparer: CountingPreparer(), vision: vision)
        do { _ = try await runtime.load(); Issue.record("Failed load succeeded") } catch {}
        #expect(await text.unloadCount == 1); #expect(await vision.text.unloadCount == 2)
    }

    @Test func injectedArtifactsAreCheckedBeforeAnyCoreMLLoad() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let tower = root.appendingPathComponent("tower.mlmodelc")
        let chunks = root.appendingPathComponent("chunks")
        let positions = root.appendingPathComponent("positions.npy")
        let config = ServiceConfiguration(modelRoot: root.path, visionTowerPath: tower.path,
            visionExtropeChunksDirectory: chunks.path, visionPositionTablePath: positions.path)
        #expect(try VisionServingArtifacts.paths(.init()) == nil)
        #expect(throws: (any Error).self) { try VisionServingArtifacts.paths(config) }
        func makeModel(_ model: URL) throws {
            try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
            try Data("readability fixture, not a model".utf8).write(to: model.appendingPathComponent("model.mil"))
        }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("readability fixture, not a table".utf8).write(to: positions)
        // Legacy extrope names are no longer resolved: chunks must be chunk{i}.mlmodelc.
        let legacy = (0..<6).map { chunks.appendingPathComponent("wemm_chunk\($0)_of6_seq512_extrope.mlmodelc") }
        for model in [tower] + legacy { try makeModel(model) }
        #expect(throws: (any Error).self) { try VisionServingArtifacts.paths(config) }
        for model in legacy { try FileManager.default.removeItem(at: model) }
        let models = [tower] + (0..<6).map { chunks.appendingPathComponent("chunk\($0).mlmodelc") }
        for model in models.dropFirst() { try makeModel(model) }
        let resolved = try VisionServingArtifacts.paths(config)
        let paths = try #require(resolved)
        #expect(paths.chunks.count == 6); #expect(paths.positionTable == positions)
        #expect(paths.tower.path == tower.path); #expect(paths.chunks.map(\.path) == models.dropFirst().map(\.path))
        #expect(paths.tokenizerDirectory == root.appendingPathComponent(ModelABI.defaultModelID))
        let first = try #require(models.first).appendingPathComponent("model.mil")
        try FileManager.default.removeItem(at: first)
        try FileManager.default.createSymbolicLink(at: first, withDestinationURL: positions)
        #expect(throws: EmbedANEError.self) { try VisionServingArtifacts.paths(config) }
    }
}
