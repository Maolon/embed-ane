import Foundation
import Testing
@testable import EmbedANECore

@Suite("CoreAI Bookmark Store Contracts")
struct CoreAIBookmarkStoreTests {
    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-bookmarks-\(UUID())")
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test func writeReadAndReplaceRoundtrip() throws {
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = CoreAIBookmarkStore(directory: dir)

        // Read nonexistent
        let missing = try store.read(modelID: "model1", asset: "chunk0")
        #expect(missing == nil)

        // Write
        let data1 = Data("bookmark-chunk-0-v1".utf8)
        try store.save(data1, modelID: "model1", asset: "chunk0")

        let read1 = try store.read(modelID: "model1", asset: "chunk0")
        #expect(read1 == data1)

        // Replace
        let data2 = Data("bookmark-chunk-0-v2-refreshed".utf8)
        try store.save(data2, modelID: "model1", asset: "chunk0")

        let read2 = try store.read(modelID: "model1", asset: "chunk0")
        #expect(read2 == data2)

        // Verify file permissions 0600
        let filePath = dir.appendingPathComponent("model1-chunk0.bookmark").path
        let attributes = try FileManager.default.attributesOfItem(atPath: filePath)
        let perms = (attributes[.posixPermissions] as? NSNumber)?.intValue
        #expect(perms == 0o600)
    }

    @Test func separateAssetsAndModelsDoNotCollide() throws {
        let dir = temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let store = CoreAIBookmarkStore(directory: dir)

        try store.save(Data("m1c0".utf8), modelID: "m1", asset: "chunk0")
        try store.save(Data("m1c1".utf8), modelID: "m1", asset: "chunk1")
        try store.save(Data("m2c0".utf8), modelID: "m2", asset: "chunk0")

        #expect(try store.read(modelID: "m1", asset: "chunk0") == Data("m1c0".utf8))
        #expect(try store.read(modelID: "m1", asset: "chunk1") == Data("m1c1".utf8))
        #expect(try store.read(modelID: "m2", asset: "chunk0") == Data("m2c0".utf8))
    }
}
