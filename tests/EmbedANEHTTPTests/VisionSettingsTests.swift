import EmbedANECore
import Foundation
import Testing

struct VisionSettingsTests {
    @Test func allThreeKeysPersistButOnlyTakeEffectAfterRestart() async throws {
        try await withHTTPFixture { f in
            let patch = try jsonData(["vision_tower_path": "~/tower.mlmodelc", "vision_extrope_chunks_dir": "~/extrope", "vision_position_table_path": "~/positions.npy"])
            let response = try await f.request("PUT", "/control/settings", body: patch, auth: true)
            #expect(response.status == 200)
            let result = try response.json()
            #expect(result["vision_tower_path"] == nil)
            let fields: WireJSON = .array([.string("vision_extrope_chunks_dir"), .string("vision_position_table_path"), .string("vision_tower_path")])
            #expect(result["restart_required"] == fields)
            let stored = try f.store.load()
            #expect(stored.visionTowerPath == f.store.home.appendingPathComponent("tower.mlmodelc").path)
            #expect(stored.visionExtropeChunksDirectory == f.store.home.appendingPathComponent("extrope").path)
            #expect(stored.visionPositionTablePath == f.store.home.appendingPathComponent("positions.npy").path)
            #expect(await f.server.lifecycle.supportsImages == false)
            #expect(try await f.request("PUT", "/control/settings", body: jsonData(["max_queue_depth": 2]), auth: true).json()["restart_required"] == fields)
            let before = try f.store.load()
            for invalid in ["{\"vision_tower_path\":null}", "{\"vision_position_table_path\":\"relative\"}"] {
                #expect(try await f.request("PUT", "/control/settings", body: Data(invalid.utf8), auth: true).status == 400)
                #expect(try f.store.load() == before)
            }
        }
        try await withHTTPFixture { f in
            let response = try await f.request("PUT", "/control/settings", body: jsonData(["vision_tower_path": "/alone"]), auth: true)
            let stored = try f.store.load()
            #expect(response.status == 400)
            #expect(stored.visionTowerPath == nil)
        }
    }

    @Test func visionResizeModePersistsAndRequiresRestart() async throws {
        try await withHTTPFixture { f in
            // The default is smart; switching away from it must require a restart.
            let patch = try jsonData(["vision_resize_mode": "origin-bucket"])
            let response = try await f.request("PUT", "/control/settings", body: patch, auth: true)
            #expect(response.status == 200)
            let result = try response.json()
            guard case let .array(restartFields) = result["restart_required"] else {
                Issue.record("Expected restart_required array")
                return
            }
            #expect(restartFields.contains(.string("vision_resize_mode")))
            let stored = try f.store.load()
            #expect(stored.visionResizeMode == .originBucket)

            // Invalid value rejected with 400
            let badResponse = try await f.request("PUT", "/control/settings", body: jsonData(["vision_resize_mode": "bilinear"]), auth: true)
            #expect(badResponse.status == 400)
            #expect(try f.store.load().visionResizeMode == .originBucket) // unchanged by the rejected patch
        }
    }
}
