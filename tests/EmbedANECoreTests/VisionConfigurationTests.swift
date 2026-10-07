import EmbedANECore
import Foundation
import Testing

struct VisionConfigurationTests {
    @Test func optionalPathsRoundTripAndExpandWithoutAffectingExistingConfiguration() throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: home) }
        let store = ConfigurationStore(home: home)
        var configured = ServiceConfiguration(port: 15535, visionTowerPath: "~/vision/tower.mlmodelc",
            visionExtropeChunksDirectory: "~/vision/chunks", visionPositionTablePath: "~/vision/positions.npy")
        try store.save(configured)
        #expect(try store.load() == configured)
        let expanded = try store.resolve(environment: [:])
        #expect(expanded.port == 15535)
        #expect(expanded.visionTowerPath == home.appendingPathComponent("vision/tower.mlmodelc").path)
        #expect(expanded.visionExtropeChunksDirectory == home.appendingPathComponent("vision/chunks").path)
        #expect(expanded.visionPositionTablePath == home.appendingPathComponent("vision/positions.npy").path)
        // Disabling is an explicit all-three omission; null is never a default.
        configured.visionTowerPath = nil; configured.visionExtropeChunksDirectory = nil; configured.visionPositionTablePath = nil
        try store.save(configured)
        #expect(try store.load() == configured)
        let yaml = try String(contentsOf: store.file, encoding: .utf8)
        #expect(!yaml.contains("vision_"))
    }

    @Test func rejectsPartialNullAndInvalidPaths() throws {
        for yaml in [
            "vision_tower_path: /tower",
            "vision_tower_path: /tower\nvision_extrope_chunks_dir: /chunks",
            "vision_tower_path: null\nvision_extrope_chunks_dir: /chunks\nvision_position_table_path: /positions",
            "vision_tower_path: relative\nvision_extrope_chunks_dir: /chunks\nvision_position_table_path: /positions",
            "vision_tower_path: /tower\nvision_extrope_chunks_dir: ''\nvision_position_table_path: /positions",
            "vision_unknown_path: /x"
        ] {
            #expect(throws: EmbedANEError.self) { try StrictYAML.decode(ServiceConfiguration.self, from: "config_version: 1\n" + yaml + "\n") }
        }
        let base = ServiceConfiguration()
        #expect(throws: EmbedANEError.self) { try base.applying(.init(visionTowerPath: "/tower")) }
        let configured = try base.applying(.init(visionTowerPath: "/tower", visionExtropeChunksDirectory: "/chunks", visionPositionTablePath: "/positions"))
        #expect(try configured.applying(.init(visionPositionTablePath: "/new")).visionPositionTablePath == "/new")
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(ConfigurationOverrides.self, from: Data("{\"vision_position_table_path\":null}".utf8))
        }
    }

    @Test func visionResizeModeDecodesDefaultAndExplicitValuesAndRoundTrips() throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: home) }
        let store = ConfigurationStore(home: home)

        // Default is smart (native size, as upstream)
        #expect(ServiceConfiguration().visionResizeMode == .smart)

        let decodedDefault = try StrictYAML.decode(ServiceConfiguration.self, from: "config_version: 1\n")
        #expect(decodedDefault.visionResizeMode == .smart)

        let decodedOriginBucket = try StrictYAML.decode(ServiceConfiguration.self, from: "config_version: 1\nvision_resize_mode: origin-bucket\n")
        #expect(decodedOriginBucket.visionResizeMode == .originBucket)

        let decodedSmart = try StrictYAML.decode(ServiceConfiguration.self, from: "config_version: 1\nvision_resize_mode: smart\n")
        #expect(decodedSmart.visionResizeMode == .smart)

        // Patch updates mode
        let base = ServiceConfiguration()
        let patched = try base.applying(.init(visionResizeMode: .originBucket))
        #expect(patched.visionResizeMode == .originBucket)

        // Round-trip save and load for both modes
        try store.save(decodedSmart)
        #expect(try store.load().visionResizeMode == .smart)

        try store.save(decodedOriginBucket)
        #expect(try store.load().visionResizeMode == .originBucket)
    }

    @Test func visionResizeModeRejectsInvalidValuesStrictly() throws {
        for invalidYaml in [
            "vision_resize_mode: bilinear",
            "vision_resize_mode: 123",
            "vision_resize_mode: ''",
            "vision_resize_mode: null",
            "vision_resize_mode: [smart]"
        ] {
            #expect(throws: EmbedANEError.self) {
                try StrictYAML.decode(ServiceConfiguration.self, from: "config_version: 1\n" + invalidYaml + "\n")
            }
        }
        for invalidJSON in [
            "{\"vision_resize_mode\":\"unknown\"}",
            "{\"vision_resize_mode\":null}",
            "{\"vision_resize_mode\":123}"
        ] {
            #expect(throws: (any Error).self) {
                try JSONDecoder().decode(ConfigurationOverrides.self, from: Data(invalidJSON.utf8))
            }
        }
    }
}
