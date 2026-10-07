import EmbedANECore
import Foundation
import Testing

@Suite("Configuration")
struct ConfigurationTests {
    private func temporaryStore() -> ConfigurationStore {
        ConfigurationStore(home: FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-config-\(UUID())"))
    }
    @Test func defaultsAndEnvironmentAndFlagsHaveSpecifiedPrecedence() throws {
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.home) }
        let defaults = try store.resolve(environment: [:])
        #expect(defaults.port == 8080 && defaults.idleTimeoutS == 0)
        #expect(defaults.maxQueueDepth == 16 && defaults.maxBatch == 8)
        #expect(defaults.computeUnits == .cpuAndNE)
        #expect(defaults.modelRoot == store.home.appendingPathComponent(".embed-ane/models").path)
        var file = ServiceConfiguration(port: 9001, modelRoot: "/from-file")
        file.maxQueueDepth = 7
        try store.save(file)
        #expect(try store.resolve(environment: [:]).modelRoot == "/from-file")
        let env = try store.resolve(environment: ["EMBED_ANE_MODEL_ROOT": "/from-env"])
        #expect(env.modelRoot == "/from-env" && env.port == 9001 && env.maxQueueDepth == 7)
        let cli = try store.resolve(cli: .init(port: 9002, modelRoot: "/from-cli"),
                                    environment: ["EMBED_ANE_MODEL_ROOT": "/from-env"])
        #expect(cli.modelRoot == "/from-cli" && cli.port == 9002)
        #expect(try store.resolve(cli: .init(modelRoot: "/valid"), environment: ["EMBED_ANE_MODEL_ROOT": ""]).modelRoot == "/valid")
        #expect(throws: EmbedANEError.self) { try store.resolve(environment: ["EMBED_ANE_MODEL_ROOT": ""]) }
    }

    @Test func strictVersionKeysNullDuplicatesAndRanges() throws {
        let invalid = [
            "port: 8080\n", "config_version: 2\n", "config_version: 1\nunknown: true\n",
            "config_version: 1\nport: null\n", "config_version: 1\nport: 1\nport: 2\n",
            "config_version: 1\nport: 0\n", "config_version: 1\nport: 65536\n",
            "config_version: 1\nmodel_id: ../outside\n", "config_version: 1\nmodel_root: relative\n",
            "config_version: 1\nidle_timeout_s: -1\n", "config_version: 1\nidle_timeout_s: .inf\n",
            "config_version: 1\nmax_batch: 9\n", "config_version: 1\nmax_batch: 0\n",
            "config_version: 1\nmax_queue_depth: 0\n", "config_version: 1\ncompute_units: gpu_magic\n",
            "config_version: 1\nauto_load: not_a_boolean\n"
        ]
        for text in invalid {
            #expect(throws: EmbedANEError.self) { try StrictYAML.decode(ServiceConfiguration.self, from: text) }
        }
        let minimal = try StrictYAML.decode(ServiceConfiguration.self, from: "config_version: 1\n")
        #expect(minimal == ServiceConfiguration())
        #expect(minimal.autoLoad == true)
        let disabled = try StrictYAML.decode(ServiceConfiguration.self, from: "config_version: 1\nauto_load: false\n")
        #expect(disabled.autoLoad == false)
        let enabled = try StrictYAML.decode(ServiceConfiguration.self, from: "config_version: 1\nauto_load: true\n")
        #expect(enabled.autoLoad == true)
    }

    @Test func atomicRoundTripAndCreateOnlyPersistence() throws {
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.home) }
        let first = ServiceConfiguration(port: 9080)
        try store.saveIfAbsent(first)
        try store.saveIfAbsent(ServiceConfiguration(port: 9999))
        #expect(try store.load() == first)
        var changed = first; changed.idleTimeoutS = 1.25; changed.maxQueueDepth = 3
        try store.save(changed)
        #expect(try store.load() == changed)
        let entries = try FileManager.default.contentsOfDirectory(atPath: store.file.deletingLastPathComponent().path)
        #expect(entries == ["config.yaml"])
        let attributes = try FileManager.default.attributesOfItem(atPath: store.file.path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    }

    @Test func readersNeverObservePartialConfigDuringAtomicReplacement() async throws {
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.home) }
        let a = ServiceConfiguration(port: 9001), b = ServiceConfiguration(port: 9002)
        try store.save(a)
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { for _ in 0..<30 { try store.save(a) } }
            group.addTask { for _ in 0..<30 { try store.save(b) } }
            group.addTask {
                for _ in 0..<60 {
                    let value = try store.load()
                    #expect(value == a || value == b)
                }
            }
            try await group.waitForAll()
        }
    }

    @Test func symlinkStateFileIsRejectedWithoutTouchingTarget() throws {
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.home) }
        try FileManager.default.createDirectory(at: store.file.deletingLastPathComponent(), withIntermediateDirectories: true)
        let target = store.home.appendingPathComponent("outside")
        try Data("sentinel".utf8).write(to: target)
        try FileManager.default.createSymbolicLink(at: store.file, withDestinationURL: target)
        #expect(throws: EmbedANEError.self) { try store.load() }
        #expect(throws: EmbedANEError.self) { try store.save(ServiceConfiguration()) }
        #expect(try Data(contentsOf: target) == Data("sentinel".utf8))
    }

    @Test func malformedExistingConfigDoesNotSilentlyFallBack() throws {
        let store = temporaryStore()
        defer { try? FileManager.default.removeItem(at: store.home) }
        try StateFileStorage(directory: store.file.deletingLastPathComponent())
            .write("config.yaml", data: Data("config_version: 2\n".utf8))
        #expect(throws: EmbedANEError.self) { try store.resolve(environment: [:]) }
        #expect(try String(contentsOf: store.file, encoding: .utf8) == "config_version: 2\n")
    }
}
