import EmbedANECore
import Foundation
import Testing

@Suite("CoreAI configuration parsing and settings")
struct CoreAIConfigurationTests {
    @Test func defaultBackendIsCoreML() throws {
        let config = ServiceConfiguration()
        #expect(config.engineBackend == .coreml)

        let minimal = try StrictYAML.decode(ServiceConfiguration.self, from: "config_version: 1\n")
        #expect(minimal.engineBackend == .coreml)
    }

    @Test func decodesCoreAIBackendExplicitly() throws {
        let yaml = "config_version: 1\nengine_backend: coreai\n"
        let config = try StrictYAML.decode(ServiceConfiguration.self, from: yaml)
        #expect(config.engineBackend == .coreai)
    }

    @Test func decodesCoreMLBackendExplicitly() throws {
        let yaml = "config_version: 1\nengine_backend: coreml\n"
        let config = try StrictYAML.decode(ServiceConfiguration.self, from: yaml)
        #expect(config.engineBackend == .coreml)
    }

    @Test func rejectsInvalidEngineBackend() throws {
        let invalid = "config_version: 1\nengine_backend: tensorrt\n"
        #expect(throws: EmbedANEError.self) {
            try StrictYAML.decode(ServiceConfiguration.self, from: invalid)
        }
    }

    @Test func overridesApplyEngineBackend() throws {
        let base = ServiceConfiguration()
        #expect(base.engineBackend == .coreml)

        let patch = ConfigurationOverrides(engineBackend: .coreai)
        let applied = try base.applying(patch)
        #expect(applied.engineBackend == .coreai)
    }
}
