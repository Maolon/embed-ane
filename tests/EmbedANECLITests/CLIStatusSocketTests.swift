import EmbedANECore
import EmbedANEHTTP
import EmbedANETestSupport
import Foundation
import Testing
@testable import embed_ane

private actor BoundPort {
    var value: Int?
    func set(_ value: Int) { self.value = value }
}

struct CLIStatusSocketTests {
    @Test func statusReadsConfiguredPortOverRealLoopbackSocket() async throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: home) }
        let store = ConfigurationStore(home: home)
        let configuration = ServiceConfiguration(modelRoot: home.appendingPathComponent("models").path)
        let predictor = MockPredictor()
        let server = try EmbeddingHTTPServer(configuration: configuration, predictor: predictor,
            preparer: CountingPreparer(), store: store, portOverride: 0)
        _ = try await server.lifecycle.load()
        let port = BoundPort()
        let task = Task { try await server.run(onServerRunning: { await port.set($0) }) }
        do {
            try await eventually { await port.value != nil }
            let actualPort = try #require(await port.value)
            var stored = configuration; stored.port = actualPort
            try store.save(stored)
            let result = await CLIApplication.run(arguments: ["status"], store: store,
                workingDirectory: home, environment: [:], executor: ProductionCommands())
            #expect(result.exitCode == 0)
            let body = try #require(JSONSerialization.jsonObject(with: result.stdout) as? [String: Any])
            #expect(body["status"] as? String == "ok")
            #expect((body["model"] as? [String: Any])?["state"] as? String == "ready")
            #expect(await predictor.predictCount == 0)
        } catch {
            task.cancel(); _ = try? await task.value
            _ = try? await server.lifecycle.unload()
            throw error
        }
        task.cancel(); _ = try? await task.value
        _ = try await server.lifecycle.unload()
    }
}
