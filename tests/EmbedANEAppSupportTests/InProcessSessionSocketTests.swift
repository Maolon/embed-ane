import EmbedANEAppSupport
import EmbedANECore
import EmbedANETestSupport
import Foundation
import Testing

private final class AppPortCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = 0
    var value: Int { lock.withLock { stored } }
    func set(_ value: Int) { lock.withLock { stored = value } }
}
private func jsonRequest(_ path: String, port: Int, method: String = "GET", body: Data? = nil,
                         token: String? = nil) async throws -> (Data, Int) {
    let url = try #require(URL(string: "http://127.0.0.1:\(port)\(path)"))
    var request = URLRequest(url: url)
    request.httpMethod = method; request.httpBody = body; request.timeoutInterval = 10
    if body != nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
    if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
    let configuration = URLSessionConfiguration.ephemeral
    configuration.connectionProxyDictionary = [:]
    let client = URLSession(configuration: configuration)
    defer { client.invalidateAndCancel() }
    let (data, response) = try await client.data(for: request)
    return (data, try #require(response as? HTTPURLResponse).statusCode)
}

@Suite("App's in-process server over real loopback sockets", .serialized)
struct InProcessSessionSocketTests {
    @Test func appLifecycleAndHTTPUseTheSameMockRuntime() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)
        let config = ServiceConfiguration(modelRoot: temp.url.appendingPathComponent("models").path)
        let predictor = MockPredictor()
        defer { Task { await predictor.releaseAll() } }
        let session = try InProcessAppSession(configuration: config, predictor: predictor,
            preparer: SerialTokenizationPreparer(tokenizer: FixtureTokenizer()), store: store, portOverride: 0)
        let port = AppPortCapture()
        let running = Task { try await session.run { port.set($0) } }
        defer { running.cancel() }
        try await eventually { port.value > 0 }
        try await session.load()
        let payload = Data(#"{"model":"wemm-embedding-2b-ane","input":"A"}"#.utf8)
        let (data, status) = try await jsonRequest("/v1/embeddings", port: port.value, method: "POST", body: payload)
        #expect(status == 200)
        let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(json["object"] as? String == "list")
        #expect(await predictor.loadCount == 1)
        #expect(await predictor.predictCount == 1)
        let snapshot = try await session.snapshot()
        #expect(snapshot.state == .ready && snapshot.windowCount == 1)
        try await session.unload()
        #expect(try await session.snapshot().state == .unloaded)
        running.cancel(); _ = try? await running.value
    }
    @Test func externalHTTPSettingsAppearAsEffectiveAndPendingInTheApp() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)
        let session = try InProcessAppSession(configuration: ServiceConfiguration(modelRoot: temp.url.path),
            predictor: MockPredictor(), preparer: SerialTokenizationPreparer(tokenizer: FixtureTokenizer()),
            store: store, portOverride: 0)
        let port = AppPortCapture()
        let running = Task { try await session.run { port.set($0) } }
        defer { running.cancel() }
        try await eventually { port.value > 0 }
        let readData = try StateFileStorage(directory: temp.url.appendingPathComponent(".embed-ane"))
            .read("control-token", requirePrivate: true)
        let tokenData = try #require(readData)
        let token = String(decoding: tokenData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        let (_, status) = try await jsonRequest("/control/settings", port: port.value, method: "PUT",
            body: Data(#"{"port":9191,"idle_timeout_s":20,"max_queue_depth":3,"compute_units":"cpu_only"}"#.utf8), token: token)
        #expect(status == 200)
        let snapshot = try await session.snapshot()
        #expect(snapshot.effective.port == 8080 && snapshot.effective.computeUnits == .cpuAndNE)
        #expect(snapshot.effective.idleTimeoutS == 20 && snapshot.effective.maxQueueDepth == 3)
        #expect(snapshot.desired.port == 9191 && snapshot.desired.computeUnits == .cpuOnly)
        #expect(snapshot.restartRequired == ["compute_units", "port"])
        let (_, healthStatus) = try await jsonRequest("/health", port: port.value)
        #expect(healthStatus == 200)
        running.cancel(); _ = try? await running.value
    }
    @Test func appUnloadCannotInterruptAnHTTPPrediction() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        let predictor = MockPredictor()
        defer { Task { await predictor.releaseAll() } }
        let session = try InProcessAppSession(configuration: ServiceConfiguration(modelRoot: temp.url.path),
            predictor: predictor, preparer: SerialTokenizationPreparer(tokenizer: FixtureTokenizer()),
            store: ConfigurationStore(home: temp.url), portOverride: 0)
        let port = AppPortCapture()
        let running = Task { try await session.run { port.set($0) } }
        defer { running.cancel() }
        try await eventually { port.value > 0 }
        try await session.load()
        await predictor.predictGate.close()
        let request = Task { try await jsonRequest("/v1/embeddings", port: port.value, method: "POST",
            body: Data(#"{"model":"test","input":"B"}"#.utf8)) }
        try await eventually { await predictor.predictCount == 1 }
        let before = try await session.snapshot()
        #expect(before.hasWork && !before.canUnload)
        await #expect(throws: EmbedANEError.busy) { try await session.unload() }
        await predictor.predictGate.open()
        #expect(try await request.value.1 == 200)
        #expect(await predictor.unloadCount == 0)
        running.cancel(); _ = try? await running.value
    }
}
