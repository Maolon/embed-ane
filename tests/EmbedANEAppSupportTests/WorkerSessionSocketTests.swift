import Darwin
import EmbedANEAppSupport
import EmbedANECore
import EmbedANEHTTP
import EmbedANETestSupport
import Foundation
import Testing

private final class AppPortCapture: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = 0
    var value: Int { lock.withLock { stored } }
    func set(_ value: Int) { lock.withLock { stored = value } }
}

private func httpReq(
    _ path: String,
    port: Int,
    method: String = "GET",
    body: Data? = nil,
    token: String? = nil,
    headers: [String: String] = [:]
) async throws -> (data: Data, status: Int, headers: [String: String]) {
    let url = try #require(URL(string: "http://127.0.0.1:\(port)\(path)"))
    var request = URLRequest(url: url)
    request.httpMethod = method
    request.httpBody = body
    request.timeoutInterval = 10
    if body != nil && headers["Content-Type"] == nil {
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    }
    if let token {
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }
    for (k, v) in headers {
        request.setValue(v, forHTTPHeaderField: k)
    }
    let configuration = URLSessionConfiguration.ephemeral
    configuration.connectionProxyDictionary = [:]
    let client = URLSession(configuration: configuration)
    defer { client.invalidateAndCancel() }
    let (data, response) = try await client.data(for: request)
    let http = try #require(response as? HTTPURLResponse)
    var respHeaders: [String: String] = [:]
    for (k, v) in http.allHeaderFields {
        if let ks = k as? String, let vs = v as? String {
            respHeaders[ks] = vs
        }
    }
    return (data, http.statusCode, respHeaders)
}

@Suite("Worker session and supervisor proxy over loopback sockets", .serialized)
struct WorkerSessionSocketTests {

    @Test func workerModeArgumentParsingAndForcedIdleZero() throws {
        // 1. Worker requested detection
        let withFlag = AppWorkerRunner.isWorkerRequested(arguments: ["Embed ANE", "--worker"], environment: [:])
        #expect(withFlag == true)

        let withEnv = AppWorkerRunner.isWorkerRequested(arguments: ["Embed ANE"], environment: ["EMBED_ANE_WORKER": "1"])
        #expect(withEnv == true)

        let withoutWorker = AppWorkerRunner.isWorkerRequested(arguments: ["Embed ANE"], environment: [:])
        #expect(withoutWorker == false)

        // 2. Port parsing
        let internalPortArg = AppWorkerRunner.parseInternalPort(
            arguments: ["Embed ANE", "--worker", "--internal-port", "16001"],
            environment: [:]
        )
        #expect(internalPortArg == 16001)

        let portArg = AppWorkerRunner.parseInternalPort(
            arguments: ["Embed ANE", "--worker", "--port", "16002"],
            environment: [:]
        )
        #expect(portArg == 16002)

        let envPort = AppWorkerRunner.parseInternalPort(
            arguments: ["Embed ANE", "--worker"],
            environment: ["EMBED_ANE_INTERNAL_PORT": "16003"]
        )
        #expect(envPort == 16003)

        let defaultPort = AppWorkerRunner.parseInternalPort(
            arguments: ["Embed ANE", "--worker"],
            environment: [:],
            defaultPort: 15536
        )
        #expect(defaultPort == 15536)

        // 3. Forced idle_timeout_s: 0
        let temp = try AppTemporaryDirectory()
        defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)
        var customConfig = ServiceConfiguration()
        customConfig.port = 15535
        customConfig.idleTimeoutS = 300 // Config specifies 300s
        try store.save(customConfig)

        let workerConfig = try AppWorkerRunner.prepareConfiguration(store: store, internalPort: 16000)
        #expect(workerConfig.port == 16000)
        #expect(workerConfig.idleTimeoutS == 0) // Forced to 0
    }

    @Test func supervisorStateMachineWithFakeWorker() async throws {
        let temp = try AppTemporaryDirectory()
        defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)
        var config = ServiceConfiguration()
        config.idleTimeoutS = 10
        try store.save(config)

        let controller = MockWorkerProcessController()
        let session = WorkerSession(
            configuration: config,
            store: store,
            internalPort: 16000,
            workerController: controller,
            portOverride: 0,
            customHealthPoller: { _ in true }
        )

        // Initial state
        let initialState = await session.getWorkerState()
        #expect(initialState == .down)
        let initialSnap = try await session.snapshot()
        #expect(initialSnap.state == .unloaded)

        // Load
        try await session.load()
        let spawnedCount = controller.spawnCount
        #expect(spawnedCount == 1)
        let readyState = await session.getWorkerState()
        #expect(readyState == .ready)
        let readySnap = try await session.snapshot()
        #expect(readySnap.state == .ready)

        // Idle eviction check
        await session.checkIdleEviction()
        // Last activity was just now, should not evict yet
        let stillReady = await session.getWorkerState()
        #expect(stillReady == .ready)

        // Unload
        try await session.unload()
        let termCount = controller.terminateCount
        #expect(termCount == 1)
        let unloadedState = await session.getWorkerState()
        #expect(unloadedState == .down)
        let unloadedSnap = try await session.snapshot()
        #expect(unloadedSnap.state == .unloaded)

        // Crash and respawn when keep-loaded (idleTimeoutS == 0)
        var keepLoadedConfig = config
        keepLoadedConfig.idleTimeoutS = 0
        try await session.apply(ConfigurationOverrides(idleTimeoutS: 0))
        try await session.load()
        #expect(await session.getWorkerState() == .ready)

        // Simulate crash
        controller.simulateCrash(exitCode: 1)
        // Auto-respawn should trigger
        try await eventually {
            if controller.spawnCount >= 3 {
                return await session.getWorkerState() == .ready
            }
            return false
        }
        let totalSpawns = controller.spawnCount
        #expect(totalSpawns >= 3)
    }

    @Test func proxyRoutingAgainstLoopbackStubWorker() async throws {
        let temp = try AppTemporaryDirectory()
        defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)
        let workerConfig = ServiceConfiguration(modelRoot: temp.url.appendingPathComponent("models").path)

        // Start real loopback EmbeddingHTTPServer as stub worker
        let predictor = MockPredictor()
        defer { Task { await predictor.releaseAll() } }
        let workerServer = try EmbeddingHTTPServer(
            configuration: workerConfig,
            predictor: predictor,
            preparer: CountingPreparer(),
            store: store,
            portOverride: 0
        )
        _ = try await workerServer.lifecycle.load()
        let workerPort = AppPortCapture()
        let workerTask = Task { try await workerServer.run { workerPort.set($0) } }
        defer { workerTask.cancel() }
        try await eventually { workerPort.value > 0 }

        // Start supervisor WorkerSession pointing to workerPort
        let controller = MockWorkerProcessController(isRunning: true)
        let supervisorSession = WorkerSession(
            configuration: workerConfig,
            store: store,
            internalPort: workerPort.value,
            workerController: controller,
            portOverride: 0,
            customHealthPoller: { _ in true }
        )
        let supervisorPort = AppPortCapture()
        let supervisorTask = Task { try await supervisorSession.run { supervisorPort.set($0) } }
        defer { supervisorTask.cancel() }
        try await eventually { supervisorPort.value > 0 }

        // 1. Worker is UP
        try await supervisorSession.load()

        // Inference /v1/embeddings -> forwards to worker, 200
        let payload = Data(#"{"model":"wemm-embedding-2b","input":"test"}"#.utf8)
        let (embData, embStatus, _) = try await httpReq("/v1/embeddings", port: supervisorPort.value, method: "POST", body: payload)
        #expect(embStatus == 200)
        let embJson = try #require(JSONSerialization.jsonObject(with: embData) as? [String: Any])
        #expect(embJson["object"] as? String == "list")
        #expect(await predictor.predictCount == 1)

        // Error fidelity: invalid dimensions -> 400
        let badDimPayload = Data(#"{"model":"wemm-embedding-2b","input":"test","dimensions":9999}"#.utf8)
        let (errData, errStatus, _) = try await httpReq("/v1/embeddings", port: supervisorPort.value, method: "POST", body: badDimPayload)
        #expect(errStatus == 400)
        let errJson = try #require(JSONSerialization.jsonObject(with: errData) as? [String: Any])
        let errObj = try #require(errJson["error"] as? [String: Any])
        #expect(errObj["code"] as? String == "unsupported_dimension")

        // /health -> forwards to worker
        let (healthData, healthStatus, _) = try await httpReq("/health", port: supervisorPort.value)
        #expect(healthStatus == 200)
        let healthJson = try #require(JSONSerialization.jsonObject(with: healthData) as? [String: Any])
        #expect(healthJson["status"] as? String == "ok")

        // /v1/models -> forwards to worker
        let (modelsData, modelsStatus, _) = try await httpReq("/v1/models", port: supervisorPort.value)
        #expect(modelsStatus == 200)
        let modelsJson = try #require(JSONSerialization.jsonObject(with: modelsData) as? [String: Any])
        #expect(modelsJson["object"] as? String == "list")

        // 2. Worker is DOWN
        try await supervisorSession.unload()

        // /health when DOWN -> supervisor returns 200 JSON with worker: "down"
        let (downHealthData, downHealthStatus, _) = try await httpReq("/health", port: supervisorPort.value)
        #expect(downHealthStatus == 200)
        let downHealthJson = try #require(JSONSerialization.jsonObject(with: downHealthData) as? [String: Any])
        #expect(downHealthJson["status"] as? String == "ok")
        #expect(downHealthJson["worker"] as? String == "down")

        // /v1/models when DOWN -> supervisor returns 200 JSON with models
        let (downModelsData, downModelsStatus, _) = try await httpReq("/v1/models", port: supervisorPort.value)
        #expect(downModelsStatus == 200)
        let downModelsJson = try #require(JSONSerialization.jsonObject(with: downModelsData) as? [String: Any])
        #expect(downModelsJson["object"] as? String == "list")

        // Inference /v1/embeddings when DOWN -> returns 503 Retry-After: 20 and triggers spawn
        let (downEmbData, downEmbStatus, downEmbHeaders) = try await httpReq("/v1/embeddings", port: supervisorPort.value, method: "POST", body: payload)
        #expect(downEmbStatus == 503)
        #expect(downEmbHeaders["Retry-After"] == "20" || downEmbHeaders["retry-after"] == "20")
        let downEmbJson = try #require(JSONSerialization.jsonObject(with: downEmbData) as? [String: Any])
        let downErrObj = try #require(downEmbJson["error"] as? [String: Any])
        #expect(downErrObj["code"] as? String == "overloaded")
        #expect(downErrObj["type"] as? String == "overloaded_error")

        // Wait for spawn to complete
        try await eventually {
            await supervisorSession.getWorkerState() == .ready
        }

        supervisorTask.cancel(); _ = try? await supervisorTask.value
        workerTask.cancel(); _ = try? await workerTask.value
    }

    @Test func controlLoadAndUnloadInterception() async throws {
        let temp = try AppTemporaryDirectory()
        defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)
        let config = ServiceConfiguration(modelRoot: temp.url.path)

        let controller = MockWorkerProcessController(isRunning: false)
        let session = WorkerSession(
            configuration: config,
            store: store,
            internalPort: 16050,
            workerController: controller,
            portOverride: 0,
            customHealthPoller: { _ in true }
        )
        let port = AppPortCapture()
        let task = Task { try await session.run { port.set($0) } }
        defer { task.cancel() }
        try await eventually { port.value > 0 }

        let tokenData = try #require(try StateFileStorage(directory: temp.url.appendingPathComponent(".embed-ane"))
            .read("control-token", requirePrivate: true))
        let token = String(decoding: tokenData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)

        // /control/load without token -> 401
        let (_, unauthStatus, _) = try await httpReq("/control/load", port: port.value, method: "POST")
        #expect(unauthStatus == 401)

        // /control/load with token -> 200 LoadResponse
        let (loadData, loadStatus, _) = try await httpReq("/control/load", port: port.value, method: "POST", token: token)
        #expect(loadStatus == 200)
        let loadJson = try #require(JSONSerialization.jsonObject(with: loadData) as? [String: Any])
        #expect(loadJson["state"] as? String == "ready")
        #expect(controller.spawnCount >= 1)

        // /control/unload with token -> 200 UnloadResponse
        let (unloadData, unloadStatus, _) = try await httpReq("/control/unload", port: port.value, method: "POST", token: token)
        #expect(unloadStatus == 200)
        let unloadJson = try #require(JSONSerialization.jsonObject(with: unloadData) as? [String: Any])
        #expect(unloadJson["state"] as? String == "unloaded")
        #expect(controller.terminateCount >= 1)

        task.cancel(); _ = try? await task.value
    }

    @Test func inProcessSessionStillSelectable() async throws {
        let temp = try AppTemporaryDirectory()
        defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)
        let config = ServiceConfiguration(modelRoot: temp.url.path)
        let session = try InProcessAppSession(
            configuration: config,
            predictor: MockPredictor(),
            preparer: CountingPreparer(),
            store: store,
            portOverride: 0
        )
        let snapshot = try await session.snapshot()
        #expect(snapshot.state == .unloaded)
    }

    @Test func listenerUpBeforeLoadAnswersHealthWithStateLoading() async throws {
        let temp = try AppTemporaryDirectory()
        defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)

        let predictor = MockPredictor()
        defer { Task { await predictor.releaseAll() } }
        await predictor.loadGate.close() // Keep load() paused

        let components = ServingRuntimeComponents(
            predictor: predictor,
            preparer: CountingPreparer(),
            isMultimodal: false
        )

        let internalPort = findFreePort()
        let (server, loadTask) = try await AppWorkerRunner.createAndStartWorker(
            arguments: ["Embed ANE", "--worker", "--internal-port", "\(internalPort)"],
            store: store,
            makeRuntime: { _ in components }
        )

        let workerPort = AppPortCapture()
        let serverTask = Task {
            try await server.run { workerPort.set($0) }
        }
        defer {
            Task { await predictor.loadGate.open() }
            serverTask.cancel()
        }

        try await eventually { workerPort.value > 0 }

        // While load() is still blocked at loadGate, the HTTP listener is ALREADY UP!
        let (loadingData, loadingStatus, _) = try await httpReq("/health", port: workerPort.value)
        #expect(loadingStatus == 200)
        let loadingJson = try #require(JSONSerialization.jsonObject(with: loadingData) as? [String: Any])
        #expect(loadingJson["status"] as? String == "ok")
        let modelLoadingObj = try #require(loadingJson["model"] as? [String: Any])
        #expect(modelLoadingObj["state"] as? String == "loading")

        // Unblock load()
        await predictor.loadGate.open()
        _ = try await loadTask.value

        // Health transitions to ready
        try await eventually {
            if let (readyData, readyStatus, _) = try? await httpReq("/health", port: workerPort.value),
               readyStatus == 200,
               let readyJson = try? JSONSerialization.jsonObject(with: readyData) as? [String: Any],
               let readyModel = readyJson["model"] as? [String: Any],
               readyModel["state"] as? String == "ready" {
                return true
            }
            return false
        }
    }

    @Test func supervisorRespawnTerminatesPreviousWorkerExactlyOnce() async throws {
        let temp = try AppTemporaryDirectory()
        defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)
        var config = ServiceConfiguration()
        config.idleTimeoutS = 0
        try store.save(config)

        let controller = MockWorkerProcessController(isRunning: false)
        let session = WorkerSession(
            configuration: config,
            store: store,
            internalPort: 16099,
            workerController: controller,
            portOverride: 0,
            customHealthPoller: { _ in false } // Fail readiness probes
        )
        await session.setReadinessTiming(timeoutSeconds: 1, pollIntervalMs: 50)

        // Expect load to fail after bounded retry
        await #expect(throws: EmbedANEError.self) {
            try await session.load()
        }

        // Verify ordering: spawn -> terminate -> spawn -> terminate
        #expect(controller.spawnCount == 2)
        #expect(controller.terminateCount == 2)
        #expect(controller.events == ["spawn", "terminate", "spawn", "terminate"])
        #expect(controller.isRunning == false)
    }

    @Test func loadFailureInsideWorkerDoesNotExitProcessAndSurfacesErrorViaStats() async throws {
        let temp = try AppTemporaryDirectory()
        defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)

        final class FailingLoadPredictor: EmbeddingPredictor, EmbeddingPreparer, @unchecked Sendable {
            func load() async throws -> LoadReport {
                throw EmbedANEError.abiMismatch(chunk: 0, reason: "Corrupted model weights")
            }
            func unload() async throws -> UnloadReport { UnloadReport(residentBytes: 0) }
            func predict(_ request: PredictRequest) async throws -> PredictResult { PredictResult(embeddings: []) }
            func prepare(_ texts: [String]) async throws -> PredictRequest { try PredictRequest(inputs: []) }
        }

        let predictor = FailingLoadPredictor()
        let components = ServingRuntimeComponents(
            predictor: predictor,
            preparer: predictor,
            isMultimodal: false
        )

        let internalPort = findFreePort()
        let (server, loadTask) = try await AppWorkerRunner.createAndStartWorker(
            arguments: ["Embed ANE", "--worker", "--internal-port", "\(internalPort)"],
            store: store,
            makeRuntime: { _ in components }
        )

        let workerPort = AppPortCapture()
        let serverTask = Task {
            try await server.run { workerPort.set($0) }
        }
        defer { serverTask.cancel() }
        try await eventually { workerPort.value > 0 }

        // Wait for load task to fail
        await #expect(throws: EmbedANEError.self) {
            try await loadTask.value
        }

        // Worker server task is STILL ALIVE and responding!
        #expect(!serverTask.isCancelled)

        // Health endpoint responds 200 with model state failed
        let (healthData, healthStatus, _) = try await httpReq("/health", port: workerPort.value)
        #expect(healthStatus == 200)
        let healthJson = try #require(JSONSerialization.jsonObject(with: healthData) as? [String: Any])
        #expect(healthJson["status"] as? String == "ok")
        let modelObj = try #require(healthJson["model"] as? [String: Any])
        #expect(modelObj["state"] as? String == "failed")

        // Stats endpoint surfaces error via stats
        let tokenData = try #require(try StateFileStorage(directory: temp.url.appendingPathComponent(".embed-ane"))
            .read("control-token", requirePrivate: true))
        let token = String(decoding: tokenData, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)

        let (statsData, statsStatus, _) = try await httpReq("/control/stats", port: workerPort.value, token: token)
        #expect(statsStatus == 200)
        let statsJson = try #require(JSONSerialization.jsonObject(with: statsData) as? [String: Any])
        let runtimeObj = try #require(statsJson["runtime"] as? [String: Any])
        #expect(runtimeObj["state"] as? String == "failed")
        #expect(statsJson["last_error"] as? String == "abi_mismatch")
    }

    @Test func readinessUnboundedWhileLoading() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)
        var config = ServiceConfiguration()
        config.idleTimeoutS = 0
        try store.save(config)

        let controller = MockWorkerProcessController(isRunning: false)
        let session = WorkerSession(
            configuration: config,
            store: store,
            internalPort: 16102,
            workerController: controller,
            portOverride: 0
        )
        // Previous timeout bound was 1 second; fake slow load stays in .loading for 5x that bound (5.0s)
        await session.setReadinessTiming(
            timeoutSeconds: 1,
            pollIntervalMs: 25,
            maxLoadingWaitSeconds: nil,
            progressLogIntervalSeconds: 0.1
        )

        final class PollerState: @unchecked Sendable {
            private let lock = NSLock()
            private var pollCount = 0
            func next() -> WorkerSession.WorkerHealthStatus {
                lock.withLock {
                    pollCount += 1
                    // 200 polls * 25ms = 5.0 seconds = 5x the 1.0s timeout bound
                    if pollCount <= 200 {
                        return .loading
                    } else {
                        return .ready
                    }
                }
            }
        }
        let poller = PollerState()
        await session.setCustomHealthStatusPoller { _ in poller.next() }

        try await session.load()

        #expect(controller.spawnCount == 1)
        #expect(controller.terminateCount == 0)
        #expect(controller.events == ["spawn"])
        #expect(controller.isRunning == true)
        #expect(await session.getWorkerState() == .ready)

        // Verify app.log received worker_probe_waiting events
        let appLogFile = temp.url.appendingPathComponent(".embed-ane/logs/app.log")
        let logData = try #require(try? Data(contentsOf: appLogFile))
        let logText = String(decoding: logData, as: UTF8.self)
        #expect(logText.contains("worker_probe_waiting"))
    }

    @Test func bindRetryOnAddressInUseSucceedsOncePortFreed() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)

        final class DummyPredictor: EmbeddingPredictor, EmbeddingPreparer, @unchecked Sendable {
            func load() async throws -> LoadReport { LoadReport(perChunkNS: Array(repeating: 0, count: 6), residentBytes: 0, computePlanChecked: true) }
            func unload() async throws -> UnloadReport { UnloadReport(residentBytes: 0) }
            func predict(_ request: PredictRequest) async throws -> PredictResult { PredictResult(embeddings: []) }
            func prepare(_ texts: [String]) async throws -> PredictRequest { try PredictRequest(inputs: []) }
        }
        let predictor = DummyPredictor()
        let components = ServingRuntimeComponents(predictor: predictor, preparer: predictor, isMultimodal: false)

        let heldPort = findFreePort()

        // Create a native Darwin socket holding heldPort
        let sock = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        #expect(sock >= 0)
        var yes: Int32 = 1
        _ = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(heldPort).bigEndian
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        var bindAddr = addr
        let bindRes = withUnsafePointer(to: &bindAddr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                Darwin.bind(sock, sa, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        #expect(bindRes == 0)
        let listenRes = Darwin.listen(sock, 5)
        #expect(listenRes == 0)

        let (server, _) = try await AppWorkerRunner.createAndStartWorker(
            arguments: ["Embed ANE", "--worker", "--internal-port", "\(heldPort)"],
            store: store,
            makeRuntime: { _ in components }
        )

        let listeningCaptured = AppPortCapture()
        let serverTask = Task {
            try await WorkerSignalServer.run(
                server,
                maxBindRetrySeconds: 5.0,
                bindRetryIntervalSeconds: 0.1,
                home: store.home,
                onListening: { port in
                    listeningCaptured.set(port)
                }
            )
        }
        defer { serverTask.cancel() }

        // Give the worker time to hit EADDRINUSE and enter retry loop
        try await Task.sleep(for: .milliseconds(150))
        #expect(listeningCaptured.value == 0)

        // Free the held port
        _ = Darwin.close(sock)

        // Worker should retry, bind heldPort, and call onListening
        try await eventually { listeningCaptured.value == heldPort }
        #expect(listeningCaptured.value == heldPort)

        // Verify /health endpoint responds on the bound port
        let (healthData, healthStatus, _) = try await httpReq("/health", port: heldPort)
        #expect(healthStatus == 200)
        let healthJson = try #require(JSONSerialization.jsonObject(with: healthData) as? [String: Any])
        #expect(healthJson["status"] as? String == "ok")

        // Verify worker.log recorded bind_retry event
        let workerLogFile = temp.url.appendingPathComponent(".embed-ane/logs/worker.log")
        let logData = try #require(try? Data(contentsOf: workerLogFile))
        let logText = String(decoding: logData, as: UTF8.self)
        #expect(logText.contains("bind_retry"))
    }

    @Test func slowLoadLoggingNoTerminateOrdering() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)

        final class SlowPredictor: EmbeddingPredictor, EmbeddingPreparer, @unchecked Sendable {
            func load() async throws -> LoadReport {
                try await Task.sleep(for: .milliseconds(300))
                return LoadReport(perChunkNS: Array(repeating: 0, count: 6), residentBytes: 0, computePlanChecked: true)
            }
            func unload() async throws -> UnloadReport { UnloadReport(residentBytes: 0) }
            func predict(_ request: PredictRequest) async throws -> PredictResult { PredictResult(embeddings: []) }
            func prepare(_ texts: [String]) async throws -> PredictRequest { try PredictRequest(inputs: []) }
        }
        let predictor = SlowPredictor()
        let components = ServingRuntimeComponents(predictor: predictor, preparer: predictor, isMultimodal: false)

        let internalPort = findFreePort()
        // Slow load threshold is 50ms, load takes 300ms -> triggers slow load warning!
        let (server, loadTask) = try await AppWorkerRunner.createAndStartWorker(
            arguments: ["Embed ANE", "--worker", "--internal-port", "\(internalPort)"],
            store: store,
            makeRuntime: { _ in components },
            slowLoadThresholdSeconds: 0.05
        )

        let workerPort = AppPortCapture()
        let serverTask = Task {
            try await WorkerSignalServer.run(
                server,
                maxBindRetrySeconds: 5.0,
                bindRetryIntervalSeconds: 0.1,
                home: store.home,
                onListening: { workerPort.set($0) }
            )
        }
        defer { serverTask.cancel() }
        try await eventually { workerPort.value > 0 }

        var config = ServiceConfiguration()
        config.idleTimeoutS = 0
        try store.save(config)

        let controller = MockWorkerProcessController(isRunning: false)
        let session = WorkerSession(
            configuration: config,
            store: store,
            internalPort: workerPort.value,
            workerController: controller,
            portOverride: 0
        )
        await session.setReadinessTiming(
            timeoutSeconds: 1,
            pollIntervalMs: 20,
            maxLoadingWaitSeconds: nil,
            progressLogIntervalSeconds: 0.05
        )

        // Supervisor loads worker: waits through the slow load
        _ = try await session.loadWorker()

        // Wait for load task to complete
        _ = try await loadTask.value

        // Verify supervisor readiness
        let isReady = await session.checkWorkerHealth()
        #expect(isReady == true)

        // Verify worker was NEVER terminated
        #expect(controller.terminateCount == 0)

        // Verify worker.log contains the operational guidance line
        let workerLogFile = temp.url.appendingPathComponent(".embed-ane/logs/worker.log")
        let workerLogData = try #require(try? Data(contentsOf: workerLogFile))
        let workerLogText = String(decoding: workerLogData, as: UTF8.self)
        #expect(workerLogText.contains("slow load: possible E5 respecialization; do not kill mid-compile"))
        #expect(workerLogText.contains("load_completed"))

        // Verify ordering: catalog_checked -> slow load warning -> load_completed
        let catalogRange = try #require(workerLogText.range(of: "catalog_checked"))
        let slowLoadRange = try #require(workerLogText.range(of: "slow load: possible E5 respecialization; do not kill mid-compile"))
        let loadCompletedRange = try #require(workerLogText.range(of: "load_completed"))

        #expect(catalogRange.lowerBound < slowLoadRange.lowerBound)
        #expect(slowLoadRange.lowerBound < loadCompletedRange.lowerBound)

        // Verify app.log contains worker_probe_waiting
        let appLogFile = temp.url.appendingPathComponent(".embed-ane/logs/app.log")
        let appLogData = try #require(try? Data(contentsOf: appLogFile))
        let appLogText = String(decoding: appLogData, as: UTF8.self)
        #expect(appLogText.contains("worker_probe_waiting"))
    }
}
