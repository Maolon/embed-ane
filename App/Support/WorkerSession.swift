import Darwin
import EmbedANECore
import EmbedANEHTTP
import Foundation

public enum WorkerSessionState: String, Sendable {
    case down
    case starting
    case ready
    case terminating
    case failed
}

public protocol WorkerProcessControlling: AnyObject, Sendable {
    func spawn(port: Int, configuration: ServiceConfiguration) async throws
    func terminate() async
    var isRunning: Bool { get }
    var exitHandler: (@Sendable (Int32) -> Void)? { get set }
}

public final class DefaultWorkerProcessController: WorkerProcessControlling, @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var activeTerminationTask: Task<Void, Never>?
    private let customExecutableURL: URL?
    private let store: ConfigurationStore
    public var exitHandler: (@Sendable (Int32) -> Void)?

    public init(customExecutableURL: URL? = nil, store: ConfigurationStore = .init()) {
        self.customExecutableURL = customExecutableURL
        self.store = store
    }

    public var isRunning: Bool {
        lock.withLock { process?.isRunning ?? false }
    }

    public func spawn(port: Int, configuration: ServiceConfiguration) async throws {
        // Idempotent and clean: ensure any existing process is fully terminated before spawning
        await terminate()

        try lock.withLock {
            let execURL: URL
            if let custom = customExecutableURL {
                execURL = custom
            } else if let bundleExec = Bundle.main.executableURL {
                execURL = bundleExec
            } else {
                execURL = URL(fileURLWithPath: CommandLine.arguments[0])
            }
            let p = Process()
            p.executableURL = execURL
            p.arguments = ["--worker", "--internal-port", "\(port)"]
            var env = ProcessInfo.processInfo.environment
            env["EMBED_ANE_WORKER"] = "1"
            env["EMBED_ANE_INTERNAL_PORT"] = "\(port)"
            p.environment = env

            let logsDir = store.home.appendingPathComponent(".embed-ane/logs")
            try? FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
            let logFile = logsDir.appendingPathComponent("worker.log")
            if !FileManager.default.fileExists(atPath: logFile.path) {
                FileManager.default.createFile(atPath: logFile.path, contents: nil)
            }
            if let handle = try? FileHandle(forWritingTo: logFile) {
                handle.seekToEndOfFile()
                p.standardOutput = handle
                p.standardError = handle
            }

            p.terminationHandler = { [weak self] proc in
                let status = proc.terminationStatus
                self?.lock.withLock {
                    if self?.process === proc {
                        self?.process = nil
                    }
                }
                self?.exitHandler?(status)
            }

            try p.run()
            self.process = p
        }
    }

    public func terminate() async {
        let (procToTerminate, existingTask): (Process?, Task<Void, Never>?) = lock.withLock {
            if let active = activeTerminationTask {
                return (nil, active)
            }
            guard let proc = process, proc.isRunning else {
                process = nil
                return (nil, nil)
            }
            process = nil
            let task = Task { [weak self] in
                await Self.performTermination(proc)
                self?.lock.withLock {
                    self?.activeTerminationTask = nil
                }
            }
            activeTerminationTask = task
            return (proc, task)
        }
        if let existingTask {
            await existingTask.value
            return
        }
        if procToTerminate != nil {
            await activeTerminationTask?.value
        }
    }

    private static func performTermination(_ p: Process) async {
        guard p.isRunning else { return }
        p.terminate()

        let start = DispatchTime.now().uptimeNanoseconds
        var sentKill = false
        while p.isRunning {
            let elapsed = DispatchTime.now().uptimeNanoseconds - start
            if elapsed >= 3_000_000_000 && !sentKill { // 3 seconds grace period before SIGKILL escalation
                kill(p.processIdentifier, SIGKILL)
                sentKill = true
            }
            if elapsed >= 8_000_000_000 { // 8 seconds maximum bounded wait
                break
            }
            try? await Task.sleep(for: .milliseconds(50))
        }
        if p.isRunning {
            kill(p.processIdentifier, SIGKILL)
            p.waitUntilExit()
        }
    }
}

public final class MockWorkerProcessController: WorkerProcessControlling, @unchecked Sendable {
    private let lock = NSLock()
    public var exitHandler: (@Sendable (Int32) -> Void)?
    public private(set) var spawnCount = 0
    public private(set) var terminateCount = 0
    public private(set) var lastPort: Int?
    public private(set) var events: [String] = []
    private var running = false

    public init(isRunning: Bool = false) {
        self.running = isRunning
    }

    public var isRunning: Bool {
        lock.withLock { running }
    }

    public func spawn(port: Int, configuration: ServiceConfiguration) async throws {
        lock.withLock {
            spawnCount += 1
            lastPort = port
            running = true
            events.append("spawn")
        }
    }

    public func terminate() async {
        let shouldNotify: Bool = lock.withLock {
            terminateCount += 1
            events.append("terminate")
            let wasRunning = running
            running = false
            return wasRunning
        }
        if shouldNotify {
            exitHandler?(0)
        }
    }

    public func simulateCrash(exitCode: Int32 = 1) {
        let shouldNotify: Bool = lock.withLock {
            let wasRunning = running
            running = false
            return wasRunning
        }
        if shouldNotify {
            exitHandler?(exitCode)
        }
    }
}

public actor WorkerSession: AppSession, SupervisorSessionControlling {
    public let internalPort: Int
    public let userPort: Int
    public private(set) var configuration: ServiceConfiguration
    private let store: ConfigurationStore
    private let workerController: any WorkerProcessControlling
    private var proxy: SupervisorProxy?
    public private(set) var state: WorkerSessionState = .down
    public var startupTimeoutSeconds: Int = 120
    public var maxLoadingWaitSeconds: Int? = nil
    public var readinessPollIntervalMs: Int = 250
    public var progressLogIntervalSeconds: Double = 30.0

    public var readinessTimeoutSeconds: Int {
        get { startupTimeoutSeconds }
        set { startupTimeoutSeconds = newValue }
    }
    private let appEventLog: AppEventLog
    private var lastActivity: Date = Date()

    public func setReadinessTiming(
        timeoutSeconds: Int = 120,
        pollIntervalMs: Int = 250,
        maxLoadingWaitSeconds: Int? = nil,
        progressLogIntervalSeconds: Double = 30.0
    ) {
        self.startupTimeoutSeconds = timeoutSeconds
        self.readinessPollIntervalMs = pollIntervalMs
        self.maxLoadingWaitSeconds = maxLoadingWaitSeconds
        self.progressLogIntervalSeconds = progressLogIntervalSeconds
    }
    private var inFlightInferences: Int = 0
    private var idleTimerTask: Task<Void, Never>?
    private var consecutiveFailures: Int = 0
    private var lastError: String? = nil
    private var intentionalTermination: Bool = false
    private var loadWaiters: [CheckedContinuation<LoadReport, any Error>] = []
    public var onNotice: (@Sendable (AppNotice) -> Void)?
    public var customHealthPoller: (@Sendable (Int) async -> Bool)?
    public var customHealthStatusPoller: (@Sendable (Int) async -> WorkerHealthStatus)?

    public func setCustomHealthStatusPoller(_ poller: (@Sendable (Int) async -> WorkerHealthStatus)?) {
        self.customHealthStatusPoller = poller
    }

    public func setCustomHealthPoller(_ poller: (@Sendable (Int) async -> Bool)?) {
        self.customHealthPoller = poller
    }

    /// Consulted before idle eviction: a respawned worker reloads from the E5
    /// cache, so evicting while macOS has purged it turns the next request into
    /// a full ANE re-specialization.
    public private(set) var cacheHealthProbe: @Sendable (ServiceConfiguration) -> E5CacheHealth = {
        E5CacheHealth.probe(configuration: $0)
    }

    public func setCacheHealthProbe(_ probe: @escaping @Sendable (ServiceConfiguration) -> E5CacheHealth) {
        cacheHealthProbe = probe
    }
    private let urlSession: URLSession
    private let tokenDirectory: URL

    public init(
        configuration: ServiceConfiguration,
        store: ConfigurationStore,
        internalPort: Int,
        workerController: any WorkerProcessControlling,
        tokenDirectory: URL? = nil,
        portOverride: Int? = nil,
        customHealthPoller: (@Sendable (Int) async -> Bool)? = nil
    ) {
        self.configuration = configuration
        self.store = store
        self.internalPort = internalPort
        self.userPort = portOverride ?? configuration.port
        self.workerController = workerController
        self.customHealthPoller = customHealthPoller
        self.appEventLog = AppEventLog(home: store.home)
        let tokDir = tokenDirectory ?? store.home.appendingPathComponent(".embed-ane")
        self.tokenDirectory = tokDir
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 5
        config.timeoutIntervalForResource = 10
        config.connectionProxyDictionary = [:]
        self.urlSession = URLSession(configuration: config)

        workerController.exitHandler = { [weak self] status in
            Task { [weak self] in
                await self?.handleWorkerExit(status: status)
            }
        }
    }

    public static func production(configuration: ServiceConfiguration, store: ConfigurationStore) throws -> WorkerSession {
        let expanded = store.expandingPaths(configuration)
        let internalPort = findInternalPort(userPort: expanded.port)
        let controller = DefaultWorkerProcessController(store: store)
        return WorkerSession(
            configuration: expanded,
            store: store,
            internalPort: internalPort,
            workerController: controller
        )
    }

    public func run(onListening: @escaping @Sendable (Int) async -> Void) async throws {
        let proxy = SupervisorProxy(
            listenPort: userPort,
            internalPort: internalPort,
            configuration: configuration,
            store: store,
            session: self,
            tokenDirectory: tokenDirectory
        )
        self.proxy = proxy
        armIdleTimerIfNeeded()
        try await proxy.run(onServerRunning: onListening)
    }

    public func snapshot() async throws -> AppSnapshot {
        let baseConfig = configuration
        switch state {
        case .ready:
            if let (stats, effSettings) = await fetchWorkerStatsAndSettings() {
                return AppSnapshot(
                    state: .ready,
                    queueDepth: stats.runtime.queueDepth,
                    inFlight: inFlightInferences,
                    preparing: stats.runtime.preparing,
                    residentBytes: stats.runtime.residentBytes,
                    windowCount: stats.windowCount,
                    p50NS: stats.p50NS,
                    p95NS: stats.p95NS,
                    effective: effSettings.configuration,
                    desired: baseConfig,
                    restartRequired: effSettings.restartRequired
                )
            }
            return AppSnapshot(
                state: .ready,
                inFlight: inFlightInferences,
                effective: baseConfig,
                desired: baseConfig
            )
        case .starting:
            return AppSnapshot(
                state: .loading,
                inFlight: inFlightInferences,
                effective: baseConfig,
                desired: baseConfig
            )
        case .terminating:
            return AppSnapshot(
                state: .unloading,
                effective: baseConfig,
                desired: baseConfig
            )
        case .down:
            return AppSnapshot(
                state: .unloaded,
                effective: baseConfig,
                desired: baseConfig
            )
        case .failed:
            return AppSnapshot(
                state: .failed,
                effective: baseConfig,
                desired: baseConfig
            )
        }
    }

    public func load() async throws {
        _ = try await loadWorker()
    }

    public func unload() async throws {
        _ = try await unloadWorker()
    }

    public func loadWorker() async throws -> LoadReport {
        switch state {
        case .ready:
            return LoadReport(perChunkNS: Array(repeating: 0, count: 6), residentBytes: 0, computePlanChecked: true)
        case .starting:
            return try await withCheckedThrowingContinuation { continuation in
                loadWaiters.append(continuation)
            }
        case .terminating:
            while state == .terminating {
                try? await Task.sleep(for: .milliseconds(50))
            }
            return try await loadWorker()
        case .down, .failed:
            state = .starting
            intentionalTermination = false
            lastError = nil
            do {
                let report = try await spawnAndAwaitReady()
                state = .ready
                consecutiveFailures = 0
                lastActivity = Date()
                armIdleTimerIfNeeded()
                resumeWaiters(returning: report)
                return report
            } catch {
                state = .failed
                consecutiveFailures += 1
                lastError = (error as? EmbedANEError)?.code ?? "load_failed"
                resumeWaiters(throwing: error)
                throw error
            }
        }
    }

    public func unloadWorker() async throws -> UnloadReport {
        intentionalTermination = true
        cancelIdleTimer()
        state = .terminating
        await workerController.terminate()
        state = .down
        return UnloadReport(residentBytes: 0)
    }

    public func ensureWorkerStarted() async {
        guard state == .down else { return }
        Task {
            _ = try? await self.loadWorker()
        }
    }

    public func getWorkerState() async -> WorkerSessionState {
        state
    }

    public func getEffectiveConfiguration() async -> ServiceConfiguration {
        configuration
    }

    public func apply(_ overrides: ConfigurationOverrides) async throws {
        _ = try await applyConfigurationOverrides(overrides)
    }

    public func applyConfigurationOverrides(_ patch: ConfigurationOverrides) async throws -> EffectiveSettings {
        let candidate = try configuration.applying(patch)
        try candidate.validate()
        try store.save(candidate)
        self.configuration = candidate
        armIdleTimerIfNeeded()
        if state == .ready {
            if let forwarded = await forwardSettingsToWorker(patch) {
                return forwarded
            }
        }
        var restartFields: [String] = []
        if patch.port != nil && patch.port != configuration.port { restartFields.append("port") }
        if patch.modelID != nil && patch.modelID != configuration.modelID { restartFields.append("model_id") }
        if patch.modelRoot != nil && patch.modelRoot != configuration.modelRoot { restartFields.append("model_root") }
        if patch.maxBatch != nil && patch.maxBatch != configuration.maxBatch { restartFields.append("max_batch") }
        if patch.computeUnits != nil && patch.computeUnits != configuration.computeUnits { restartFields.append("compute_units") }
        return EffectiveSettings(configuration: candidate, restartRequired: restartFields.sorted())
    }

    public func recordInferenceActivity() async {
        inFlightInferences &+= 1
        lastActivity = Date()
    }

    public func finishInferenceActivity() async {
        inFlightInferences = max(0, inFlightInferences - 1)
        lastActivity = Date()
    }

    private func spawnAndAwaitReady() async throws -> LoadReport {
        if workerController.isRunning {
            intentionalTermination = true
            await workerController.terminate()
            intentionalTermination = false
        }
        try await workerController.spawn(port: internalPort, configuration: configuration)
        let ready = await pollUntilReady(timeoutSeconds: readinessTimeoutSeconds)
        if ready {
            return LoadReport(perChunkNS: Array(repeating: 0, count: 6), residentBytes: 0, computePlanChecked: true)
        }
        // Bounded retry: terminate previous worker cleanly before retrying spawn
        intentionalTermination = true
        await workerController.terminate()
        intentionalTermination = false

        try await workerController.spawn(port: internalPort, configuration: configuration)
        let secondTryReady = await pollUntilReady(timeoutSeconds: readinessTimeoutSeconds)
        if secondTryReady {
            return LoadReport(perChunkNS: Array(repeating: 0, count: 6), residentBytes: 0, computePlanChecked: true)
        }
        intentionalTermination = true
        await workerController.terminate()
        intentionalTermination = false
        throw EmbedANEError.failed("Worker process failed to become ready.")
    }

    public enum WorkerHealthStatus: Sendable, Equatable {
        case ready
        case loading
        case failed(reason: String)
        case unreachable
    }

    private func pollUntilReady(timeoutSeconds: Int) async -> Bool {
        let start = DispatchTime.now().uptimeNanoseconds
        let startupDeadlineNS = UInt64(timeoutSeconds) * 1_000_000_000
        var lastProgressLogNS = DispatchTime.now().uptimeNanoseconds
        var loadingStartNS: UInt64? = nil
        var unreachableStartNS: UInt64? = nil

        while !Task.isCancelled {
            if !workerController.isRunning {
                appEventLog.record(.workerProbeFailed)
                return false
            }

            let status: WorkerHealthStatus
            if let customStatus = customHealthStatusPoller {
                status = await customStatus(internalPort)
            } else if let customPoll = customHealthPoller {
                status = await customPoll(internalPort) ? .ready : .unreachable
            } else {
                status = await checkWorkerHealthStatus()
            }

            let now = DispatchTime.now().uptimeNanoseconds
            switch status {
            case .ready:
                return true

            case .loading:
                unreachableStartNS = nil
                // Server is listening, model is actively loading.
                // Readiness waiting is UNBOUNDED while loading unless maxLoadingWaitSeconds is set for tests.
                if loadingStartNS == nil {
                    loadingStartNS = now
                }
                if let maxLoading = maxLoadingWaitSeconds {
                    let loadingElapsed = now - (loadingStartNS ?? now)
                    if loadingElapsed >= UInt64(Double(maxLoading) * 1_000_000_000) {
                        appEventLog.record(.workerProbeFailed)
                        return false
                    }
                }
                let intervalNS = UInt64(progressLogIntervalSeconds * 1_000_000_000)
                if now - lastProgressLogNS >= intervalNS {
                    appEventLog.record(.workerProbeWaiting)
                    lastProgressLogNS = now
                }

            case let .failed(reason):
                appEventLog.record(.workerProbeFailed)
                self.lastError = reason
                return false

            case .unreachable:
                appEventLog.record(.workerProbeFailed)
                if loadingStartNS == nil {
                    let elapsed = now - start
                    if elapsed >= startupDeadlineNS { return false }
                } else {
                    if !workerController.isRunning { return false }
                    if unreachableStartNS == nil {
                        unreachableStartNS = now
                    }
                    let unreachableElapsed = now - (unreachableStartNS ?? now)
                    if unreachableElapsed >= startupDeadlineNS { return false }
                }
            }
            try? await Task.sleep(for: .milliseconds(readinessPollIntervalMs))
        }
        return false
    }

    public func checkWorkerHealthStatus() async -> WorkerHealthStatus {
        guard let url = URL(string: "http://127.0.0.1:\(internalPort)/health") else { return .unreachable }
        var req = URLRequest(url: url)
        req.timeoutInterval = 10
        guard let (data, response) = try? await urlSession.data(for: req),
              let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            return .unreachable
        }
        guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              obj["status"] as? String == "ok" else {
            return .unreachable
        }
        guard let model = obj["model"] as? [String: Any],
              let state = model["state"] as? String else {
            return .unreachable
        }
        switch state {
        case "ready":
            return .ready
        case "loading":
            return .loading
        case "failed":
            let code = (obj["error"] as? [String: Any])?["code"] as? String ?? "load_failed"
            return .failed(reason: code)
        default:
            return .unreachable
        }
    }

    public func checkWorkerHealth() async -> Bool {
        await checkWorkerHealthStatus() == .ready
    }

    private func resumeWaiters(returning report: LoadReport) {
        let waiters = loadWaiters
        loadWaiters.removeAll()
        for waiter in waiters {
            waiter.resume(returning: report)
        }
    }

    private func resumeWaiters(throwing error: any Error) {
        let waiters = loadWaiters
        loadWaiters.removeAll()
        for waiter in waiters {
            waiter.resume(throwing: error)
        }
    }

    private func armIdleTimerIfNeeded() {
        cancelIdleTimer()
        guard configuration.idleTimeoutS > 0, state == .ready else { return }
        let timeout = configuration.idleTimeoutS
        idleTimerTask = Task { [weak self] in
            while !Task.isCancelled {
                let interval = max(1.0, min(timeout, 5.0))
                try? await Task.sleep(for: .milliseconds(Int(interval * 1000)))
                guard !Task.isCancelled, let self else { return }
                await self.checkIdleEviction()
            }
        }
    }

    private func cancelIdleTimer() {
        idleTimerTask?.cancel()
        idleTimerTask = nil
    }

    public func checkIdleEviction() async {
        guard state == .ready, inFlightInferences == 0, configuration.idleTimeoutS > 0 else { return }
        let idleDuration = Date().timeIntervalSince(lastActivity)
        if idleDuration >= configuration.idleTimeoutS {
            guard cacheHealthProbe(configuration).isWarm else {
                // Stay resident and re-check after another full idle period.
                appEventLog.record(.evictionDeferredColdCache)
                lastActivity = Date()
                return
            }
            _ = try? await unloadWorker()
        }
    }

    private func handleWorkerExit(status: Int32) {
        guard state == .ready, !intentionalTermination else {
            if state != .starting {
                state = .down
            }
            return
        }
        let keepLoaded = (configuration.idleTimeoutS == 0)
        if keepLoaded {
            onNotice?(AppNotice(message: "Worker process exited unexpectedly. Restarting...", code: "worker_crashed"))
            state = .starting
            Task { [weak self] in
                guard let self else { return }
                do {
                    let report = try await self.spawnAndAwaitReady()
                    await self.setReady(report: report)
                } catch {
                    await self.setFailed(error: error)
                }
            }
        } else {
            state = .down
            onNotice?(AppNotice(message: "Worker process exited.", code: "worker_stopped"))
        }
    }

    public func getLastError() async -> String? {
        lastError
    }

    private func setReady(report: LoadReport) {
        state = .ready
        consecutiveFailures = 0
        lastActivity = Date()
        armIdleTimerIfNeeded()
        resumeWaiters(returning: report)
    }

    private func setFailed(error: any Error) {
        state = .failed
        consecutiveFailures += 1
        lastError = (error as? EmbedANEError)?.code ?? "load_failed"
        resumeWaiters(throwing: error)
    }

    private func fetchWorkerStatsAndSettings() async -> (RuntimeStatistics, EffectiveSettings)? {
        guard let token = try? ControlToken.loadOrCreate(in: tokenDirectory) else { return nil }
        guard let statsURL = URL(string: "http://127.0.0.1:\(internalPort)/control/stats"),
              let settingsURL = URL(string: "http://127.0.0.1:\(internalPort)/control/settings") else { return nil }

        var statsReq = URLRequest(url: statsURL)
        statsReq.setValue(token.authorizationHeader, forHTTPHeaderField: "Authorization")
        var settingsReq = URLRequest(url: settingsURL)
        settingsReq.setValue(token.authorizationHeader, forHTTPHeaderField: "Authorization")

        guard let (statsData, statsResp) = try? await urlSession.data(for: statsReq),
              let statsHTTP = statsResp as? HTTPURLResponse, statsHTTP.statusCode == 200,
              let stats = try? JSONDecoder().decode(RuntimeStatistics.self, from: statsData) else {
            return nil
        }
        guard let (settData, settResp) = try? await urlSession.data(for: settingsReq),
              let settHTTP = settResp as? HTTPURLResponse, settHTTP.statusCode == 200,
              let effSettings = try? JSONDecoder().decode(EffectiveSettings.self, from: settData) else {
            return nil
        }
        return (stats, effSettings)
    }

    private func forwardSettingsToWorker(_ patch: ConfigurationOverrides) async -> EffectiveSettings? {
        guard let token = try? ControlToken.loadOrCreate(in: tokenDirectory),
              let url = URL(string: "http://127.0.0.1:\(internalPort)/control/settings"),
              let body = try? JSONEncoder().encode(patch) else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = "PUT"
        req.setValue(token.authorizationHeader, forHTTPHeaderField: "Authorization")
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = body
        guard let (data, resp) = try? await urlSession.data(for: req),
              let http = resp as? HTTPURLResponse, http.statusCode == 200,
              let eff = try? JSONDecoder().decode(EffectiveSettings.self, from: data) else {
            return nil
        }
        return eff
    }
}

public func isPortAvailable(_ port: Int) -> Bool {
    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.stride)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(port).bigEndian
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    let sock = socket(AF_INET, SOCK_STREAM, 0)
    guard sock >= 0 else { return false }
    defer { close(sock) }
    var opt: Int32 = 1
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.stride))
    let bindResult = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.stride))
        }
    }
    return bindResult == 0
}

public func findFreePort() -> Int {
    var addr = sockaddr_in()
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.stride)
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = 0
    addr.sin_addr.s_addr = inet_addr("127.0.0.1")
    let sock = socket(AF_INET, SOCK_STREAM, 0)
    guard sock >= 0 else { return 15536 }
    defer { close(sock) }
    var opt: Int32 = 1
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.stride))
    let bindResult = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(sock, $0, socklen_t(MemoryLayout<sockaddr_in>.stride))
        }
    }
    guard bindResult == 0 else { return 15536 }
    var len = socklen_t(MemoryLayout<sockaddr_in>.stride)
    let nameResult = withUnsafeMutablePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            getsockname(sock, $0, &len)
        }
    }
    guard nameResult == 0 else { return 15536 }
    return Int(UInt16(bigEndian: addr.sin_port))
}

public func findInternalPort(userPort: Int, preferred: Int = 15536) -> Int {
    if preferred != userPort && isPortAvailable(preferred) {
        return preferred
    }
    let free = findFreePort()
    if free != userPort {
        return free
    }
    return findFreePort()
}
