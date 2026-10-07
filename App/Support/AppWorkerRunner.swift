import Darwin
import Dispatch
import EmbedANECore
import EmbedANEDownload
import EmbedANEHTTP
import Foundation

public enum WorkerSignalServer {
    public static func isAddressInUseError(_ error: any Error) -> Bool {
        if let aneError = error as? EmbedANEError, case let .io(_, reason) = aneError {
            return reason.localizedCaseInsensitiveContains("Address already in use") ||
                   reason.localizedCaseInsensitiveContains("EADDRINUSE") ||
                   reason.contains("errno: 48")
        }
        let desc = String(describing: error)
        return desc.localizedCaseInsensitiveContains("Address already in use") ||
               desc.localizedCaseInsensitiveContains("EADDRINUSE") ||
               desc.contains("errno: 48")
    }

    public static func runServerWithBindRetry(
        _ server: EmbeddingHTTPServer,
        maxBindRetrySeconds: Double = 60.0,
        bindRetryIntervalSeconds: Double = 2.0,
        home: URL? = nil,
        onListening: (@Sendable (Int) async -> Void)? = nil
    ) async throws {
        final class ListenState: @unchecked Sendable {
            private let lock = NSLock()
            private var started = false
            func markStarted() { lock.withLock { started = true } }
            var hasStarted: Bool { lock.withLock { started } }
        }

        let start = DispatchTime.now().uptimeNanoseconds
        let maxDurationNS = UInt64(maxBindRetrySeconds * 1_000_000_000)
        var attempt = 1
        while !Task.isCancelled {
            let state = ListenState()
            do {
                try await server.run { port in
                    state.markStarted()
                    if let onListening {
                        await onListening(port)
                    }
                }
                return
            } catch {
                if state.hasStarted {
                    throw error
                }
                guard isAddressInUseError(error) else {
                    throw error
                }
                let elapsed = DispatchTime.now().uptimeNanoseconds - start
                if elapsed >= maxDurationNS {
                    AppWorkerRunner.logWorkerEvent("bind_retry_exhausted attempts=\(attempt)", home: home)
                    fputs("Embed ANE worker bind retry exhausted after \(attempt) attempts: \(error)\n", stderr)
                    fflush(stderr)
                    throw error
                }
                AppWorkerRunner.logWorkerEvent("bind_retry attempt=\(attempt) interval=\(bindRetryIntervalSeconds)s error=\(error)", home: home)
                fputs("Embed ANE worker address in use, retrying bind in \(bindRetryIntervalSeconds)s (attempt \(attempt)): \(error)\n", stderr)
                fflush(stderr)
                attempt += 1
                try await Task.sleep(for: .milliseconds(Int(bindRetryIntervalSeconds * 1000)))
            }
        }
        throw CancellationError()
    }

    public static func run(
        _ server: EmbeddingHTTPServer,
        maxBindRetrySeconds: Double = 60.0,
        bindRetryIntervalSeconds: Double = 2.0,
        home: URL? = nil,
        onListening: (@Sendable (Int) async -> Void)? = nil
    ) async throws {
        let previousINT = Darwin.signal(SIGINT, SIG_IGN)
        let previousTERM = Darwin.signal(SIGTERM, SIG_IGN)
        let (events, continuation) = AsyncStream<Void>.makeStream()
        let signals = [SIGINT, SIGTERM].map { value in
            DispatchSource.makeSignalSource(signal: value, queue: .global(qos: .userInitiated))
        }
        for source in signals {
            source.setEventHandler { continuation.yield(()) }
            source.resume()
        }
        defer {
            for source in signals { source.cancel() }
            continuation.finish()
            _ = Darwin.signal(SIGINT, previousINT)
            _ = Darwin.signal(SIGTERM, previousTERM)
        }
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask {
                try await runServerWithBindRetry(
                    server,
                    maxBindRetrySeconds: maxBindRetrySeconds,
                    bindRetryIntervalSeconds: bindRetryIntervalSeconds,
                    home: home,
                    onListening: onListening
                )
            }
            group.addTask {
                for await _ in events { break }
            }
            defer { group.cancelAll() }
            _ = try await group.next()
        }
    }
}

public enum AppWorkerRunner {
    public static func isWorkerRequested(
        arguments: [String] = CommandLine.arguments,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Bool {
        arguments.contains("--worker") || environment["EMBED_ANE_WORKER"] == "1"
    }

    public static func parseInternalPort(
        arguments: [String] = CommandLine.arguments,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        defaultPort: Int = 15536
    ) -> Int {
        if let index = arguments.firstIndex(of: "--internal-port"), index + 1 < arguments.count,
           let port = Int(arguments[index + 1]), (1...65535).contains(port) {
            return port
        }
        if let index = arguments.firstIndex(of: "--port"), index + 1 < arguments.count,
           let port = Int(arguments[index + 1]), (1...65535).contains(port) {
            return port
        }
        if let envPortStr = environment["EMBED_ANE_INTERNAL_PORT"],
           let port = Int(envPortStr), (1...65535).contains(port) {
            return port
        }
        return defaultPort
    }

    public static func prepareConfiguration(
        store: ConfigurationStore,
        internalPort: Int
    ) throws -> ServiceConfiguration {
        var config = try store.resolve()
        config.port = internalPort
        config.idleTimeoutS = 0 // FORCED: in-process unload is forbidden (the E5RT bug path)!
        return config
    }

    public static func setupWorkerLog(home: URL) {
        let logsDir = home.appendingPathComponent(".embed-ane/logs")
        try? FileManager.default.createDirectory(at: logsDir, withIntermediateDirectories: true)
        let logFile = logsDir.appendingPathComponent("worker.log")
        if !FileManager.default.fileExists(atPath: logFile.path) {
            FileManager.default.createFile(atPath: logFile.path, contents: nil)
        }
        _ = freopen(logFile.path, "a", stdout)
        _ = freopen(logFile.path, "a", stderr)
        setbuf(stdout, nil)
        setbuf(stderr, nil)
    }

    public static func logWorkerEvent(_ event: String, home: URL? = nil) {
        let line = ISO8601DateFormatter().string(from: Date()) + " " + event + "\n"
        fputs(line, stdout)
        fflush(stdout)
        if let home {
            let logDir = home.appendingPathComponent(".embed-ane/logs")
            try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
            let logFile = logDir.appendingPathComponent("worker.log")
            if !FileManager.default.fileExists(atPath: logFile.path) {
                FileManager.default.createFile(atPath: logFile.path, contents: nil)
            }
            if let handle = try? FileHandle(forWritingTo: logFile) {
                handle.seekToEndOfFile()
                if let data = line.data(using: .utf8) {
                    try? handle.write(contentsOf: data)
                }
                try? handle.close()
            }
        }
    }

    public static func createAndStartWorker(
        arguments: [String] = CommandLine.arguments,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        store: ConfigurationStore = .init(),
        makeRuntime: (@Sendable (ServiceConfiguration) throws -> ServingRuntimeComponents)? = nil,
        slowLoadThresholdSeconds: Double = 120.0
    ) async throws -> (server: EmbeddingHTTPServer, loadTask: Task<LoadReport, any Error>) {
        let internalPort = parseInternalPort(arguments: arguments, environment: environment)
        let config = try prepareConfiguration(store: store, internalPort: internalPort)
        let expanded = store.expandingPaths(config)
        let components: ServingRuntimeComponents
        if let makeRuntime {
            components = try makeRuntime(expanded)
        } else {
            let root = URL(fileURLWithPath: expanded.modelRoot, isDirectory: true)
            let id = expanded.modelID
            // Serving skips the compute-plan audit: it would specialize every
            // chunk a second time on each cold load (benchmarks keep it).
            components = try ServingRuntimeFactory.make(configuration: expanded, acquireLease: {
                try ModelUseLease(modelRoot: root, modelID: id)
            }, auditComputePlan: false)
        }
        logWorkerEvent("catalog_checked", home: store.home)

        let server = try EmbeddingHTTPServer(
            configuration: expanded,
            predictor: components.predictor,
            preparer: components.preparer,
            store: store,
            portOverride: internalPort
        )

        let slowLoadWatcher = Task {
            do {
                try await Task.sleep(for: .milliseconds(Int(slowLoadThresholdSeconds * 1000)))
                logWorkerEvent("slow load: possible E5 respecialization; do not kill mid-compile", home: store.home)
            } catch {
                // Cancelled if load finishes before threshold
            }
        }

        // Detached concurrent load task: server listener comes up FIRST so supervisor
        // can observe startup and /health progress immediately.
        let loadTask = Task<LoadReport, any Error> {
            defer { slowLoadWatcher.cancel() }
            do {
                let report = try await server.lifecycle.load()
                logWorkerEvent("load_completed", home: store.home)
                return report
            } catch {
                logWorkerEvent("load_failed", home: store.home)
                fputs("Embed ANE worker load failed: \(error)\n", stderr)
                fflush(stderr)
                throw error
            }
        }

        return (server, loadTask)
    }

    public static func run(
        arguments: [String] = CommandLine.arguments,
        environment: [String: String] = ProcessInfo.processInfo.environment,
        store: ConfigurationStore = .init(),
        makeRuntime: (@Sendable (ServiceConfiguration) throws -> ServingRuntimeComponents)? = nil,
        runServer: (@Sendable (EmbeddingHTTPServer) async throws -> Void)? = nil,
        slowLoadThresholdSeconds: Double = 120.0,
        maxBindRetrySeconds: Double = 60.0,
        bindRetryIntervalSeconds: Double = 2.0
    ) {
        setupWorkerLog(home: store.home)
        let semaphore = DispatchSemaphore(value: 0)

        Task {
            do {
                let (server, _) = try await createAndStartWorker(
                    arguments: arguments,
                    environment: environment,
                    store: store,
                    makeRuntime: makeRuntime,
                    slowLoadThresholdSeconds: slowLoadThresholdSeconds
                )

                let onListening: @Sendable (Int) async -> Void = { _ in
                    logWorkerEvent("server_listening", home: store.home)
                }

                if let runServer {
                    try await runServer(server)
                } else {
                    try await WorkerSignalServer.run(
                        server,
                        maxBindRetrySeconds: maxBindRetrySeconds,
                        bindRetryIntervalSeconds: bindRetryIntervalSeconds,
                        home: store.home,
                        onListening: onListening
                    )
                }
                logWorkerEvent("server_stopped", home: store.home)
            } catch {
                fputs("Embed ANE worker startup error: \(error)\n", stderr)
                fflush(stderr)
                Darwin.exit(1)
            }
            semaphore.signal()
        }

        semaphore.wait()
        Darwin.exit(0)
    }
}
