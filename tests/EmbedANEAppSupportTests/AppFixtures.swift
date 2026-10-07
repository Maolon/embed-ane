import EmbedANEAppSupport
import EmbedANECore
import EmbedANETestSupport
import Foundation

actor AppFixtureConfiguration {
    var value = ServiceConfiguration(modelRoot: "/tmp/embed-ane-app-fixture")
    var failRead = false
    var saves = 0
    func read() throws -> ServiceConfiguration {
        if failRead { throw EmbedANEError.invalidSpec("fixture config") }
        return value
    }
    func save(_ value: ServiceConfiguration) { self.value = value; saves += 1 }
    func setFailRead(_ enabled: Bool) { failRead = enabled }
}
actor AppFixtureLibrary: AppModelLibrary {
    var catalog = InstallCatalog(models: [
        .init(id: ModelABI.defaultModelID, commit: String(repeating: "a", count: 40)),
        .init(id: "other-model", commit: String(repeating: "b", count: 40))])
    var failScan = false
    var delayedInstall = false
    var failFetchSpec = false
    var delayFetchSpec = false
    var conflictOnInstall: EmbedANEError?
    var scans = 0
    var installs = 0
    var replaces = 0
    var lastRoot = ""
    var lastRepo = ""
    var fixtureSpecYAML: String?

    func setCatalog(_ value: InstallCatalog) { catalog = value }
    func setFailScan(_ enabled: Bool) { failScan = enabled }
    func setDelayedInstall(_ enabled: Bool) { delayedInstall = enabled }
    func setFailFetchSpec(_ enabled: Bool) { failFetchSpec = enabled }
    func setDelayFetchSpec(_ enabled: Bool) { delayFetchSpec = enabled }
    func setConflictOnInstall(_ error: EmbedANEError?) { conflictOnInstall = error }
    func setFixtureSpecYAML(_ yaml: String) { fixtureSpecYAML = yaml }

    func scan(root: URL) throws -> InstallCatalog {
        scans += 1; lastRoot = root.path
        if failScan { throw EmbedANEError.verification(path: "fixture", reason: "corrupt") }
        return catalog
    }
    func install(spec: URL, root: URL, replace: Bool,
                 progress: @escaping @Sendable (InstallationProgress) -> Void) async throws -> InstallationResult {
        installs += 1; lastRoot = root.path
        if let conflict = conflictOnInstall, !replace {
            throw conflict
        }
        if replace { replaces += 1 }
        for value in 0..<4096 {
            progress(.init(phase: "downloading", path: "weights", received: Int64(value), total: 4096))
        }
        if delayedInstall { try await Task.sleep(for: .seconds(60)) }
        let specString = (try? String(contentsOf: spec, encoding: .utf8)) ?? ""
        let modelID = (try? ModelSpec.parse(specString))?.model.id
            ?? (try? StrictYAML.decode(SpecModelIDProbe.self, from: specString))?.model.id
            ?? ModelABI.defaultModelID
        return InstallationResult(modelID: modelID)
    }
    func fetchSpec(repo: String, endpoint: URL) async throws -> URL {
        lastRepo = repo
        if delayFetchSpec { try await Task.sleep(for: .seconds(60)) }
        if failFetchSpec { throw EmbedANEError.transport(path: repo, reason: "Repository or spec.yaml not found on Hugging Face (HTTP 404).") }
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("fixture-spec-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        let file = tempDir.appendingPathComponent("spec.yaml")
        let chunkManifests = (0..<6).map {
            """
              - path: chunks/chunk\($0).mlmodelc/manifest.txt
                sha256: \(String(repeating: "0", count: 64))
                size: 10
            """
        }.joined(separator: "\n")
        let yaml = fixtureSpecYAML ?? """
        spec_version: 1
        model:
          id: \(ModelABI.defaultModelID)
          dim: 2048
          max_seq: 512
          normalize: l2
        source:
          repo: \(repo)
          revision: \(String(repeating: "a", count: 40))
          endpoint: \(endpoint.absoluteString)
        files:
          - path: tokenizer.json
            sha256: \(String(repeating: "0", count: 64))
            size: 10
          - path: tokenizer_config.json
            sha256: \(String(repeating: "0", count: 64))
            size: 10
          - path: embed_table.fp16.npy
            sha256: \(String(repeating: "0", count: 64))
            size: 10
        \(chunkManifests)
        runtime:
          compute_units: cpu_and_ne
          pad_side: right
          mask_dtype: fp32
        """
        try Data(yaml.utf8).write(to: file)
        return file
    }
}

private struct SpecModelIDProbe: Decodable {
    struct Model: Decodable { let id: String }
    let model: Model
}
actor AppFixtureSession: AppSession {
    nonisolated let predictor: MockPredictor
    nonisolated let lifecycle: LifecycleActor
    let snapshotGate = AsyncGate()
    var runCount = 0
    var applyCount = 0
    var snapshotFails = false
    var runFails = false
    var applyFails = false
    var effective = ServiceConfiguration(modelRoot: "/tmp/embed-ane-app-fixture")
    var requested = ServiceConfiguration(modelRoot: "/tmp/embed-ane-app-fixture")
    init() throws {
        let predictor = MockPredictor()
        self.predictor = predictor
        lifecycle = try LifecycleActor(predictor: predictor, residentBytes: { 4096 })
    }
    func setRunFailure(_ enabled: Bool) { runFails = enabled }
    func setSnapshotFailure(_ enabled: Bool) { snapshotFails = enabled }
    func setApplyFailure(_ enabled: Bool) { applyFails = enabled }
    func run(onListening: @escaping @Sendable (Int) async -> Void) async throws {
        runCount += 1
        if runFails { throw EmbedANEError.io(path: "127.0.0.1:8080", reason: "EADDRINUSE") }
        await onListening(8080)
        try await Task.sleep(for: .seconds(3600))
    }
    func snapshot() async throws -> AppSnapshot {
        await snapshotGate.wait()
        if snapshotFails { throw EmbedANEError.io(path: "config.yaml", reason: "fixture read failure") }
        let s = await lifecycle.statistics()
        var pending: [String] = []
        if effective.port != requested.port { pending.append("port") }
        if effective.modelID != requested.modelID { pending.append("model_id") }
        if effective.modelRoot != requested.modelRoot { pending.append("model_root") }
        if effective.maxBatch != requested.maxBatch { pending.append("max_batch") }
        if effective.computeUnits != requested.computeUnits { pending.append("compute_units") }
        return AppSnapshot(state: s.runtime.state, queueDepth: s.runtime.queueDepth,
            inFlight: s.runtime.inFlight, preparing: s.runtime.preparing,
            residentBytes: s.runtime.residentBytes, windowCount: s.windowCount,
            p50NS: s.p50NS, p95NS: s.p95NS, effective: effective, desired: requested,
            restartRequired: pending.sorted())
    }
    func load() async throws { _ = try await lifecycle.load() }
    func unload() async throws { _ = try await lifecycle.unload() }
    func apply(_ overrides: ConfigurationOverrides) async throws {
        applyCount += 1
        if applyFails { throw EmbedANEError.io(path: "config.yaml", reason: "fixture write failure") }
        let candidate = try requested.applying(overrides)
        try await lifecycle.updateLiveSettings(idleTimeoutS: candidate.idleTimeoutS, maxQueueDepth: candidate.maxQueueDepth, autoLoad: candidate.autoLoad)
        requested = candidate; effective.idleTimeoutS = candidate.idleTimeoutS
        effective.maxQueueDepth = candidate.maxQueueDepth
        effective.autoLoad = candidate.autoLoad
    }
}
final class AppFixtureFactory: @unchecked Sendable {
    let session: AppFixtureSession
    private let lock = NSLock()
    private var calls = 0
    var count: Int { lock.withLock { calls } }
    init(session: AppFixtureSession) { self.session = session }
    func make(_ config: ServiceConfiguration) throws -> any AppSession {
        lock.withLock { calls += 1 }
        guard config.computeUnits == .cpuAndNE else { throw EmbedANEError.invalidRequest("Unsupported fixture units", param: "compute_units") }
        return session
    }
}
@MainActor struct AppFixture {
    let configuration = AppFixtureConfiguration()
    let library = AppFixtureLibrary()
    let session: AppFixtureSession
    let factory: AppFixtureFactory
    let model: MenuBarModel
    init(cacheWarm: Bool? = nil) throws {
        let session = try AppFixtureSession(), configuration = self.configuration, library = self.library
        self.session = session
        let factory = AppFixtureFactory(session: session); self.factory = factory
        model = MenuBarModel(services: AppServices(
            configuration: { try await configuration.read() },
            saveConfiguration: { await configuration.save($0) },
            makeSession: { try factory.make($0) }, library: library,
            expandRoot: { path in path.hasPrefix("~/") ? "/tmp/fixture-home/" + path.dropFirst(2) : path },
            cacheHealth: { _ in cacheWarm }))
    }
    func ready() async throws {
        model.start()
        try await eventually { @MainActor in self.model.snapshot?.state == .ready && !self.model.commanding && !self.model.scanning }
    }
}

struct AppTemporaryDirectory {
    let url: URL
    init() throws {
        url = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-s5-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }
    func remove() { try? FileManager.default.removeItem(at: url) }
}
