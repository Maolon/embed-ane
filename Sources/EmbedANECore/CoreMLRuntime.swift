import Foundation

/// Production adapter for injection into LifecycleActor / EmbeddingHTTPServer.
/// The same serial worker owns tokenization, table mapping and all six models.
/// Async compute-plan inspection also prefers that worker's executor. CoreML's
/// own internal asynchronous work remains managed by the framework.
public final class CoreMLRuntime: EmbeddingPredictor, EmbeddingPreparer, Sendable {
    private final class Metadata: @unchecked Sendable {
        let lock = NSLock()
        var loading = false
        var verification: VerificationReport?
        var audit: [ChunkComputePlanAudit] = []
    }
    private let metadata: Metadata
    private let worker: SerialEmbeddingWorker
    private let bundle: URL
    private let auditComputePlan: Bool

    /// The composition must supply a retained installation lease, such as the downloader's
    /// ModelUseLease. It is acquired on the worker BEFORE bundle verification.
    ///
    /// `auditComputePlan` runs `MLComputePlan` over all six chunks after load.
    /// CoreML specializes a compute plan separately from the loaded model (a
    /// second E5 bundle per chunk), so a cold load costs twice the ANE compile
    /// time. Serving compositions turn it off; benchmarks keep it.
    public init(configuration: ServiceConfiguration,
                acquireLease: @escaping @Sendable () throws -> any Sendable,
                auditComputePlan: Bool = true) throws {
        try configuration.validate()
        guard configuration.computeUnits == .cpuAndNE else {
            throw EmbedANEError.invalidRequest("The Core ML runtime requires compute_units=cpu_and_ne.", param: "compute_units")
        }
        guard configuration.modelRoot.hasPrefix("/") else {
            throw EmbedANEError.invalidRequest("Resolve model_root before constructing the runtime.", param: "model_root")
        }
        let bundle = URL(fileURLWithPath: configuration.modelRoot, isDirectory: true)
            .appendingPathComponent(configuration.modelID, isDirectory: true)
        self.bundle = bundle
        self.auditComputePlan = auditComputePlan
        let metadata = Metadata()
        self.metadata = metadata
        worker = SerialEmbeddingWorker {
            CoreMLCascadeBackend(bundle: bundle, modelID: configuration.modelID, acquireLease: acquireLease,
                verified: { report in metadata.lock.withLock { metadata.verification = report } })
        }
    }
    public func load() async throws -> LoadReport {
        metadata.lock.withLock { metadata.loading = true; metadata.verification = nil; metadata.audit = [] }
        defer { metadata.lock.withLock { metadata.loading = false } }
        let report = try await worker.load()
        guard auditComputePlan else {
            return LoadReport(perChunkNS: report.perChunkNS, residentBytes: ProcessMemory.residentBytes(),
                computePlanChecked: false)
        }
        let audits = try await worker.withExecutor { [bundle] in await ComputePlanAuditor.inspect(bundle: bundle) }
        metadata.lock.withLock { metadata.audit = audits }
        return LoadReport(perChunkNS: report.perChunkNS, residentBytes: ProcessMemory.residentBytes(),
            computePlanChecked: audits.count == 6 && audits.allSatisfy { $0.status == "checked" })
    }
    public func prepare(_ texts: [String]) async throws -> PredictRequest {
        // A shared-queue tokenizer must not wait behind a slow load and turn a
        // request made during loading into a silently admitted ready request.
        if metadata.lock.withLock({ metadata.loading }) { throw EmbedANEError.loading }
        return try await worker.prepare(texts)
    }
    public func predict(_ request: PredictRequest) async throws -> PredictResult { try await worker.predict(request) }
    public func unload() async throws -> UnloadReport { try await worker.unload() }
    public func verificationReport() -> VerificationReport? { metadata.lock.withLock { metadata.verification } }
    public func computePlanAudit() -> [ChunkComputePlanAudit] { metadata.lock.withLock { metadata.audit } }
}
