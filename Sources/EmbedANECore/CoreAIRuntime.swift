import Foundation

public final class CoreAIRuntime: EmbeddingPredictor, EmbeddingPreparer, Sendable {
    private final class Metadata: @unchecked Sendable {
        let lock = NSLock()
        var loading = false
        var verification: VerificationReport?
    }
    private let metadata: Metadata
    private let worker: SerialEmbeddingWorker
    private let bundle: URL

    public init(
        configuration: ServiceConfiguration,
        acquireLease: @escaping @Sendable () throws -> any Sendable
    ) throws {
        try configuration.validate()
        guard configuration.modelRoot.hasPrefix("/") else {
            throw EmbedANEError.invalidRequest("Resolve model_root before constructing the runtime.", param: "model_root")
        }
        let bundle = URL(fileURLWithPath: configuration.modelRoot, isDirectory: true)
            .appendingPathComponent(configuration.modelID, isDirectory: true)
        self.bundle = bundle
        let metadata = Metadata()
        self.metadata = metadata
        let options = CoreAISpecializationOptions(
            preferredUnit: configuration.computeUnits == .cpuOnly ? .cpu : .neuralEngine
        )
        worker = SerialEmbeddingWorker {
            CoreAICascadeBackend(
                bundle: bundle,
                modelID: configuration.modelID,
                acquireLease: acquireLease,
                verified: { report in metadata.lock.withLock { metadata.verification = report } },
                options: options
            )
        }
    }

    public func load() async throws -> LoadReport {
        metadata.lock.withLock { metadata.loading = true; metadata.verification = nil }
        defer { metadata.lock.withLock { metadata.loading = false } }
        let report = try await worker.load()
        return LoadReport(
            perChunkNS: report.perChunkNS,
            residentBytes: ProcessMemory.residentBytes(),
            computePlanChecked: false
        )
    }

    public func prepare(_ texts: [String]) async throws -> PredictRequest {
        if metadata.lock.withLock({ metadata.loading }) { throw EmbedANEError.loading }
        return try await worker.prepare(texts)
    }

    public func predict(_ request: PredictRequest) async throws -> PredictResult {
        try await worker.predict(request)
    }

    public func unload() async throws -> UnloadReport {
        try await worker.unload()
    }

    public func verificationReport() -> VerificationReport? {
        metadata.lock.withLock { metadata.verification }
    }

    public func computePlanAudit() -> [ChunkComputePlanAudit] {
        []
    }
}
