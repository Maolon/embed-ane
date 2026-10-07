import Dispatch
import EmbedANECore
import EmbedANEDownload
import EmbedANEHTTP
import Foundation

struct RuntimeComponents: Sendable {
    let predictor: any EmbeddingPredictor
    let preparer: any EmbeddingPreparer
    let verification: @Sendable () -> VerificationReport?
    let audit: @Sendable () -> [ChunkComputePlanAudit]
    static func production(_ configuration: ServiceConfiguration) throws -> Self {
        let root = URL(fileURLWithPath: configuration.modelRoot, isDirectory: true)
        let components = try ServingRuntimeFactory.make(configuration: configuration) {
            try ModelUseLease(modelRoot: root, modelID: configuration.modelID)
        }
        return .init(predictor: components.predictor, preparer: components.preparer,
                     verification: components.verification ?? { nil },
                     audit: components.audit ?? { [] })
    }
}

struct ProductionCommands: CLICommandExecuting {
    /// `org/name` with Hugging Face's allowed characters and no file extension.
    static func isRepositoryName(_ value: String) -> Bool {
        let parts = value.split(separator: "/", omittingEmptySubsequences: false)
        return parts.count == 2 && !value.hasSuffix(".yaml") && !value.hasSuffix(".yml") && parts.allSatisfy { part in
            !part.isEmpty && part != "." && part != ".." && part.unicodeScalars.allSatisfy {
                CharacterSet.alphanumerics.contains($0) && $0.isASCII || "-._".unicodeScalars.contains($0)
            }
        }
    }

    let io = CLIFileWork()
    let makeRuntime: @Sendable (ServiceConfiguration) throws -> RuntimeComponents
    let runServer: @Sendable (EmbeddingHTTPServer) async throws -> Void
    init(makeRuntime: @escaping @Sendable (ServiceConfiguration) throws -> RuntimeComponents = RuntimeComponents.production,
         runServer: @escaping @Sendable (EmbeddingHTTPServer) async throws -> Void = SignalServer.run) {
        self.makeRuntime = makeRuntime; self.runServer = runServer
    }

    func execute(_ invocation: CLIInvocation, context: CLIContext) async throws -> CLIResult {
        let root = URL(fileURLWithPath: context.configuration.modelRoot, isDirectory: true)
        switch invocation.command {
        case .verify:
            guard let id = invocation.operand else { throw CLIUsageError("Missing model id.") }
            let report = try await ModelDownloader().verify(modelID: id, modelRoot: root)
            return .init(stdout: try CLIJSON.encode(report))
        case .fetch:
            guard let operand = invocation.operand else { throw CLIUsageError("Missing spec path or org/name repository.") }
            let downloader = ModelDownloader(environment: {
                ModelDownloader.defaultEnvironment(
                    baseEnvironment: context.environment,
                    tokenDirectory: context.store.home.appendingPathComponent(".embed-ane")
                )
            })
            // A local spec file wins; otherwise `org/name` reads spec.yaml from Hugging Face.
            let localSpec = context.path(operand)
            if FileManager.default.fileExists(atPath: localSpec.path) || invocation.offline || !Self.isRepositoryName(operand) {
                let report = try await downloader.fetch(specPath: localSpec, modelRoot: root,
                    offline: invocation.offline, replace: invocation.replace)
                return .init(stdout: try CLIJSON.encode(report))
            }
            let yaml = try await downloader.fetchSpec(repo: operand)
            let directory = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-spec-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: directory) }
            let spec = try await io.run {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                        attributes: [.posixPermissions: 0o700])
                let file = directory.appendingPathComponent("spec.yaml")
                try Data(yaml.utf8).write(to: file, options: .atomic)
                return file
            }
            let report = try await downloader.fetch(specPath: spec, modelRoot: root, offline: false, replace: invocation.replace)
            return .init(stdout: try CLIJSON.encode(report))
        case .verifyTokenizer: return try await verifyTokenizer(invocation, context: context)
        case .verifyVision: return try await VerifyVisionCommand.run(invocation, context: context, io: io)
        case .status:
            return .init(stdout: try await StatusClient.read(port: context.configuration.port))
        case .serve:
            let runtime = try makeRuntime(context.configuration)
            let predictor: any EmbeddingPredictor
            let preparer: any EmbeddingPreparer
            if !(runtime.predictor is any MultimodalEmbeddingPredictor),
               let visionPaths = try await io.run({ try VisionServingArtifacts.paths(context.configuration) }) {
                let visionPredictor: any MultimodalEmbeddingPredictor
                switch context.configuration.engineBackend {
                case .coreml:
                    visionPredictor = MultimodalPredictor(paths: visionPaths, resizeMode: context.configuration.visionResizeMode)
                case .coreai:
                    visionPredictor = CoreAIMultimodalPredictor(paths: visionPaths, resizeMode: context.configuration.visionResizeMode)
                }
                let combined = ServingMultimodalRuntime(text: runtime.predictor, preparer: runtime.preparer,
                                                       vision: visionPredictor)
                predictor = combined; preparer = combined
            } else {
                predictor = runtime.predictor; preparer = runtime.preparer
            }
            let server = try EmbeddingHTTPServer(configuration: context.configuration,
                predictor: predictor, preparer: preparer, store: context.store)
            // Keep-loaded is the default; never silently serve a mock adapter.
            _ = try await server.lifecycle.load()
            do { try await runServer(server) }
            catch {
                try? await finish(lifecycle: server.lifecycle)
                throw error
            }
            try await finish(lifecycle: server.lifecycle)
            return .init()
        case .bench: return try await bench(invocation, context: context)
        case .help: return .init(stdout: Data((CLIInvocation.help + "\n").utf8))
        }
    }

    private func verifyTokenizer(_ invocation: CLIInvocation, context: CLIContext) async throws -> CLIResult {
        guard let input = invocation.operand else { throw CLIUsageError("Missing golden JSONL.") }
        let assets = invocation.assets.map(context.path) ?? URL(fileURLWithPath: context.configuration.modelRoot)
            .appendingPathComponent(context.configuration.modelID)
        let report = try await io.run {
            // Explicit --assets is also supported for tokenizer-only folders
            // without a manifest/table/chunks. Default installed assets use the downloader's lease.
            let lease: ModelUseLease? = try invocation.assets == nil ? ModelUseLease(
                modelRoot: URL(fileURLWithPath: context.configuration.modelRoot), modelID: context.configuration.modelID) : nil
            defer { withExtendedLifetime(lease) {} }
            let directory = try SecureDirectory(assets)
            let names = ["tokenizer.json", "tokenizer_config.json"]
            var digests: [String: String] = [:]
            for name in names { digests[name] = try FileDigest.ofFile(name, in: directory) }
            let tokenizer = try LocalAssetTokenizer(assets: assets)
            let goldenURL = context.path(input)
            let golden = try SecureDirectory(goldenURL.deletingLastPathComponent())
                .read(goldenURL.lastPathComponent, limit: 256 * 1_024 * 1_024)
            let report = try TokenizerDifferential.verify(jsonl: golden, tokenizer: tokenizer,
                                                         minimumCases: 1000, assetDigests: digests)
            for name in names {
                guard try FileDigest.ofFile(name, in: directory) == digests[name] else {
                    throw EmbedANEError.verification(path: name, reason: "Tokenizer asset changed during verification.")
                }
            }
            let receipt = TokenizerGateReceipt(receiptVersion: 1,
                createdAt: ISO8601DateFormatter().string(from: Date()), goldenSHA256: FileDigest.sha256(golden), report: report)
            try StateFileStorage(directory: context.store.home.appendingPathComponent(".embed-ane"))
                .write("tokenizer-gate.json", data: CLIJSON.encode(receipt))
            return report
        }
        return .init(exitCode: report.gatePassed ? 0 : 2, stdout: try CLIJSON.encode(report))
    }

    /// Stop accepting HTTP first, then let already admitted shared predictions
    /// finish and release the model lease. Cleanup is not cancelled by its caller.
    private func finish(lifecycle: LifecycleActor) async throws {
        try await Task.detached {
            while true {
                let state = await lifecycle.snapshot()
                if state.queueDepth == 0, state.inFlight == 0, state.preparing == 0 { break }
                try await Task.sleep(for: .milliseconds(10))
            }
            _ = try await lifecycle.unload()
        }.value
    }

    private func bench(_ invocation: CLIInvocation, context: CLIContext) async throws -> CLIResult {
        guard let operand = invocation.operand else { throw CLIUsageError("Missing corpus path.") }
        let inputs = try await io.run {
            let url = context.path(operand)
            let data = try SecureDirectory(url.deletingLastPathComponent()).read(url.lastPathComponent, limit: 64 * 1_024 * 1_024)
            let texts = try BenchmarkInput.corpus(data)
            let references: [ReferenceEmbedding]?
            var referenceDigest: String?
            if let path = invocation.reference {
                let referenceURL = context.path(path)
                let bytes = try SecureDirectory(referenceURL.deletingLastPathComponent())
                    .read(referenceURL.lastPathComponent, limit: 256 * 1_024 * 1_024)
                references = try BenchmarkInput.references(bytes, texts: texts)
                referenceDigest = FileDigest.sha256(bytes)
            } else { references = nil }
            let receipt = try references == nil ? nil : TokenizerGateReceipt.read(store: context.store)
            return BenchInputs(texts: texts, references: references, receipt: receipt,
                               corpusDigest: FileDigest.sha256(data), referenceDigest: referenceDigest)
        }
        let runtime = try makeRuntime(context.configuration)
        // Disable eviction for a deterministic explicit load/measure/unload/reload
        // sequence. This affects only this benchmark instance, not config.yaml.
        let lifecycle = try LifecycleActor(predictor: runtime.predictor,
            maxQueueDepth: context.configuration.maxQueueDepth, maxBatch: context.configuration.maxBatch, idleTimeoutS: 0)
        do {
            let beforeLoad = ProcessMemory.residentBytes()
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let initialLoad = try await lifecycle.load()
            let initialLoadNS = DispatchTime.now().uptimeNanoseconds - loadStart
            guard let verification = runtime.verification() else {
                throw EmbedANEError.verification(path: "manifest.yaml", reason: "Runtime has no verified artifact provenance.")
            }
            try inputs.receipt?.validate(artifactDigests: verification.artifactDigests)
            let initialAudit = runtime.audit()
            let measurements = try await BenchmarkRunner.run(texts: inputs.texts, references: inputs.references,
                tokenizerGatePassed: inputs.receipt != nil, lifecycle: lifecycle, preparer: runtime.preparer)
            let loadedBytes = ProcessMemory.residentBytes()
            let unloaded = try await lifecycle.unload()
            let reloadStart = DispatchTime.now().uptimeNanoseconds
            let reload = try await lifecycle.load()
            let reloadNS = DispatchTime.now().uptimeNanoseconds - reloadStart
            guard let reverified = runtime.verification(), reverified.specDigest == verification.specDigest,
                  reverified.resolvedCommit == verification.resolvedCommit,
                  reverified.artifactDigests == verification.artifactDigests else {
                throw EmbedANEError.verification(path: "manifest.yaml", reason: "Bundle changed between unload and same-process reload.")
            }
            // A separately labeled first prediction proves reload is functional;
            // it is not mixed into the main corpus latency distribution.
            guard let first = inputs.texts.first else { throw CLIUsageError("Empty corpus.") }
            let reloadPrediction = try await lifecycle.embed([first], using: runtime.preparer)
            let finalUnload = try await lifecycle.unload()
            let metadata = try await io.run { HostProvenance.capture() }
            let artifact = BenchmarkArtifact(reportVersion: 1,
                createdAt: ISO8601DateFormatter().string(from: Date()), configuration: context.configuration,
                benchmarkIdleTimeoutS: 0, provenance: metadata, verification: verification,
                corpusSHA256: inputs.corpusDigest, referenceSHA256: inputs.referenceDigest,
                tokenizerGate: inputs.receipt, initialLoad: .init(totalNS: initialLoadNS, report: initialLoad),
                sameProcessReload: .init(totalNS: reloadNS, report: reload),
                reloadPrediction: reloadPrediction.timing, measurements: measurements,
                processRSS: .init(beforeLoad: beforeLoad, loaded: loadedBytes, postUnload: unloaded.residentBytes,
                    reloaded: reload.residentBytes, finalUnload: finalUnload.residentBytes), computePlans: initialAudit)
            let outputDirectory = invocation.outputDirectory.map(context.path) ?? context.workingDirectory.appendingPathComponent("bench")
            let name = "\(Int64(Date().timeIntervalSince1970 * 1000))-\(UUID().uuidString).json"
            let destination = outputDirectory.appendingPathComponent(name)
            _ = try await io.run {
                try StateFileStorage(directory: outputDirectory).write(name, data: CLIJSON.encode(artifact), replace: false)
            }
            let summary = BenchmarkSummary(report: destination.path, samples: measurements.samples.count,
                p50NS: measurements.latency.p50NS, p95NS: measurements.latency.p95NS,
                minCos: measurements.parity?.minimumCosine, meanCos: measurements.parity?.meanCosine,
                parityGatePassed: measurements.parity?.gatePassed)
            return .init(exitCode: measurements.parity?.gatePassed == false ? 2 : 0, stdout: try CLIJSON.encode(summary))
        } catch {
            try? await finish(lifecycle: lifecycle)
            throw error
        }
    }
}

private struct BenchInputs: Sendable {
    let texts: [String]
    let references: [ReferenceEmbedding]?
    let receipt: TokenizerGateReceipt?
    let corpusDigest: String
    let referenceDigest: String?
}
