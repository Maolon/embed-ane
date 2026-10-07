import Foundation
import EmbedANECore

public struct FetchProgress: Sendable {
    public enum Phase: String, Sendable { case resolving, downloading, verified, verifying, promoted }
    public let phase: Phase
    public let path: String?
    public let receivedBytes: Int64
    public let totalBytes: Int64

    public init(phase: Phase, path: String? = nil, receivedBytes: Int64 = 0, totalBytes: Int64 = 0) {
        self.phase = phase
        self.path = path
        self.receivedBytes = receivedBytes
        self.totalBytes = totalBytes
    }
}

public struct FetchReport: Codable, Sendable {
    public enum Disposition: String, Codable, Sendable {
        case installed, replaced
        case alreadyInstalled = "already_installed"
        case verifiedOffline = "verified_offline"
    }
    public let disposition: Disposition
    public let modelID: String
    public let installURL: URL
    public let verification: VerificationReport
    /// A successful atomic rename is the commit point. Cleanup/durability
    /// warnings after that point must not be disguised as pre-commit failures.
    public let warnings: [String]
}

/// Explicit-trigger acquisition. Construction never accesses the network.
/// All digests and bundle activation checks are delegated to EmbedANECore.
public struct ModelDownloader: Sendable {
    private let transport: any DownloadTransport
    private let policy: DownloadTransportPolicy
    private let environment: @Sendable () -> [String: String]
    private let io = DownloadIO()

    public static func loadHubToken(
        from directory: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".embed-ane"),
        requirePrivate: Bool = true
    ) throws -> String? {
        let storage = StateFileStorage(directory: directory)
        guard let data = try storage.read("hf-token", maximumBytes: 4096, requirePrivate: requirePrivate) else {
            return nil
        }
        guard let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty else {
            return nil
        }
        return text
    }

    public static func defaultEnvironment(
        baseEnvironment: [String: String] = ProcessInfo.processInfo.environment,
        tokenDirectory: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".embed-ane")
    ) -> [String: String] {
        var env = baseEnvironment
        if let explicit = env["HF_TOKEN"], !explicit.isEmpty {
            return env
        }
        if let fileToken = (try? loadHubToken(from: tokenDirectory, requirePrivate: true)) ?? nil, !fileToken.isEmpty {
            env["HF_TOKEN"] = fileToken
        }
        return env
    }

    public init(
        transport: any DownloadTransport = URLSessionDownloadTransport(),
        policy: DownloadTransportPolicy = .httpsOnly,
        environment: @escaping @Sendable () -> [String: String] = { ModelDownloader.defaultEnvironment() }
    ) {
        self.transport = transport
        self.policy = policy
        self.environment = environment
    }

    public func verify(modelID: String, modelRoot: URL) async throws -> VerificationReport {
        try ModelSpec.validateID(modelID)
        return try await io.run {
            let lease = try ModelUseLease(modelRoot: modelRoot, modelID: modelID)
            defer { withExtendedLifetime(lease) {} }
            return try BundleVerifier.verify(at: modelRoot.appendingPathComponent(modelID), expectedID: modelID)
        }
    }

    public func fetchSpec(
        repo: String,
        endpoint: URL = URL(string: "https://huggingface.co")!,
        revision: String = "main",
        timeout: TimeInterval = 15
    ) async throws -> String {
        try Task.checkCancellation()
        let cleanRepo = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = cleanRepo.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && $0.utf8.allSatisfy { (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45, 46, 95].contains($0) } }) else {
            throw EmbedANEError.invalidSpec("Repository must be in org/name format.")
        }
        guard var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false) else {
            throw EmbedANEError.invalidSpec("Invalid endpoint URL: \(endpoint.absoluteString)")
        }
        let unreserved = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~")
        let segments = parts.map(String.init) + ["resolve", revision, "spec.yaml"]
        components.percentEncodedPath = "/" + (try segments.map { segment in
            guard let encoded = segment.addingPercentEncoding(withAllowedCharacters: unreserved) else {
                throw EmbedANEError.invalidSpec("Unencodable remote path")
            }
            return encoded
        }).joined(separator: "/")
        guard let url = components.url else { throw EmbedANEError.invalidSpec("Invalid download URL") }
        try policy.validate(url)

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        request.setValue("embed-ane/0.1.0", forHTTPHeaderField: "User-Agent")
        request.timeoutInterval = timeout
        if let token = environment()["HF_TOKEN"], !token.isEmpty {
            let hostLower = url.host?.lowercased() ?? ""
            let isHF = hostLower == "huggingface.co" || hostLower.hasSuffix(".huggingface.co")
            if isHF || policy == .loopbackHTTPForTests {
                request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
            }
        }

        final class Buffer: @unchecked Sendable {
            var data = Data()
            let lock = NSLock()
            func append(_ chunk: Data) throws {
                try lock.withLock {
                    guard data.count + chunk.count <= 1_048_576 else {
                        throw EmbedANEError.invalidSpec("spec.yaml exceeds 1 MiB limit")
                    }
                    data.append(chunk)
                }
            }
        }
        let buffer = Buffer()
        do {
            try await transport.execute(request, policy: policy, receiveResponse: { response in
                if response.status == 404 {
                    throw EmbedANEError.transport(path: cleanRepo, reason: "Repository or spec.yaml not found on Hugging Face (HTTP 404).")
                }
                guard response.status == 200 else {
                    throw EmbedANEError.transport(path: cleanRepo, reason: "HTTP \(response.status)")
                }
            }, receiveData: { chunk in
                try buffer.append(chunk)
            })
        } catch let error as EmbedANEError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw EmbedANEError.transport(path: cleanRepo, reason: error.localizedDescription)
        }

        let yaml = try buffer.lock.withLock {
            guard let text = String(data: buffer.data, encoding: .utf8), !text.isEmpty else {
                throw EmbedANEError.invalidSpec("Downloaded spec.yaml is empty or not UTF-8.")
            }
            return text
        }
        let spec = try ModelSpec.parse(yaml)
        // P0 security constraint: prevent credential exfiltration via malicious remote specs.
        // Remotely acquired specs must strictly match the user-selected repository and download endpoint,
        // and are forbidden from declaring authentication directives (`auth: env:*`).
        guard spec.source.repo.lowercased() == cleanRepo.lowercased() else {
            throw EmbedANEError.invalidSpec("Remote spec source.repo '\(spec.source.repo)' does not match requested repository '\(cleanRepo)'.")
        }
        func normalizeEndpoint(_ raw: String) -> String {
            var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            while s.hasSuffix("/") { s.removeLast() }
            return s.lowercased()
        }
        guard normalizeEndpoint(spec.source.endpoint) == normalizeEndpoint(endpoint.absoluteString) else {
            throw EmbedANEError.invalidSpec("Remote spec source.endpoint '\(spec.source.endpoint)' does not match download origin '\(endpoint.absoluteString)'.")
        }
        guard spec.source.auth == nil else {
            throw EmbedANEError.invalidSpec("Remote spec cannot specify authentication directives ('\(spec.source.auth ?? "")').")
        }
        return yaml
    }

    public func fetch(
        specPath: URL,
        modelRoot: URL,
        offline: Bool = false,
        replace: Bool = false,
        progress: @escaping @Sendable (FetchProgress) -> Void = { _ in }
    ) async throws -> FetchReport {
        try Task.checkCancellation()
        let (yaml, spec): (String, ModelSpec) = try await io.run {
            let directory = try SecureDirectory(specPath.deletingLastPathComponent())
            let bytes = try directory.read(specPath.lastPathComponent)
            guard let yaml = String(data: bytes, encoding: .utf8) else {
                throw EmbedANEError.invalidSpec("Model spec is not UTF-8")
            }
            return (yaml, try ModelSpec.parse(yaml))
        }
        let digest = FileDigest.sha256(Data(yaml.utf8))
        let install = modelRoot.appendingPathComponent(spec.model.id, isDirectory: true)
        if offline {
            guard !replace else { throw EmbedANEError.invalidRequest("--offline cannot replace an install", param: "replace") }
            // Deliberately precedes credentials, HubSource, resolution, staging,
            // session creation and transport invocation.
            let report = try await verify(modelID: spec.model.id, modelRoot: modelRoot)
            guard report.specDigest == digest else { throw EmbedANEError.conflict("Installed bundle differs from the requested spec") }
            return .init(disposition: .verifiedOffline, modelID: spec.model.id,
                         installURL: install, verification: report, warnings: [])
        }

        let storage = try await io.run { try InstallStorage(root: modelRoot, modelID: spec.model.id) }
        defer { withExtendedLifetime(storage) {} }
        try await io.run { try storage.validateArtifactPaths(spec.files) }
        let source = try HubSource(spec.source, environment: environment(), policy: policy)
        progress(.init(phase: .resolving))
        let commit = try await source.resolve(using: transport)
        try Task.checkCancellation()

        let existing = try await io.run { try storage.filesystem.isDirectory(spec.model.id) }
        if existing {
            // Invalid existing installs are not silently deleted. Even repairing
            // an invalid installation requires the explicit replacement flag.
            let verified: VerificationReport? = try await io.run {
                do { return try BundleVerifier.verify(at: install, expectedID: spec.model.id) }
                catch { return nil }
            }
            if let verified, verified.specDigest == digest, verified.resolvedCommit == commit {
                return .init(disposition: .alreadyInstalled, modelID: spec.model.id,
                             installURL: install, verification: verified, warnings: [])
            }
            guard replace else { throw EmbedANEError.conflict("Different or invalid bundle already installed as \(spec.model.id); use --replace after unloading") }
        }
        // Acquired before staging/transfer, held through commit. Runtime clients
        // cannot activate the old bundle midway through explicit replacement.
        let useLease = try await io.run { try storage.exclusiveUseLease() }
        defer { withExtendedLifetime(useLease) {} }
        try await io.run { try storage.prepareStage(commit: commit, specDigest: digest) }
        let transfer = ResumableTransfer(storage: storage, source: source, commit: commit,
                                         transport: transport, io: io, progress: progress)
        var paths = Set(spec.files.map { SafePath.canonical($0.path) })
        for file in spec.files {
            try await transfer.fetch(file)
            // Every model directory (chunks and the vision tower) is listed by its
            // inner manifest; expand all of them, as BundleVerifier does.
            if file.path.hasSuffix("/manifest.txt") {
                let inner = try await io.run {
                    try InnerManifest(data: SecureDirectory(storage.stageURL).read(file.path))
                }
                let prefix = String(file.path.dropLast("manifest.txt".count))
                for child in inner.files {
                    let member = try ArtifactFile(path: prefix + child.path, sha256: child.sha256, size: child.size)
                    guard paths.insert(SafePath.canonical(member.path)).inserted else {
                        throw EmbedANEError.invalidSpec("Duplicate expanded chunk member: \(member.path)")
                    }
                    try await transfer.fetch(member)
                }
                // Path safety, all digests, and unexpected-file rejection reuse
                // Core; the HTTP downloader does not invent directory digests.
                try await io.run {
                    try inner.verify(in: storage.stageURL.appendingPathComponent(file.path).deletingLastPathComponent())
                }
            }
        }
        try Task.checkCancellation()
        progress(.init(phase: .verifying))
        let verification = try await io.run {
            let manifest = try InstallManifest(specYAML: yaml, spec: spec, resolvedCommit: commit)
            try SecureDirectory(storage.stageURL).atomicWrite(Data(manifest.yaml().utf8), to: "manifest.yaml")
            let verification = try BundleVerifier.verify(at: storage.stageURL, expectedID: spec.model.id)
            // Internal resume metadata stays out of the promoted bundle. SHA and
            // spec receipts remain for crash recovery of interrupted swaps.
            try storage.filesystem.removeTree(storage.stagePath + "/" + InstallStorage.transferDirectory)
            return verification
        }
        try Task.checkCancellation()
        let warnings: [String] = try await io.run {
            try storage.filesystem.promote(stage: storage.stagePath, modelID: spec.model.id, replacing: existing)
            var warnings: [String] = []
            if !storage.filesystem.syncRoot() { warnings.append("Install committed; parent-directory fsync failed") }
            if existing {
                do { try storage.filesystem.removeTree(storage.stagePath) }
                catch { warnings.append("Install committed; retired staging cleanup pending") }
            }
            return warnings
        }
        progress(.init(phase: .promoted))
        return .init(disposition: existing ? .replaced : .installed, modelID: spec.model.id,
                     installURL: install, verification: verification, warnings: warnings)
    }
}
