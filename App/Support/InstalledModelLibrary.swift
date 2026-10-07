import EmbedANECore
import EmbedANEDownload
import Foundation

/// Every selectable entry has passed the same downloader/Core verification as the CLI.
/// A directory name or a manifest's existence alone is never sufficient.
public struct InstalledModelLibrary: AppModelLibrary, Sendable {
    public typealias Verify = @Sendable (String, URL) async throws -> VerifiedModel
    private let downloader: ModelDownloader
    private let verify: Verify
    private let io = AppIO()

    public init(downloader: ModelDownloader = .init(), verify: Verify? = nil) {
        self.downloader = downloader
        self.verify = verify ?? { id, root in
            let report = try await downloader.verify(modelID: id, modelRoot: root)
            return VerifiedModel(id: report.modelID, commit: report.resolvedCommit)
        }
    }
    public func scan(root: URL) async throws -> InstallCatalog {
        let candidates: (valid: [String], rejected: [RejectedInstall]) = try await io.run {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory) else {
                // A dangling root symlink must not be mistaken for a fresh install.
                if (try? root.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true {
                    throw EmbedANEError.unsafePath(root.path)
                }
                return ([], [])
            }
            guard isDirectory.boolValue else { throw EmbedANEError.unsafePath(root.path) }
            _ = try SecureDirectory(root)
            let entries = try FileManager.default.contentsOfDirectory(at: root,
                includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles])
            var ids: [String] = [], rejected: [RejectedInstall] = []
            for entry in entries {
                let id = entry.lastPathComponent
                do {
                    let values = try entry.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
                    guard values.isSymbolicLink != true else { throw EmbedANEError.unsafePath(id) }
                    // Ordinary files at the root are not offered as installations.
                    guard values.isDirectory == true else { continue }
                    try ModelSpec.validateID(id)
                    ids.append(id)
                } catch { rejected.append(.init(id: id, code: (error as? EmbedANEError)?.code ?? "io_error")) }
            }
            return (ids.sorted(), rejected.sorted { $0.id < $1.id })
        }
        var models: [VerifiedModel] = [], rejected = candidates.rejected
        for id in candidates.valid {
            try Task.checkCancellation()
            do {
                let result = try await verify(id, root)
                guard result.id == id else { throw EmbedANEError.verification(path: id, reason: "Model id mismatch.") }
                models.append(result)
            } catch is CancellationError { throw CancellationError() }
            catch { rejected.append(.init(id: id, code: (error as? EmbedANEError)?.code ?? "verification_failed")) }
        }
        return InstallCatalog(models: models, rejected: rejected.sorted { $0.id < $1.id })
    }
    public func install(spec: URL, root: URL, replace: Bool = false,
                        progress: @escaping @Sendable (InstallationProgress) -> Void) async throws -> InstallationResult {
        let report = try await downloader.fetch(specPath: spec, modelRoot: root, offline: false, replace: replace) {
            progress(InstallationProgress(phase: $0.phase.rawValue, path: $0.path,
                                          received: $0.receivedBytes, total: $0.totalBytes))
        }
        return InstallationResult(modelID: report.modelID, warnings: report.warnings)
    }

    public func fetchSpec(repo: String, endpoint: URL = URL(string: "https://huggingface.co")!) async throws -> URL {
        let yaml = try await downloader.fetchSpec(repo: repo, endpoint: endpoint)
        return try await io.run {
            let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-spec-" + UUID().uuidString)
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            let file = tempDir.appendingPathComponent("spec.yaml")
            do {
                try Data(yaml.utf8).write(to: file, options: .atomic)
                return file
            } catch {
                try? FileManager.default.removeItem(at: tempDir)
                throw error
            }
        }
    }
}
