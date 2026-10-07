import Foundation
import CryptoKit
import Darwin
import Yams

public enum SafePath {
    public static func validate(_ path: String) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty, path.utf8.count <= 4_096, !path.contains("\\"), !path.contains(":"),
              !path.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) || CharacterSet.whitespacesAndNewlines.contains($0) }),
              components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else { throw EmbedANEError.unsafePath(path) }
    }
    public static func canonical(_ path: String) -> String { path.precomposedStringWithCanonicalMapping.lowercased() }
}

/// Immutable root descriptor; each operation owns its own descendant descriptors.
/// openat/O_NOFOLLOW also rejects intermediate symlinks during path traversal.
public final class SecureDirectory: @unchecked Sendable {
    private let descriptor: Int32
    public let url: URL
    public init(_ url: URL) throws {
        self.url = url.standardizedFileURL
        let fd = open(url.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard fd >= 0 else { throw EmbedANEError.io(path: url.path, reason: String(cString: strerror(errno))) }
        descriptor = fd
    }
    deinit { close(descriptor) }
    private func parent(_ path: String, create: Bool) throws -> (Int32, String) {
        try SafePath.validate(path)
        let parts = path.split(separator: "/").map(String.init)
        guard let leaf = parts.last else { throw EmbedANEError.unsafePath(path) }
        var current = dup(descriptor)
        guard current >= 0 else { throw EmbedANEError.io(path: path, reason: "dup failed") }
        do {
            for part in parts.dropLast() {
                var next = openat(current, part, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
                if next < 0, errno == ENOENT, create {
                    guard mkdirat(current, part, 0o700) == 0 || errno == EEXIST else {
                        throw EmbedANEError.io(path: path, reason: String(cString: strerror(errno)))
                    }
                    next = openat(current, part, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
                }
                guard next >= 0 else { throw EmbedANEError.unsafePath(path) }
                close(current); current = next
            }
            return (current, leaf)
        } catch { close(current); throw error }
    }
    public func openFile(_ path: String, writable: Bool = false, create: Bool = false, exclusive: Bool = false) throws -> Int32 {
        let (parentFD, leaf) = try parent(path, create: create)
        defer { close(parentFD) }
        let flags = (writable ? O_RDWR : O_RDONLY) | O_CLOEXEC | O_NOFOLLOW | (create ? O_CREAT : 0) | (exclusive ? O_EXCL : 0)
        let fd = openat(parentFD, leaf, flags, 0o600)
        guard fd >= 0 else { throw EmbedANEError.io(path: path, reason: String(cString: strerror(errno))) }
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG, !writable || info.st_nlink == 1 else {
            close(fd); throw EmbedANEError.unsafePath(path)
        }
        return fd
    }
    public func remove(_ path: String) throws {
        let (parentFD, leaf) = try parent(path, create: false)
        defer { close(parentFD) }
        guard unlinkat(parentFD, leaf, 0) == 0 || errno == ENOENT else { throw EmbedANEError.io(path: path, reason: "unlink failed") }
    }
    public func rename(_ source: String, to destination: String) throws {
        let (sourceFD, sourceLeaf) = try parent(source, create: false)
        defer { close(sourceFD) }
        let (destinationFD, destinationLeaf) = try parent(destination, create: true)
        defer { close(destinationFD) }
        guard renameat(sourceFD, sourceLeaf, destinationFD, destinationLeaf) == 0 else {
            throw EmbedANEError.io(path: destination, reason: String(cString: strerror(errno)))
        }
        guard fsync(destinationFD) == 0 else { throw EmbedANEError.io(path: destination, reason: "directory fsync failed") }
    }
    public func read(_ path: String, limit: Int = 8 * 1_024 * 1_024) throws -> Data {
        guard limit >= 0, limit < Int.max else { throw EmbedANEError.invalidRequest("Invalid read limit.", param: nil) }
        let handle = FileHandle(fileDescriptor: try openFile(path), closeOnDealloc: true)
        defer { try? handle.close() }
        var result = Data()
        while let data = try handle.read(upToCount: min(65_536, limit + 1 - result.count)), !data.isEmpty {
            result.append(data)
            guard result.count <= limit else { throw EmbedANEError.verification(path: path, reason: "metadata size limit exceeded") }
        }
        return result
    }
    public func atomicWrite(_ data: Data, to path: String) throws {
        let temporary = path + ".tmp-" + UUID().uuidString
        let handle = FileHandle(fileDescriptor: try openFile(temporary, writable: true, create: true, exclusive: true), closeOnDealloc: true)
        do {
            try handle.write(contentsOf: data); try handle.synchronize(); try handle.close()
            try rename(temporary, to: path)
        } catch { try? handle.close(); try? remove(temporary); throw error }
    }
}

public enum FileDigest {
    public static func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    public static func verify(_ file: ArtifactFile, in directory: SecureDirectory) throws {
        let handle = FileHandle(fileDescriptor: try directory.openFile(file.path), closeOnDealloc: true)
        defer { try? handle.close() }
        var info = stat()
        guard fstat(handle.fileDescriptor, &info) == 0, info.st_size == file.size else {
            throw EmbedANEError.verification(path: file.path, reason: "size mismatch")
        }
        var hash = SHA256(); var count: Int64 = 0
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty {
            hash.update(data: data); count += Int64(data.count)
            guard count <= file.size else { throw EmbedANEError.verification(path: file.path, reason: "file grew during verification") }
        }
        let digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
        guard count == file.size, digest == file.sha256 else {
            throw EmbedANEError.verification(path: file.path, reason: "SHA-256 or length mismatch")
        }
    }
    public static func ofFile(_ path: String, in directory: SecureDirectory) throws -> String {
        let handle = FileHandle(fileDescriptor: try directory.openFile(path), closeOnDealloc: true)
        defer { try? handle.close() }
        var hash = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty { hash.update(data: data) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

public struct InnerManifest: Sendable {
    public let files: [ArtifactFile]
    public init(data: Data) throws {
        guard let text = String(data: data, encoding: .utf8), text.hasSuffix("\n"), !text.contains("\r"), !text.isEmpty else {
            throw EmbedANEError.invalidSpec("inner manifest must be nonempty UTF-8 with LF endings")
        }
        var entries: [ArtifactFile] = []; var paths = Set<String>()
        for line in text.dropLast().split(separator: "\n", omittingEmptySubsequences: false) {
            let fields = line.split(separator: " ", omittingEmptySubsequences: false)
            guard fields.count == 3, let size = Int64(fields[1]), size >= 0 else { throw EmbedANEError.invalidSpec("invalid inner manifest line") }
            let path = String(fields[0]); let canonical = SafePath.canonical(path)
            guard canonical != "manifest.txt", !canonical.hasSuffix(".part"), !canonical.hasSuffix(".etag"),
                  !canonical.split(separator: "/").contains(where: { $0.hasPrefix(".embed-ane-") }),
                  paths.insert(canonical).inserted else { throw EmbedANEError.invalidSpec("duplicate, reserved or self-referential inner manifest path") }
            entries.append(try ArtifactFile(path: path, sha256: String(fields[2]), size: size))
        }
        guard entries.map(\.path) == entries.map(\.path).sorted(by: { $0.utf8.lexicographicallyPrecedes($1.utf8) }) else {
            throw EmbedANEError.invalidSpec("inner manifest paths must be sorted by UTF-8 bytes")
        }
        for file in entries {
            let parts = file.path.split(separator: "/")
            for length in 1..<parts.count {
                guard !paths.contains(SafePath.canonical(parts.prefix(length).joined(separator: "/"))) else {
                    throw EmbedANEError.invalidSpec("inner manifest file/directory collision")
                }
            }
        }
        files = entries
    }
    public func verify(in directoryURL: URL) throws {
        let directory = try SecureDirectory(directoryURL)
        for file in files { try FileDigest.verify(file, in: directory) }
        let expected = Set(files.map(\.path)).union(["manifest.txt"])
        try inspect(directoryURL, relative: "", expected: expected)
    }
    private func inspect(_ directory: URL, relative: String, expected: Set<String>) throws {
        let keys: Set<URLResourceKey> = [.isSymbolicLinkKey, .isRegularFileKey, .isDirectoryKey]
        // Unlike enumerator's default error handler, a failed directory read throws.
        for url in try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: Array(keys)) {
            let path = relative + url.lastPathComponent
            let values = try url.resourceValues(forKeys: keys)
            guard values.isSymbolicLink != true else { throw EmbedANEError.unsafePath(path) }
            if values.isDirectory == true {
                guard expected.contains(where: { $0.hasPrefix(path + "/") }) else {
                    throw EmbedANEError.verification(path: path, reason: "unexpected directory")
                }
                _ = try SecureDirectory(url)
                try inspect(url, relative: path + "/", expected: expected)
            } else {
                guard values.isRegularFile == true, expected.contains(path) else {
                    throw EmbedANEError.verification(path: path, reason: "unexpected chunk member")
                }
            }
        }
    }
}

public struct InstallManifest: Codable, Sendable {
    public let manifestVersion: Int
    public let specDigest: String
    public let resolvedCommit: String
    public let files: [ArtifactFile]
    public let generator: String
    public let createdAt: String
    public let specYAML: String
    enum CodingKeys: String, CodingKey {
        case manifestVersion = "manifest_version", specDigest = "spec_digest", resolvedCommit = "resolved_commit"
        case files, generator, createdAt = "created_at", specYAML = "spec_yaml"
    }
    public init(specYAML: String, spec: ModelSpec, resolvedCommit: String) throws {
        guard Revision.isCommit(resolvedCommit) else { throw EmbedANEError.invalidSpec("resolved revision is not a commit") }
        manifestVersion = 1; specDigest = FileDigest.sha256(Data(specYAML.utf8)); self.resolvedCommit = resolvedCommit.lowercased()
        files = spec.files; generator = "embed-ane/0.1.0"; createdAt = ISO8601DateFormatter().string(from: Date()); self.specYAML = specYAML
    }
    public init(from decoder: any Swift.Decoder) throws {
        try StrictCoding.rejectUnknown(decoder, allowed: ["manifest_version", "spec_digest", "resolved_commit", "files", "generator", "created_at", "spec_yaml"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        manifestVersion = try c.decode(Int.self, forKey: .manifestVersion); specDigest = try c.decode(String.self, forKey: .specDigest)
        resolvedCommit = try c.decode(String.self, forKey: .resolvedCommit); files = try c.decode([ArtifactFile].self, forKey: .files)
        generator = try c.decode(String.self, forKey: .generator); createdAt = try c.decode(String.self, forKey: .createdAt)
        specYAML = try c.decode(String.self, forKey: .specYAML)
        guard manifestVersion == 1, ArtifactFile.isDigest(specDigest), Revision.isCommit(resolvedCommit),
              FileDigest.sha256(Data(specYAML.utf8)) == specDigest else { throw EmbedANEError.verification(path: "manifest.yaml", reason: "invalid provenance") }
    }
    public func yaml() throws -> String { try YAMLEncoder().encode(self) }
}

public struct VerificationReport: Codable, Sendable {
    public let modelID: String
    public let resolvedCommit: String
    public let specDigest: String
    public let artifactDigests: [String: String]
}

public enum BundleVerifier {
    public static func manifest(at url: URL) throws -> InstallManifest {
        let data = try SecureDirectory(url).read("manifest.yaml")
        guard let text = String(data: data, encoding: .utf8) else { throw EmbedANEError.invalidSpec("manifest is not UTF-8") }
        return try StrictYAML.decode(InstallManifest.self, from: text)
    }
    public static func verify(at url: URL, expectedID: String? = nil) throws -> VerificationReport {
        let manifest = try manifest(at: url)
        let spec = try ModelSpec.parse(manifest.specYAML)
        guard manifest.files == spec.files, expectedID == nil || expectedID == spec.model.id else {
            throw EmbedANEError.verification(path: "manifest.yaml", reason: "entry list or model id mismatch")
        }
        if case let .commit(commit) = spec.source.revision, commit != manifest.resolvedCommit {
            throw EmbedANEError.verification(path: "manifest.yaml", reason: "resolved SHA differs from the pinned commit")
        }
        let directory = try SecureDirectory(url)
        var digests: [String: String] = [:]
        for file in spec.files {
            try FileDigest.verify(file, in: directory); digests[file.path] = file.sha256
            if file.path.hasSuffix("/manifest.txt") {
                let inner = try InnerManifest(data: directory.read(file.path))
                let chunkURL = url.appendingPathComponent(file.path).deletingLastPathComponent()
                try inner.verify(in: chunkURL)
                let prefix = String(file.path.dropLast("manifest.txt".count))
                for child in inner.files { digests[prefix + child.path] = child.sha256 }
            }
        }
        return VerificationReport(modelID: spec.model.id, resolvedCommit: manifest.resolvedCommit, specDigest: manifest.specDigest, artifactDigests: digests)
    }
}

public enum LocalAssetVerifier {
    public static func manifest(at url: URL) throws -> InstallManifest {
        try BundleVerifier.manifest(at: url)
    }
    public static func verify(at url: URL, expectedID: String? = nil) throws -> VerificationReport {
        try BundleVerifier.verify(at: url, expectedID: expectedID)
    }
}
