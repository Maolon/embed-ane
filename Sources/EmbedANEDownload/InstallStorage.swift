import Foundation
import Darwin
import EmbedANECore

/// Nonblocking cross-process lease shared by serving compositions. Acquire this
/// before verification/loading, retain it until unload completes. Core does not
/// import Download; the CLI/App owns the lease alongside its Core runtime.
public final class ModelUseLease: @unchecked Sendable {
    private let lock: InstallFileLock

    public init(modelRoot: URL, modelID: String) throws {
        try ModelSpec.validateID(modelID)
        let filesystem = try InstallFileSystem(root: modelRoot, create: false)
        lock = try InstallFileLock(filesystem: filesystem, modelID: modelID, purpose: "use", shared: true)
        guard try filesystem.isDirectory(modelID) else { throw EmbedANEError.modelNotLoaded }
    }
}

/// Lock files are never unlinked or moved. Otherwise two processes could lock
/// different inodes under the same filename after a promotion/replacement.
final class InstallFileLock: @unchecked Sendable {
    private let descriptor: Int32
    init(filesystem: InstallFileSystem, modelID: String, purpose: String, shared: Bool = false) throws {
        try filesystem.ensureDirectory(".locks")
        let path = ".locks/\(modelID).\(purpose)"
        let fd = try filesystem.directory.openFile(path, writable: true, create: true)
        guard flock(fd, (shared ? LOCK_SH : LOCK_EX) | LOCK_NB) == 0 else {
            let code = errno
            close(fd)
            if code == EWOULDBLOCK || code == EAGAIN {
                throw EmbedANEError.conflict(purpose == "fetch"
                    ? "A fetch for \(modelID) already holds the installation lock"
                    : "Model \(modelID) is active; unload it before replacement")
            }
            throw EmbedANEError.io(path: path, reason: "Unable to acquire file lock")
        }
        descriptor = fd
    }
    deinit { _ = flock(descriptor, LOCK_UN); close(descriptor) }
}

/// Root-anchored filesystem operations. Child paths are walked with openat and
/// O_NOFOLLOW; stale-stage deletion uses unlinkat and never follows a symlink.
final class InstallFileSystem: @unchecked Sendable {
    let root: URL
    let directory: SecureDirectory
    private let descriptor: Int32

    init(root: URL, create: Bool) throws {
        self.root = root.standardizedFileURL
        if create {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
        }
        directory = try SecureDirectory(root)
        let fd = open(root.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
        guard fd >= 0 else { throw EmbedANEError.unsafePath(root.path) }
        descriptor = fd
    }
    deinit { close(descriptor) }

    func ensureDirectory(_ path: String) throws {
        try SafePath.validate(path)
        var current = dup(descriptor)
        guard current >= 0 else { throw io(path) }
        defer { close(current) }
        for component in path.split(separator: "/").map(String.init) {
            if mkdirat(current, component, 0o700) != 0, errno != EEXIST { throw io(path) }
            let child = openat(current, component, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
            guard child >= 0 else { throw EmbedANEError.unsafePath(path) }
            close(current)
            current = child
        }
    }

    private func parent(_ path: String) throws -> (Int32, String)? {
        try SafePath.validate(path)
        let components = path.split(separator: "/").map(String.init)
        guard let leaf = components.last else { throw EmbedANEError.unsafePath(path) }
        var current = dup(descriptor)
        guard current >= 0 else { throw io(path) }
        do {
            for component in components.dropLast() {
                let child = openat(current, component, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
                if child < 0, errno == ENOENT { close(current); return nil }
                guard child >= 0 else { throw EmbedANEError.unsafePath(path) }
                close(current)
                current = child
            }
            return (current, leaf)
        } catch { close(current); throw error }
    }

    func attributes(_ path: String) throws -> stat? {
        guard let (parent, leaf) = try parent(path) else { return nil }
        defer { close(parent) }
        var attributes = stat()
        guard fstatat(parent, leaf, &attributes, AT_SYMLINK_NOFOLLOW) == 0 else {
            if errno == ENOENT { return nil }
            throw io(path)
        }
        guard (attributes.st_mode & S_IFMT) != S_IFLNK else { throw EmbedANEError.unsafePath(path) }
        return attributes
    }

    func isDirectory(_ path: String) throws -> Bool {
        guard let info = try attributes(path) else { return false }
        guard (info.st_mode & S_IFMT) == S_IFDIR else { throw EmbedANEError.unsafePath(path) }
        return true
    }

    func fileSize(_ path: String) throws -> Int64? {
        guard let info = try attributes(path) else { return nil }
        guard (info.st_mode & S_IFMT) == S_IFREG, info.st_nlink == 1 else {
            throw EmbedANEError.unsafePath(path)
        }
        return Int64(info.st_size)
    }

    func removeFileIfPresent(_ path: String) throws {
        if try fileSize(path) != nil { try directory.remove(path) }
    }

    func removeTree(_ path: String) throws {
        guard let (parent, leaf) = try parent(path) else { return }
        defer { close(parent) }
        try removeEntry(parent: parent, name: leaf, display: path)
    }

    private func removeEntry(parent: Int32, name: String, display: String) throws {
        var info = stat()
        if fstatat(parent, name, &info, AT_SYMLINK_NOFOLLOW) != 0 {
            if errno == ENOENT { return }
            throw io(display)
        }
        if (info.st_mode & S_IFMT) == S_IFDIR {
            let fd = openat(parent, name, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)
            guard fd >= 0 else { throw EmbedANEError.unsafePath(display) }
            guard let stream = fdopendir(fd) else { close(fd); throw io(display) }
            defer { closedir(stream) }
            while true {
                errno = 0
                guard let entry = readdir(stream) else {
                    guard errno == 0 else { throw io(display) }
                    break
                }
                let child = withUnsafePointer(to: entry.pointee.d_name) { pointer in
                    pointer.withMemoryRebound(to: CChar.self,
                                              capacity: MemoryLayout.size(ofValue: entry.pointee.d_name)) {
                        String(cString: $0)
                    }
                }
                if child == "." || child == ".." { continue }
                try removeEntry(parent: fd, name: child, display: display + "/" + child)
            }
            guard unlinkat(parent, name, AT_REMOVEDIR) == 0 else { throw io(display) }
        } else {
            // Symlinks in discarded staging are unlinked, never traversed.
            guard unlinkat(parent, name, 0) == 0 else { throw io(display) }
        }
    }

    func promote(stage: String, modelID: String, replacing: Bool) throws {
        guard try isDirectory(stage) else { throw io(stage) }
        if replacing {
            guard try isDirectory(modelID) else { throw EmbedANEError.conflict("Install disappeared before replacement") }
            // Exchange atomically: there is no missing-install crash window.
            // The caller holds the exclusive use lease, so no active install is
            // destroyed. The retired directory is now staging and is removable.
            guard renameatx_np(descriptor, stage, descriptor, modelID, UInt32(RENAME_SWAP)) == 0 else {
                throw io(modelID)
            }
        } else {
            guard renameatx_np(descriptor, stage, descriptor, modelID, UInt32(RENAME_EXCL)) == 0 else {
                if errno == EEXIST || errno == ENOTEMPTY { throw EmbedANEError.conflict("Install appeared before promotion") }
                throw io(modelID)
            }
        }
        // A rename is the commit point. Post-commit fsync/retired cleanup is
        // reported separately; it cannot truthfully be reported as "no install".
    }

    func syncRoot() -> Bool { fsync(descriptor) == 0 }
    private func io(_ path: String) -> EmbedANEError { .io(path: path, reason: "Filesystem operation failed (errno \(errno))") }
}

final class DownloadIO: Sendable {
    private let queue = DispatchQueue(label: "embed-ane.download.filesystem", qos: .utility)
    func run<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do { continuation.resume(returning: try body()) }
                catch { continuation.resume(throwing: error) }
            }
        }
    }
}

final class InstallStorage: @unchecked Sendable {
    let filesystem: InstallFileSystem
    let modelID: String
    let stagePath: String
    private let fetchLock: InstallFileLock
    static let transferDirectory = ".embed-ane-download"

    init(root: URL, modelID: String) throws {
        try ModelSpec.validateID(modelID)
        filesystem = try InstallFileSystem(root: root, create: true)
        self.modelID = modelID
        stagePath = ".staging/" + modelID
        fetchLock = try InstallFileLock(filesystem: filesystem, modelID: modelID, purpose: "fetch")
    }

    var installURL: URL { filesystem.root.appendingPathComponent(modelID, isDirectory: true) }
    var stageURL: URL { filesystem.root.appendingPathComponent(stagePath, isDirectory: true) }

    func exclusiveUseLease() throws -> InstallFileLock {
        try InstallFileLock(filesystem: filesystem, modelID: modelID, purpose: "use")
    }

    func prepareStage(commit: String, specDigest: String) throws {
        try filesystem.ensureDirectory(".staging")
        if try filesystem.isDirectory(stagePath) {
            // Missing/malformed provenance is unbound staging. Never resume it.
            let resolved = try metadataIfPresent("RESOLVED")
            let digest = try metadataIfPresent("SPEC_SHA256")
            if resolved != commit + "\n" || digest != specDigest + "\n" {
                try filesystem.removeTree(stagePath)
            }
        }
        try filesystem.ensureDirectory(stagePath)
        let directory = try SecureDirectory(stageURL)
        try directory.atomicWrite(Data((specDigest + "\n").utf8), to: "SPEC_SHA256")
        try directory.atomicWrite(Data((commit + "\n").utf8), to: "RESOLVED")
        try filesystem.ensureDirectory(stagePath + "/" + Self.transferDirectory)
    }

    private func metadataIfPresent(_ name: String) throws -> String? {
        let path = stagePath + "/" + name
        guard let size = try filesystem.fileSize(path) else { return nil }
        guard size <= 128 else { return nil }
        return String(data: try filesystem.directory.read(path, limit: 128), encoding: .utf8)
    }

    func validateArtifactPaths(_ files: [ArtifactFile]) throws {
        let reserved = Set(["manifest.yaml", "resolved", "spec_sha256", Self.transferDirectory])
        var paths = Set<String>()
        for file in files {
            let canonical = SafePath.canonical(file.path)
            let first = canonical.split(separator: "/").first.map(String.init) ?? ""
            guard !reserved.contains(first), paths.insert(canonical).inserted else {
                throw EmbedANEError.invalidSpec("Reserved or duplicate downloader path: \(file.path)")
            }
        }
    }

    func existingFileIsValid(_ file: ArtifactFile) throws -> Bool {
        guard try filesystem.fileSize(stagePath + "/" + file.path) != nil else { return false }
        let directory = try SecureDirectory(stageURL)
        do { try FileDigest.verify(file, in: directory); return true }
        catch EmbedANEError.verification {
            try directory.remove(file.path)
            return false
        }
    }
}
