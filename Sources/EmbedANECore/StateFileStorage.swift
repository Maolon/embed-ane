import Darwin
import Foundation

/// Descriptor-relative state files. Neither reads nor writes follow a final
/// symlink; publication is tmp+rename (replace) or tmp+link (create-only).
public struct StateFileStorage: Sendable {
    public let directory: URL
    public init(directory: URL) { self.directory = directory }

    public func read(_ name: String, maximumBytes: Int = 65_536, requirePrivate: Bool = false) throws -> Data? {
        try validateName(name)
        guard maximumBytes >= 0, maximumBytes < Int.max else {
            throw failure(name, "Invalid state-file size limit.")
        }
        guard let directoryFD = try openDirectory(create: false) else { return nil }
        defer { Darwin.close(directoryFD) }
        let fd = openat(directoryFD, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        if fd < 0 {
            if errno == ENOENT { return nil }
            throw failure(name, "Cannot open state file safely.")
        }
        defer { Darwin.close(fd) }
        var metadata = stat()
        guard fstat(fd, &metadata) == 0, (metadata.st_mode & S_IFMT) == S_IFREG,
              metadata.st_uid == geteuid(), metadata.st_size >= 0,
              metadata.st_size <= maximumBytes,
              !requirePrivate || (metadata.st_mode & 0o7777) == 0o600 else {
            throw failure(name, "State file must be owned, regular, within size limit, and have required permissions.")
        }
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: min(maximumBytes + 1, 16_384))
        while true {
            let count = buffer.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress, $0.count) }
            if count == 0 { break }
            if count < 0 {
                if errno == EINTR { continue }
                throw failure(name, "Cannot read state file.")
            }
            guard count <= maximumBytes - data.count else { throw failure(name, "State file exceeds size limit.") }
            data.append(contentsOf: buffer.prefix(count))
        }
        return data
    }

    /// Returns false when another process already published the create-only file.
    @discardableResult
    public func write(_ name: String, data: Data, replace: Bool = true) throws -> Bool {
        try validateName(name)
        guard let directoryFD = try openDirectory(create: true) else { throw failure(name, "Missing state directory.") }
        defer { Darwin.close(directoryFD) }
        let temporary = ".\(name).\(UUID().uuidString).tmp"
        let fd = openat(directoryFD, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw failure(name, "Cannot create temporary state file.") }
        defer { Darwin.close(fd); unlinkat(directoryFD, temporary, 0) }
        guard fchmod(fd, 0o600) == 0 else { throw failure(name, "Cannot set private file permissions.") }
        try data.withUnsafeBytes { bytes in
            var offset = 0
            while offset < bytes.count {
                guard let address = bytes.baseAddress else { throw failure(name, "Invalid state buffer.") }
                let count = Darwin.write(fd, address.advanced(by: offset), bytes.count - offset)
                if count < 0, errno == EINTR { continue }
                guard count > 0 else { throw failure(name, "Cannot write state file.") }
                offset += count
            }
        }
        guard fsync(fd) == 0 else { throw failure(name, "Cannot synchronize state file.") }
        if replace {
            var metadata = stat()
            if fstatat(directoryFD, name, &metadata, AT_SYMLINK_NOFOLLOW) == 0 {
                guard (metadata.st_mode & S_IFMT) == S_IFREG, metadata.st_uid == geteuid() else {
                    throw failure(name, "Refusing to replace a nonregular or foreign-owned state file.")
                }
            } else if errno != ENOENT { throw failure(name, "Cannot inspect existing state file.") }
            guard renameat(directoryFD, temporary, directoryFD, name) == 0 else { throw failure(name, "Atomic state replacement failed.") }
        } else {
            if linkat(directoryFD, temporary, directoryFD, name, 0) != 0 {
                if errno == EEXIST { return false }
                throw failure(name, "Atomic state publication failed.")
            }
            // The winner is fully written before its public name becomes visible.
            guard unlinkat(directoryFD, temporary, 0) == 0 else { throw failure(name, "Cannot remove temporary state link.") }
        }
        guard fsync(directoryFD) == 0 else { throw failure(name, "Cannot synchronize state directory.") }
        return true
    }

    private func openDirectory(create: Bool) throws -> Int32? {
        if create {
            do {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true,
                                                        attributes: [.posixPermissions: 0o700])
            } catch { throw failure("", "Cannot create state directory.") }
        }
        let fd = Darwin.open(directory.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        if fd < 0 {
            if !create, errno == ENOENT { return nil }
            throw failure("", "Cannot open state directory safely.")
        }
        var metadata = stat()
        guard fstat(fd, &metadata) == 0, metadata.st_uid == geteuid(), (metadata.st_mode & S_IFMT) == S_IFDIR else {
            Darwin.close(fd); throw failure("", "State directory must be an owned directory.")
        }
        if create, fchmod(fd, 0o700) != 0 {
            Darwin.close(fd); throw failure("", "Cannot set state directory permissions.")
        }
        return fd
    }
    private func validateName(_ name: String) throws {
        guard !name.isEmpty, name != ".", name != "..", !name.contains("/"), !name.contains("\\"), !name.contains("\0") else {
            throw EmbedANEError.unsafePath(name)
        }
    }
    private func failure(_ name: String, _ reason: String) -> EmbedANEError {
        .io(path: directory.appendingPathComponent(name).path, reason: reason)
    }
}
