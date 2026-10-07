import Darwin
import Dispatch
import EmbedANECore
import Foundation

/// No URLSession: remote URLs never cause a network request. Local reads are
/// bounded, off the event loop, and reject symlinks and non-regular files.
enum ImageInputSource: Sendable {
    case data(Data), file(String)
    init(_ value: String) throws {
        let invalid = EmbedANEError.invalidRequest(
            "Use data:image/<type>;base64,<data> or file:///absolute/path for images.", param: "input")
        if value.lowercased().hasPrefix("data:") {
            guard let comma = value.firstIndex(of: ",") else { throw invalid }
            let header = value[..<comma].lowercased()
            guard header.hasPrefix("data:image/"), header.hasSuffix(";base64"),
                  header.dropFirst(11).dropLast(7).isEmpty == false,
                  header.filter({ $0 == ";" }).count == 1,
                  let bytes = Data(base64Encoded: String(value[value.index(after: comma)...])), !bytes.isEmpty else {
                throw invalid
            }
            guard bytes.count <= 64 * 1_024 * 1_024 else {
                throw EmbedANEError.invalidRequest("Encoded image exceeds 64 MiB.", param: "input")
            }
            self = .data(bytes)
            return
        }
        self = .file(try Self.filePath(value, kind: "image", invalid: invalid))
    }

    /// Strict `file:///absolute/path` parsing shared with `VideoInputSource`.
    static func filePath(_ value: String, kind: String, invalid: EmbedANEError) throws -> String {
        guard let components = URLComponents(string: value) else { throw invalid }
        if ["http", "https"].contains(components.scheme?.lowercased() ?? "") {
            throw EmbedANEError.invalidRequest("Remote \(kind) fetch not supported.", param: "input")
        }
        guard components.scheme?.lowercased() == "file", value.lowercased().hasPrefix("file:///"),
              components.host == nil || components.host == "", components.user == nil, components.password == nil,
              components.port == nil, components.query == nil, components.fragment == nil,
              let path = components.percentEncodedPath.removingPercentEncoding,
              path.hasPrefix("/"), !path.contains("\0"), !path.contains("\\") else { throw invalid }
        let parts = path.split(separator: "/", omittingEmptySubsequences: false).dropFirst()
        guard !parts.isEmpty, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else { throw invalid }
        return path
    }
    func read() throws -> Data {
        switch self {
        case let .data(bytes): return bytes
        case let .file(path):
            do { return try Self.readLocalFile(path) }
            catch {
                // No filesystem paths, existence details or arbitrary I/O errors
                // escape the public boundary.
                throw EmbedANEError.invalidRequest("Cannot read local image: use a readable regular file without symlinks, at most 64 MiB.", param: "input")
            }
        }
    }
    /// Opens a regular, nonempty file of at most `limit` bytes without
    /// following a symlink in any path component. The caller owns the fd.
    static func openRegularFile(_ path: String, limit: Int, kind: String) throws -> Int32 {
        var directory = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard directory >= 0 else { throw EmbedANEError.unsafePath(kind) }
        defer { close(directory) }
        let components = path.split(separator: "/").map(String.init)
        guard let leaf = components.last else { throw EmbedANEError.unsafePath(kind) }
        for component in components.dropLast() {
            let next = openat(directory, component, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
            guard next >= 0 else { throw EmbedANEError.unsafePath(kind) }
            close(directory); directory = next
        }
        // O_NONBLOCK prevents a FIFO from stalling before fstat can reject it.
        let fd = openat(directory, leaf, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { throw EmbedANEError.unsafePath(kind) }
        var info = stat()
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_size > 0, info.st_size <= limit else {
            close(fd); throw EmbedANEError.unsafePath(kind)
        }
        return fd
    }

    private static func readLocalFile(_ path: String) throws -> Data {
        let limit = 64 * 1_024 * 1_024
        let handle = FileHandle(fileDescriptor: try openRegularFile(path, limit: limit, kind: "image"), closeOnDealloc: true)
        defer { try? handle.close() }
        var result = Data()
        while let bytes = try handle.read(upToCount: min(65_536, limit + 1 - result.count)), !bytes.isEmpty {
            result.append(bytes)
            guard result.count <= limit else { throw EmbedANEError.unsafePath("image") }
        }
        return result
    }
}

/// Videos are decoded by AVFoundation from their path, so only a local file
/// URL is accepted, checked with the image rules (no symlinks, regular file).
struct VideoInputSource: Sendable {
    static let maximumBytes = 2 * 1_024 * 1_024 * 1_024
    let path: String
    init(_ value: String) throws {
        let invalid = EmbedANEError.invalidRequest("Use file:///absolute/path for videos.", param: "input")
        guard !value.lowercased().hasPrefix("data:") else {
            throw EmbedANEError.invalidRequest("Videos must be local files; data: URLs are not supported.", param: "input")
        }
        path = try ImageInputSource.filePath(value, kind: "video", invalid: invalid)
    }
    func validate() throws -> URL {
        do { close(try ImageInputSource.openRegularFile(path, limit: Self.maximumBytes, kind: "video")) }
        catch {
            throw EmbedANEError.invalidRequest("Cannot read local video: use a readable regular file without symlinks, at most 2 GiB.", param: "input")
        }
        return URL(fileURLWithPath: path)
    }
}

/// Runs blocking file checks and reads off the event loop.
///
/// `open()` on a privacy-protected location (an external volume, Desktop,
/// Documents, ...) blocks until the user answers macOS's permission prompt,
/// which may never happen for a background app. So the work runs on a
/// concurrent queue (one blocked file cannot hold up other requests) and the
/// request gives up after `timeout` with an actionable error.
final class ImageInputReader: Sendable {
    private let queue = DispatchQueue(label: "embed-ane.http.media-files", qos: .userInitiated, attributes: .concurrent)
    private let timeout: DispatchTimeInterval
    init(timeout: DispatchTimeInterval = .seconds(20)) { self.timeout = timeout }

    func validate(_ source: VideoInputSource) async throws -> URL { try await run(kind: "video") { try source.validate() } }
    func read(_ source: ImageInputSource) async throws -> Data { try await run(kind: "image") { try source.read() } }

    func run<T: Sendable>(kind: String, _ work: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            let once = ResumeOnce(continuation)
            queue.async { once.resume(with: Result { try work() }) }
            queue.asyncAfter(deadline: .now() + timeout) {
                once.resume(with: .failure(EmbedANEError.invalidRequest(
                    "Timed out opening the local \(kind). macOS may be waiting for permission to read that location: allow it in the prompt, or under System Settings > Privacy & Security > Files and Folders.",
                    param: "input")))
            }
        }
    }
}

/// Resumes a continuation exactly once, whichever of the read or the timeout finishes first.
private final class ResumeOnce<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<T, any Error>?
    init(_ continuation: CheckedContinuation<T, any Error>) { self.continuation = continuation }
    func resume(with result: Result<T, any Error>) {
        lock.withLock { () -> CheckedContinuation<T, any Error>? in
            defer { continuation = nil }
            return continuation
        }?.resume(with: result)
    }
}
