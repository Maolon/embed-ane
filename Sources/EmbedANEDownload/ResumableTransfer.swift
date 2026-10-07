import Foundation
import EmbedANECore

private enum RestartTransfer: Error { case rangeRejected, validatorChanged }

private struct ResumeValidator: Codable, Sendable {
    let etag: String?
    let lastModified: String?
    init(_ response: DownloadResponse) {
        etag = response["etag"]
        lastModified = response["last-modified"]
    }
    var ifRange: String? {
        if let etag, etag.hasPrefix("\""), etag.hasSuffix("\"") { return etag }
        return lastModified
    }
    func differs(from response: DownloadResponse) -> Bool {
        if let etag, etag != response["etag"] { return true }
        if let lastModified, lastModified != response["last-modified"] { return true }
        return false
    }
}

/// Each sink owns one descriptor, protected against caller/callback races. Data
/// callbacks perform bounded writes on the transport's serial blocking queue.
/// Neither the complete model file nor its fp32 equivalent is materialized.
private final class TransferSink: @unchecked Sendable {
    private let lock = NSLock()
    private let file: ArtifactFile
    private let directory: SecureDirectory
    private let handle: FileHandle
    private let metadataPath: String
    private let progress: @Sendable (FetchProgress) -> Void
    private var received: Int64
    private var validator: ResumeValidator?
    private var responseEnd: Int64?

    init(file: ArtifactFile, directory: SecureDirectory, partialPath: String,
         metadataPath: String, offset: Int64, metadata: Data?,
         progress: @escaping @Sendable (FetchProgress) -> Void) throws {
        self.file = file
        self.directory = directory
        self.metadataPath = metadataPath
        self.progress = progress
        received = offset
        validator = metadata.flatMap { try? JSONDecoder().decode(ResumeValidator.self, from: $0) }
        handle = FileHandle(fileDescriptor: try directory.openFile(partialPath, writable: true, create: true),
                            closeOnDealloc: true)
        guard try handle.seekToEnd() == UInt64(offset) else {
            try handle.close()
            try? directory.remove(partialPath)
            try? directory.remove(metadataPath)
            throw EmbedANEError.verification(path: file.path, reason: "Partial file changed or corrupt before resume: \(file.path)")
        }
    }
    deinit { try? handle.close() }

    var offset: Int64 { lock.withLock { received } }
    var ifRange: String? { lock.withLock { validator?.ifRange } }

    func begin() { lock.withLock { responseEnd = nil } }

    func header(_ response: DownloadResponse) throws {
        try lock.withLock {
            if response.status == 416 {
                if received > 0 { throw RestartTransfer.rangeRejected }
                throw EmbedANEError.transport(path: file.path, reason: "HTTP 416 on a full request")
            }
            guard response.status == 200 || response.status == 206 else {
                throw EmbedANEError.transport(path: file.path, reason: "HTTP \(response.status)")
            }
            if let encoding = response["content-encoding"], encoding.lowercased() != "identity" {
                throw EmbedANEError.transport(path: file.path, reason: "Encoded bodies are not byte-range artifacts")
            }
            let length: Int64
            if response.status == 206 {
                guard let range = response["content-range"], let parsed = Self.parseRange(range),
                      parsed.start == received, parsed.total == file.size else {
                    throw EmbedANEError.transport(path: file.path, reason: "Invalid or mismatched Content-Range")
                }
                if received > 0, validator?.differs(from: response) == true {
                    throw RestartTransfer.validatorChanged
                }
                responseEnd = parsed.end + 1
                length = parsed.end - parsed.start + 1
            } else {
                // A 200 following Range is a complete representation, not a
                // suffix. Restart before the first byte reaches disk.
                length = file.size
                if received > 0 {
                    try handle.truncate(atOffset: 0)
                    try handle.seek(toOffset: 0)
                    try handle.synchronize()
                    received = 0
                }
                responseEnd = file.size
            }
            if let header = response["content-length"] {
                guard let advertised = Int64(header), advertised == length else {
                    throw EmbedANEError.transport(path: file.path, reason: "Content-Length differs from the pinned size/range")
                }
            }
            validator = ResumeValidator(response)
            // Durable validator first, data second. SIGKILL can lose neither the
            // revision pin nor which representation the persisted prefix uses.
            try directory.atomicWrite(try JSONEncoder().encode(validator), to: metadataPath)
            progress(.init(phase: .downloading, path: file.path, receivedBytes: received, totalBytes: file.size))
        }
    }

    func write(_ data: Data) throws {
        try lock.withLock {
            guard let end = responseEnd else {
                throw EmbedANEError.transport(path: file.path, reason: "Body arrived before response validation")
            }
            guard Int64(data.count) <= end - received, Int64(data.count) <= file.size - received else {
                throw EmbedANEError.verification(path: file.path, reason: "Body exceeds the pinned size/range")
            }
            try handle.write(contentsOf: data)
            received += Int64(data.count)
            progress(.init(phase: .downloading, path: file.path, receivedBytes: received, totalBytes: file.size))
        }
    }

    func finishResponse() throws {
        try lock.withLock {
            guard let end = responseEnd, received == end else {
                throw EmbedANEError.transport(path: file.path, reason: "Truncated body; partial file retained")
            }
            try handle.synchronize()
        }
    }

    func reset() throws {
        try lock.withLock {
            try handle.truncate(atOffset: 0)
            try handle.seek(toOffset: 0)
            try handle.synchronize()
            received = 0
            validator = nil
            responseEnd = nil
            // Writing a nil validator avoids unlink races and is itself atomic.
            try directory.atomicWrite(Data("null".utf8), to: metadataPath)
        }
    }

    func flushAndClose() throws {
        try lock.withLock { try handle.synchronize(); try handle.close() }
    }

    private static func parseRange(_ value: String) -> (start: Int64, end: Int64, total: Int64)? {
        guard value.hasPrefix("bytes ") else { return nil }
        let fields = value.dropFirst(6).split(separator: "/", omittingEmptySubsequences: false)
        guard fields.count == 2 else { return nil }
        let range = fields[0].split(separator: "-", omittingEmptySubsequences: false)
        guard range.count == 2, let start = Int64(range[0]), let end = Int64(range[1]),
              let total = Int64(fields[1]), start >= 0, end >= start, end < total else { return nil }
        return (start, end, total)
    }
}

struct ResumableTransfer: Sendable {
    let storage: InstallStorage
    let source: HubSource
    let commit: String
    let transport: any DownloadTransport
    let io: DownloadIO
    let progress: @Sendable (FetchProgress) -> Void

    func fetch(_ file: ArtifactFile) async throws {
        try Task.checkCancellation()
        if try await io.run({ try storage.existingFileIsValid(file) }) {
            progress(.init(phase: .verified, path: file.path, receivedBytes: file.size, totalBytes: file.size))
            return
        }
        let key = FileDigest.sha256(Data(file.path.utf8))
        let partial = InstallStorage.transferDirectory + "/" + key + ".part"
        let metadata = InstallStorage.transferDirectory + "/" + key + ".json"
        let directory = try await io.run { try SecureDirectory(storage.stageURL) }
        let sink = try await io.run {
            let offset = try storage.filesystem.fileSize(storage.stagePath + "/" + partial) ?? 0
            guard offset <= file.size else {
                try storage.filesystem.removeFileIfPresent(storage.stagePath + "/" + partial)
                try storage.filesystem.removeFileIfPresent(storage.stagePath + "/" + metadata)
                throw EmbedANEError.verification(path: file.path, reason: "Oversized partial file removed")
            }
            let metaSize = try storage.filesystem.fileSize(storage.stagePath + "/" + metadata)
            let bytes = try metaSize.flatMap { $0 <= 16_384 ? try directory.read(metadata, limit: 16_384) : nil }
            return try TransferSink(file: file, directory: directory, partialPath: partial,
                                    metadataPath: metadata, offset: offset, metadata: bytes, progress: progress)
        }
        do {
            var restarts = 0
            while sink.offset < file.size {
                try Task.checkCancellation()
                var request = try source.artifactRequest(commit: commit, path: file.path)
                if sink.offset > 0 {
                    request.setValue("bytes=\(sink.offset)-", forHTTPHeaderField: "Range")
                    if let validator = sink.ifRange { request.setValue(validator, forHTTPHeaderField: "If-Range") }
                }
                sink.begin()
                do {
                    try await transport.execute(request, policy: source.policy,
                                                receiveResponse: { try sink.header($0) },
                                                receiveData: { try sink.write($0) })
                    try await io.run { try sink.finishResponse() }
                } catch is RestartTransfer {
                    restarts += 1
                    guard restarts <= 3 else {
                        throw EmbedANEError.transport(path: file.path, reason: "Repeated range/validator restarts")
                    }
                    try await io.run { try sink.reset() }
                }
            }
            try await io.run {
                try sink.flushAndClose()
                let staged = try ArtifactFile(path: partial, sha256: file.sha256, size: file.size)
                do { try FileDigest.verify(staged, in: directory) }
                catch EmbedANEError.verification {
                    throw EmbedANEError.verification(path: file.path, reason: "SHA-256 or size mismatch")
                }
                try directory.rename(partial, to: file.path)
                try storage.filesystem.removeFileIfPresent(storage.stagePath + "/" + metadata)
            }
            progress(.init(phase: .verified, path: file.path, receivedBytes: file.size, totalBytes: file.size))
        } catch {
            // Close before unlinking. A transport interruption never deletes a
            // useful prefix; only verification failure removes this one file.
            try? await io.run { try sink.flushAndClose() }
            if case EmbedANEError.verification = error {
                try await io.run {
                    try storage.filesystem.removeFileIfPresent(storage.stagePath + "/" + partial)
                    try storage.filesystem.removeFileIfPresent(storage.stagePath + "/" + metadata)
                }
            }
            if let typed = error as? EmbedANEError { throw typed }
            if error is CancellationError { throw CancellationError() }
            throw EmbedANEError.transport(path: file.path, reason: "Transfer interrupted; partial file retained")
        }
    }
}
