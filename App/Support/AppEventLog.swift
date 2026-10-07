import Dispatch
import EmbedANECore
import Foundation

/// Bounded, private, atomic operational log. The input type makes it impossible
/// to accidentally record request bodies, bearer tokens, or arbitrary errors.
public final class AppEventLog: Sendable {
    public let file: URL
    private let queue = DispatchQueue(label: "embed-ane.app-log", qos: .utility)
    public init(home: URL = FileManager.default.homeDirectoryForCurrentUser) {
        file = home.appendingPathComponent(".embed-ane/logs/app.log")
    }
    public func record(_ event: AppEvent) {
        queue.async { [file] in
            // Logging failure never changes whether an operation succeeded.
            // Open logs rechecks the path and surfaces file-access failures.
            try? Self.append(event, file: file)
        }
    }
    public func prepareFile() async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            queue.async { [file] in
                do {
                    let storage = StateFileStorage(directory: file.deletingLastPathComponent())
                    if try storage.read(file.lastPathComponent, maximumBytes: 131_072, requirePrivate: true) == nil {
                        try storage.write(file.lastPathComponent, data: Data(), replace: false)
                    }
                    continuation.resume(returning: file)
                } catch { continuation.resume(throwing: error) }
            }
        }
    }
    private static func append(_ event: AppEvent, file: URL) throws {
        let storage = StateFileStorage(directory: file.deletingLastPathComponent())
        let old = try storage.read(file.lastPathComponent, maximumBytes: 131_072, requirePrivate: true) ?? Data()
        let lines = String(decoding: old, as: UTF8.self).split(separator: "\n").suffix(511)
        let line = ISO8601DateFormatter().string(from: Date()) + " " + event.rawValue + "\n"
        let prefix = lines.isEmpty ? "" : lines.joined(separator: "\n") + "\n"
        try storage.write(file.lastPathComponent, data: Data((prefix + line).utf8))
    }
}
