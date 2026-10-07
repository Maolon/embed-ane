import Dispatch
import Darwin
import EmbedANEHTTP
import Foundation

/// Signal handling belongs to the executable, not an embedded App's HTTP server.
/// No async work or allocation is performed in a POSIX signal handler.
struct SignalServer {
    static func run(_ server: EmbeddingHTTPServer) async throws {
        let previousINT = Darwin.signal(SIGINT, SIG_IGN)
        let previousTERM = Darwin.signal(SIGTERM, SIG_IGN)
        let (events, continuation) = AsyncStream<Void>.makeStream()
        let signals = [SIGINT, SIGTERM].map { value in
            DispatchSource.makeSignalSource(signal: value, queue: .global(qos: .userInitiated))
        }
        for source in signals {
            source.setEventHandler { continuation.yield(()) }
            source.resume()
        }
        defer {
            for source in signals { source.cancel() }
            continuation.finish()
            _ = Darwin.signal(SIGINT, previousINT)
            _ = Darwin.signal(SIGTERM, previousTERM)
        }
        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { try await server.run() }
            group.addTask {
                for await _ in events { break }
            }
            defer { group.cancelAll() }
            _ = try await group.next()
        }
    }
}
