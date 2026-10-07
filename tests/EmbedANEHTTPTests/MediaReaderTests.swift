@testable import EmbedANEHTTP
import EmbedANECore
import Foundation
import Testing

/// A file open blocked on a macOS privacy prompt must neither hang the request
/// nor hold up other media requests.
@Suite("Media file reader") struct MediaReaderTests {
    @Test func blockedReadTimesOutWithAPermissionHint() async throws {
        let reader = ImageInputReader(timeout: .milliseconds(200))
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        do {
            _ = try await reader.run(kind: "video") { () -> Int in release.wait(); return 1 }
            Issue.record("Expected a timeout")
        } catch let error as EmbedANEError {
            #expect(error.message.contains("Privacy & Security"))
        }
    }
    @Test func aBlockedReadDoesNotHoldUpOthers() async throws {
        // No wall-clock bound: the blocked read is released only after the
        // second read returns. If reads were serialized, the second read could
        // never finish and would fail with the timeout error instead.
        let reader = ImageInputReader(timeout: .seconds(30))
        let release = DispatchSemaphore(value: 0)
        let started = Flag()
        let blocked = Task { try await reader.run(kind: "image") { () -> Int in started.set(); release.wait(); return 1 } }
        while !started.value { try await Task.sleep(for: .milliseconds(5)) }  // the blocked read now holds a thread
        #expect(try await reader.run(kind: "image") { 2 } == 2)
        release.signal()
        #expect(try await blocked.value == 1)
    }
}

private final class Flag: @unchecked Sendable {
    private let lock = NSLock()
    private var isSet = false
    var value: Bool { lock.withLock { isSet } }
    func set() { lock.withLock { isSet = true } }
}
