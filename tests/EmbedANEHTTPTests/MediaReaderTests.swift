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
        let reader = ImageInputReader(timeout: .seconds(5))
        let release = DispatchSemaphore(value: 0)
        defer { release.signal() }
        let blocked = Task { try await reader.run(kind: "image") { () -> Int in release.wait(); return 1 } }
        let start = ContinuousClock.now
        #expect(try await reader.run(kind: "image") { 2 } == 2)
        #expect(ContinuousClock.now - start < .seconds(2))
        release.signal()
        #expect(try await blocked.value == 1)
    }
}
