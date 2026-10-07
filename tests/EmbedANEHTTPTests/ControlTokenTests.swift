import EmbedANECore
import Foundation
import Testing
@testable import EmbedANEHTTP

@Suite("Control-token storage")
struct ControlTokenTests {
    private func directory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-token-\(UUID())")
    }
    @Test func generatedTokenHas256BitsPrivatePermissionsAndStableIdentity() throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let token = try ControlToken.loadOrCreate(in: directory)
        let file = directory.appendingPathComponent("control-token")
        let text = try String(contentsOf: file, encoding: .utf8)
        #expect(text.utf8.count == 65 && text.hasSuffix("\n"))
        #expect(text.dropLast().utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) })
        let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
        #expect(try ControlToken.loadOrCreate(in: directory).authorizationHeader == token.authorizationHeader)
        #expect(!String(describing: token).contains(String(text.dropLast())))
        #expect(!String(reflecting: token).contains(String(text.dropLast())))
        #expect(token.accepts(token.authorizationHeader))
        #expect(!token.accepts("Bearer " + String(repeating: "z", count: 64)))
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["control-token"])
    }
    @Test func concurrentStartupsPublishExactlyOneCompleteToken() async throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let headers = try await withThrowingTaskGroup(of: String.self) { group in
            for _ in 0..<32 { group.addTask { try ControlToken.loadOrCreate(in: directory).authorizationHeader } }
            var values = [String]()
            for try await value in group { values.append(value) }
            return values
        }
        #expect(headers.count == 32 && Set(headers).count == 1)
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["control-token"])
    }
    @Test func insecureExistingTokenIsRejectedNotSilentlyRotated() throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        _ = try ControlToken.loadOrCreate(in: directory)
        let file = directory.appendingPathComponent("control-token")
        let before = try Data(contentsOf: file)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: file.path)
        #expect(throws: EmbedANEError.self) { try ControlToken.loadOrCreate(in: directory) }
        #expect(try Data(contentsOf: file) == before)
        let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o644)
    }
    @Test func malformedEmptyAndOversizedPrivateTokensFailClosed() throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let storage = StateFileStorage(directory: directory)
        for value in ["", "short", String(repeating: "a", count: 63), String(repeating: "a", count: 66),
                      String(repeating: "G", count: 64), String(repeating: "0", count: 64) + "\r\n"] {
            try storage.write("control-token", data: Data(value.utf8))
            #expect(throws: EmbedANEError.self) { try ControlToken.loadOrCreate(in: directory) }
            #expect(try storage.read("control-token", maximumBytes: 100) == Data(value.utf8))
        }
    }
    @Test func tokenAndStateDirectorySymlinksAreRejectedWithoutWritingTarget() throws {
        let directory = directory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let target = directory.appendingPathComponent("outside")
        try Data("sentinel".utf8).write(to: target)
        let state = directory.appendingPathComponent("state")
        try FileManager.default.createDirectory(at: state, withIntermediateDirectories: false)
        try FileManager.default.createSymbolicLink(at: state.appendingPathComponent("control-token"), withDestinationURL: target)
        #expect(throws: EmbedANEError.self) { try ControlToken.loadOrCreate(in: state) }
        #expect(try Data(contentsOf: target) == Data("sentinel".utf8))
        let link = directory.appendingPathComponent("link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: state)
        #expect(throws: EmbedANEError.self) { try ControlToken.loadOrCreate(in: link) }
    }
}
