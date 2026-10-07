import EmbedANEAppSupport
import EmbedANECore
import Foundation
import Testing

@Suite("App catalogue and private event log") struct CatalogAndLogTests {
    @Test func missingModelRootIsAnEmptyCatalogue() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        let library = InstalledModelLibrary(verify: { _, _ in throw EmbedANEError.failed("must not verify") })
        let report = try await library.scan(root: temp.url.appendingPathComponent("absent"))
        #expect(report.models.isEmpty && report.rejected.isEmpty)
    }
    @Test func onlyVerifiedSafeDirectoriesAreOffered() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        for name in ["valid-z", "valid-a", "broken", "INVALID", ".staging"] {
            try FileManager.default.createDirectory(at: temp.url.appendingPathComponent(name), withIntermediateDirectories: false)
        }
        try Data().write(to: temp.url.appendingPathComponent("ordinary-file"))
        try FileManager.default.createSymbolicLink(at: temp.url.appendingPathComponent("linked"), withDestinationURL: temp.url.appendingPathComponent("valid-a"))
        let library = InstalledModelLibrary(verify: { id, _ in
            if id == "broken" { throw EmbedANEError.verification(path: id, reason: "digest mismatch") }
            return VerifiedModel(id: id, commit: String(repeating: "a", count: 40))
        })
        let report = try await library.scan(root: temp.url)
        #expect(report.models.map(\.id) == ["valid-a", "valid-z"])
        #expect(Set(report.rejected.map(\.id)) == ["broken", "INVALID", "linked"])
    }
    @Test func verifierIDMismatchIsNotSelectable() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        try FileManager.default.createDirectory(at: temp.url.appendingPathComponent("a"), withIntermediateDirectories: false)
        let library = InstalledModelLibrary(verify: { _, _ in VerifiedModel(id: "other", commit: "bad") })
        let report = try await library.scan(root: temp.url)
        #expect(report.models.isEmpty)
        #expect(report.rejected.first?.code == "verification_failed")
    }
    @Test func rootSymlinkIsRejected() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        let link = temp.url.appendingPathComponent("root-link")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: temp.url)
        let library = InstalledModelLibrary()
        await #expect(throws: EmbedANEError.self) { try await library.scan(root: link) }
    }
    @Test func eventLogIsBoundedPrivateAndContainsOnlyFixedCodes() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        let log = AppEventLog(home: temp.url)
        for _ in 0..<520 { log.record(.serverListening) }
        let file = try await log.prepareFile() // Queue barrier includes all records.
        let text = try String(contentsOf: file, encoding: .utf8)
        let lines = text.split(separator: "\n")
        #expect(lines.count == 512)
        #expect(lines.allSatisfy { $0.hasSuffix(" server_listening") })
        let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
        #expect((attributes[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    }
    @Test func eventLogRefusesFinalSymlink() async throws {
        let temp = try AppTemporaryDirectory(); defer { temp.remove() }
        let log = AppEventLog(home: temp.url)
        let file = try await log.prepareFile()
        try FileManager.default.removeItem(at: file)
        let outside = temp.url.appendingPathComponent("do-not-modify")
        try Data("sentinel".utf8).write(to: outside)
        try FileManager.default.createSymbolicLink(at: file, withDestinationURL: outside)
        await #expect(throws: EmbedANEError.self) { try await log.prepareFile() }
        #expect(try String(contentsOf: outside, encoding: .utf8) == "sentinel")
    }
}
