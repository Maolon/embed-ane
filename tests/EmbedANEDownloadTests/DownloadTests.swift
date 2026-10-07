import XCTest
import EmbedANECore
@testable import EmbedANEDownload

final class DownloadTests: XCTestCase {
    func testBootstrap() { XCTAssertEqual(EmbedANEDownloadInfo.version, "0.1.0") }

    func testLoadHubTokenValidFileReturnsToken() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-test-token-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let tokenFile = tempDir.appendingPathComponent("hf-token")
        try Data("hf_test_secret_token_123\n".utf8).write(to: tokenFile)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tokenFile.path)

        let loaded = try ModelDownloader.loadHubToken(from: tempDir, requirePrivate: true)
        XCTAssertEqual(loaded, "hf_test_secret_token_123")
    }

    func testLoadHubTokenInsecurePermissionsThrows() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-test-insecure-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let tokenFile = tempDir.appendingPathComponent("hf-token")
        try Data("hf_insecure_token\n".utf8).write(to: tokenFile)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: tokenFile.path)

        XCTAssertThrowsError(try ModelDownloader.loadHubToken(from: tempDir, requirePrivate: true))
    }

    func testLoadHubTokenMissingFileReturnsNil() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-test-missing-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let loaded = try ModelDownloader.loadHubToken(from: tempDir, requirePrivate: true)
        XCTAssertNil(loaded)
    }

    func testDefaultEnvironmentPrecedence() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-test-prec-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let tokenFile = tempDir.appendingPathComponent("hf-token")
        try Data("hf_file_token\n".utf8).write(to: tokenFile)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tokenFile.path)

        // 1. Env variable takes precedence over file
        let envWithExplicit = ModelDownloader.defaultEnvironment(
            baseEnvironment: ["HF_TOKEN": "hf_env_token"],
            tokenDirectory: tempDir
        )
        XCTAssertEqual(envWithExplicit["HF_TOKEN"], "hf_env_token")

        // 2. File token is used when env variable is missing
        let envWithFallback = ModelDownloader.defaultEnvironment(
            baseEnvironment: [:],
            tokenDirectory: tempDir
        )
        XCTAssertEqual(envWithFallback["HF_TOKEN"], "hf_file_token")

        // 3. Missing file results in no HF_TOKEN
        let emptyDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-empty-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: emptyDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: emptyDir) }

        let envAnonymous = ModelDownloader.defaultEnvironment(
            baseEnvironment: [:],
            tokenDirectory: emptyDir
        )
        XCTAssertNil(envAnonymous["HF_TOKEN"])

        // 4. Insecure file is safely ignored (falls back to anonymous)
        let insecureDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-insec-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: insecureDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: insecureDir) }
        let insecureFile = insecureDir.appendingPathComponent("hf-token")
        try Data("hf_insecure_token\n".utf8).write(to: insecureFile)
        try FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: insecureFile.path)

        let envInsecure = ModelDownloader.defaultEnvironment(
            baseEnvironment: [:],
            tokenDirectory: insecureDir
        )
        XCTAssertNil(envInsecure["HF_TOKEN"])
    }
}
