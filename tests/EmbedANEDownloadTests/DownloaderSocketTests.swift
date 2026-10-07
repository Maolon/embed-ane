import Foundation
import Darwin
import XCTest
import EmbedANECore
@testable import EmbedANEDownload

@MainActor
final class DownloaderSocketTests: XCTestCase {
    private func eventually(timeout: TimeInterval = 10, _ predicate: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !predicate() {
            guard Date() < deadline else { throw SocketFixtureError.timedOut }
            try await Task.sleep(for: .milliseconds(10))
        }
    }

    private func fails(_ code: String, file: StaticString = #filePath, line: UInt = #line,
                       _ operation: () async throws -> Void) async {
        do { try await operation(); XCTFail("Expected \(code)", file: file, line: line) }
        catch let error as EmbedANEError { XCTAssertEqual(error.code, code, file: file, line: line) }
        catch { XCTFail("Expected \(code), received \(type(of: error))", file: file, line: line) }
    }

    private func prepareInterrupted(_ directory: DownloadTestDirectory, bundle: DownloadBundleFixture,
                                    script: DownloadServerScript, server: LoopbackHTTPServer) async throws {
        try directory.write(bundle.spec(endpoint: server.endpoint))
        script.setMode(.interruptFirst)
        await fails("transport_error") {
            _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
        }
        let size = try FileManager.default.attributesOfItem(atPath: directory.partialTable.path)[.size] as? NSNumber
        XCTAssertGreaterThan(size?.intValue ?? 0, 0)
        XCTAssertLessThan(size?.intValue ?? Int.max, bundle.files[DownloadBundleFixture.tablePath]?.count ?? 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.install.path))
        XCTAssertEqual(try String(contentsOf: directory.stage("RESOLVED"), encoding: .utf8), DownloadBundleFixture.commitA + "\n")
    }

    func testHappyPathNestedManifestsAndVerifiedIdempotence() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint))
        let downloader = testDownloader()
        let report = try await downloader.fetch(specPath: directory.spec, modelRoot: directory.root)
        XCTAssertEqual(report.disposition, .installed)
        XCTAssertEqual(report.verification.resolvedCommit, DownloadBundleFixture.commitA)
        XCTAssertEqual(report.verification.artifactDigests.count, fixture.files.count)
        XCTAssertTrue(report.warnings.isEmpty)
        for (path, bytes) in fixture.files {
            XCTAssertEqual(try Data(contentsOf: directory.install.appendingPathComponent(path)), bytes, path)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.stage().path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.install.appendingPathComponent(".embed-ane-download").path))
        let before = server.requests.filter { $0.path.contains("/resolve/") }.count
        let again = try await downloader.fetch(specPath: directory.spec, modelRoot: directory.root)
        XCTAssertEqual(again.disposition, .alreadyInstalled)
        XCTAssertEqual(server.requests.filter { $0.path.contains("/resolve/") }.count, before)
        _ = try await downloader.verify(modelID: fixture.modelID, modelRoot: directory.root)
    }

    /// Regression: the downloader expanded only `chunks/*/manifest.txt`, so the
    /// root-level vision tower's members were never fetched. Every inner
    /// manifest must be expanded, as BundleVerifier does.
    func testRootLevelVisionTowerManifestDownloadsAndVerifiesAllMembers() async throws {
        var fixture = DownloadBundleFixture()
        let towerMembers = ["model.mil": Data("tower-mil".utf8),
                            "weights/weight.bin": Data("tower-weights".utf8),
                            "analytics/coremldata.bin": Data("tower-analytics".utf8)]
        for (path, data) in towerMembers { fixture.files["vision_tower.mlmodelc/" + path] = data }
        fixture.files[ModelSpec.visionTowerManifest] = Data(towerMembers.sorted(by: { $0.key < $1.key }).map {
            "\($0.key) \($0.value.count) \(FileDigest.sha256($0.value))\n"
        }.joined().utf8)
        fixture.files[ModelSpec.visionPositionTable] = Data((0..<4096).map { UInt8($0 % 13) })
        let directory = try DownloadTestDirectory(), script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint, extra: [ModelSpec.visionTowerManifest, ModelSpec.visionPositionTable]))
        let downloader = testDownloader()
        let report = try await downloader.fetch(specPath: directory.spec, modelRoot: directory.root)
        XCTAssertEqual(report.disposition, .installed)
        XCTAssertEqual(report.verification.artifactDigests.count, fixture.files.count)
        for (path, data) in towerMembers {
            let member = "vision_tower.mlmodelc/" + path
            XCTAssertEqual(report.verification.artifactDigests[member], FileDigest.sha256(data), member)
            XCTAssertEqual(try Data(contentsOf: directory.install.appendingPathComponent(member)), data, member)
            XCTAssertTrue(server.requests.contains { ($0.path.removingPercentEncoding ?? $0.path).hasSuffix("/" + member) }, member)
        }
        for (path, bytes) in fixture.files {
            XCTAssertEqual(try Data(contentsOf: directory.install.appendingPathComponent(path)), bytes, path)
        }
        let verified = try await downloader.verify(modelID: fixture.modelID, modelRoot: directory.root)
        XCTAssertEqual(verified.artifactDigests.count, fixture.files.count)
        let spec = try ModelSpec.parse(String(decoding: try Data(contentsOf: directory.spec), as: UTF8.self))
        XCTAssertTrue(spec.isMultimodal)
    }

    func testSHAAndSlashTagResolution() async throws {
        for revision in [DownloadBundleFixture.commitA, "release/one"] {
            let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
            let script = DownloadServerScript(bundle: fixture)
            let server = try LoopbackHTTPServer { script.respond($0) }
            defer { server.stop() }
            try directory.write(fixture.spec(endpoint: server.endpoint, revision: revision))
            _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
            if revision == DownloadBundleFixture.commitA {
                XCTAssertFalse(server.requests.contains { $0.path.hasSuffix("/refs") })
            } else {
                XCTAssertTrue(server.requests.contains { $0.path.contains("release%2Fone") })
            }
            XCTAssertTrue(server.requests.filter { $0.path.contains("/resolve/") }.allSatisfy {
                $0.path.contains("/resolve/" + DownloadBundleFixture.commitA + "/")
            })
        }
    }

    func testMovingBranchRejectedBeforeArtifactGET() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint, revision: "main"))
        await fails("moving_branch") {
            _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
        }
        XCTAssertFalse(server.requests.contains { $0.path.contains("/resolve/") })
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.install.path))
    }

    func testTagMovingDuringResolutionFailsClosed() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        script.setRefsCommit(DownloadBundleFixture.commitB)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint))
        await fails("conflict") {
            _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
        }
        XCTAssertFalse(server.requests.contains { $0.path.contains("/resolve/") })
    }

    func testTransportInterruptionResumesWith206AndKeepsCompleteFiles() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try await prepareInterrupted(directory, bundle: fixture, script: script, server: server)
        script.setMode(.normal)
        _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
        let table = server.requests.filter { $0.path.hasSuffix("/" + DownloadBundleFixture.tablePath) }
        XCTAssertEqual(table.count, 2)
        XCTAssertNotNil(table.last?.headers["range"])
        XCTAssertEqual(table.last?.headers["if-range"], "\"stable\"")
        XCTAssertEqual(server.requests.filter { $0.path.hasSuffix("/tokenizer.json") }.count, 1)
    }

    func testRangeIgnored200RestartsRatherThanAppending() async throws {
        try await assertRestart(mode: .ignoreRange, expectedTableRequests: 2)
    }
    func test416RestartsWithUnconditionalGET() async throws {
        try await assertRestart(mode: .range416, expectedTableRequests: 3)
    }
    func testChangedETagRestartsWithoutMixingRepresentations() async throws {
        try await assertRestart(mode: .changedETag, expectedTableRequests: 3)
    }
    private func assertRestart(mode: DownloadServerScript.Mode, expectedTableRequests: Int) async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try await prepareInterrupted(directory, bundle: fixture, script: script, server: server)
        script.setMode(mode)
        _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
        let requests = server.requests.filter { $0.path.hasSuffix("/" + DownloadBundleFixture.tablePath) }
        XCTAssertEqual(requests.count, expectedTableRequests)
        XCTAssertNotNil(requests.dropFirst().first?.headers["range"])
        if expectedTableRequests == 3 { XCTAssertNil(requests.last?.headers["range"]) }
        XCTAssertEqual(try Data(contentsOf: directory.install.appendingPathComponent(DownloadBundleFixture.tablePath)),
                       fixture.files[DownloadBundleFixture.tablePath])
    }

    func testMalformed206PreservesPrefixAndDoesNotPromote() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try await prepareInterrupted(directory, bundle: fixture, script: script, server: server)
        let prefix = try Data(contentsOf: directory.partialTable)
        script.setMode(.malformedRange)
        await fails("transport_error") {
            _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
        }
        XCTAssertEqual(try Data(contentsOf: directory.partialTable), prefix)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.install.path))
    }

    func testDigestCorruptionPrunesOnlyOffendingFileAndNeverPromotes() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        script.setMode(.corrupt)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint))
        await fails("verification_failed") {
            _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.install.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.partialTable.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.stage("tokenizer.json").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.stage("RESOLVED").path))
        script.setMode(.normal)
        _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
        XCTAssertEqual(server.requests.filter { $0.path.hasSuffix("/tokenizer.json") }.count, 1)
    }

    func testRevisionDriftAndSpecChangeDiscardStaleStage() async throws {
        for changeSHA in [false, true] {
            let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
            let script = DownloadServerScript(bundle: fixture)
            let server = try LoopbackHTTPServer { script.respond($0) }
            defer { server.stop() }
            try await prepareInterrupted(directory, bundle: fixture, script: script, server: server)
            if changeSHA { script.setCommit(DownloadBundleFixture.commitB) }
            else { try directory.write(fixture.spec(endpoint: server.endpoint, revision: "v2")) }
            script.setMode(.normal)
            let report = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
            let last = server.requests.last { $0.path.hasSuffix("/" + DownloadBundleFixture.tablePath) }
            XCTAssertNil(last?.headers["range"])
            XCTAssertEqual(server.requests.filter { $0.path.hasSuffix("/tokenizer.json") }.count, 2)
            XCTAssertEqual(report.verification.resolvedCommit,
                           changeSHA ? DownloadBundleFixture.commitB : DownloadBundleFixture.commitA)
        }
    }

    func testAtomicPromotionNotVisibleUntilLastMemberVerified() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        script.setMode(.pauseLast)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { script.gate.open(); server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint))
        let spec = directory.spec, root = directory.root
        let task = Task { try await testDownloader().fetch(specPath: spec, modelRoot: root) }
        defer { task.cancel() }
        try await eventually { script.gate.isReached }
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.install.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: directory.stage("RESOLVED").path))
        script.gate.open()
        let result = try await task.value
        XCTAssertEqual(result.disposition, .installed)
        _ = try BundleVerifier.verify(at: directory.install, expectedID: fixture.modelID)
    }

    func testOfflineVerificationMakesZeroConnectionsAndIgnoresCredentials() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint, auth: "env:PRIVATE_HF_TOKEN"))
        _ = try await testDownloader(environment: ["PRIVATE_HF_TOKEN": "fixture-secret"]).fetch(specPath: directory.spec, modelRoot: directory.root)
        let connections = server.connections
        let forbidden = RejectNetworkTransport()
        let offline = ModelDownloader(transport: forbidden, environment: { [:] })
        let report = try await offline.fetch(specPath: directory.spec, modelRoot: directory.root, offline: true)
        XCTAssertEqual(report.disposition, .verifiedOffline)
        _ = try await offline.verify(modelID: fixture.modelID, modelRoot: directory.root)
        await fails("invalid_request") {
            _ = try await offline.fetch(specPath: directory.spec, modelRoot: directory.root, offline: true, replace: true)
        }
        XCTAssertEqual(forbidden.calls, 0)
        XCTAssertEqual(server.connections, connections)
        try Data("corrupt".utf8).write(to: directory.install.appendingPathComponent("tokenizer.json"))
        await fails("verification_failed") { _ = try await offline.verify(modelID: fixture.modelID, modelRoot: directory.root) }
        XCTAssertEqual(server.connections, connections)
    }

    func testConcurrentFetchExplicitConflictAndCancellationKeepsPrefix() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        script.setMode(.slowFirst)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { script.gate.open(); server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint))
        let spec = directory.spec, root = directory.root
        let first = Task { try await testDownloader().fetch(specPath: spec, modelRoot: root) }
        defer { first.cancel() }
        try await eventually { script.gate.isReached && self.fileSize(directory.partialTable) > 0 }
        let before = server.connections
        await fails("conflict") { _ = try await testDownloader().fetch(specPath: spec, modelRoot: root) }
        XCTAssertEqual(server.connections, before)
        first.cancel()
        do { _ = try await first.value; XCTFail("Cancelled download succeeded") }
        catch is CancellationError { }
        XCTAssertGreaterThan(fileSize(directory.partialTable), 0)
        script.gate.open()
        script.setMode(.normal)
        _ = try await testDownloader().fetch(specPath: spec, modelRoot: root)
    }

    func testConflictIsNonDestructiveAndReplaceHonorsActiveLease() async throws {
        var fixture = DownloadBundleFixture()
        let directory = try DownloadTestDirectory(), script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint))
        _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root)
        let oldManifest = try Data(contentsOf: directory.install.appendingPathComponent("manifest.yaml"))
        fixture.files["tokenizer.json"] = Data("{\"fixture\":2}".utf8)
        script.setBundle(fixture)
        try directory.write(fixture.spec(endpoint: server.endpoint, revision: "v2"))
        await fails("conflict") { _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root) }
        XCTAssertEqual(try Data(contentsOf: directory.install.appendingPathComponent("manifest.yaml")), oldManifest)
        var lease: ModelUseLease? = try ModelUseLease(modelRoot: directory.root, modelID: fixture.modelID)
        await fails("conflict") {
            _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root, replace: true)
        }
        withExtendedLifetime(lease) {}
        lease = nil
        script.setMode(.corrupt)
        await fails("verification_failed") {
            _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root, replace: true)
        }
        XCTAssertEqual(try Data(contentsOf: directory.install.appendingPathComponent("manifest.yaml")), oldManifest)
        script.setMode(.normal)
        let report = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root, replace: true)
        XCTAssertEqual(report.disposition, .replaced)
        XCTAssertEqual(try Data(contentsOf: directory.install.appendingPathComponent("tokenizer.json")), fixture.files["tokenizer.json"])
    }

    func testUnexpectedChunkFileBlocksPromotion() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try await prepareInterrupted(directory, bundle: fixture, script: script, server: server)
        let unexpected = directory.stage("chunks/chunk0.mlmodelc/unexpected.bin")
        try FileManager.default.createDirectory(at: unexpected.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("not in manifest".utf8).write(to: unexpected)
        script.setMode(.normal)
        await fails("verification_failed") { _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.install.path))
    }

    func testUnsafeInnerManifestBlocksChildFetchAndPromotion() async throws {
        for body in ["../escape 1 \(String(repeating: "a", count: 64))\n",
                     "x 1 \(String(repeating: "a", count: 64))\nx 1 \(String(repeating: "a", count: 64))\n"] {
            var fixture = DownloadBundleFixture()
            fixture.files["chunks/chunk0.mlmodelc/manifest.txt"] = Data(body.utf8)
            let directory = try DownloadTestDirectory(), script = DownloadServerScript(bundle: fixture)
            let server = try LoopbackHTTPServer { script.respond($0) }
            defer { server.stop() }
            try directory.write(fixture.spec(endpoint: server.endpoint))
            do { _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root); XCTFail("Unsafe manifest accepted") }
            catch let error as EmbedANEError { XCTAssertTrue(["unsafe_path", "invalid_spec"].contains(error.code)) }
            XCTAssertFalse(server.requests.contains { $0.path.hasSuffix("/escape") || $0.path.hasSuffix("/x") })
            XCTAssertFalse(FileManager.default.fileExists(atPath: directory.install.path))
        }
    }

    func testPartialSymlinkCannotWriteOutsideStaging() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try await prepareInterrupted(directory, bundle: fixture, script: script, server: server)
        let outside = directory.home.appendingPathComponent("outside")
        try Data("unchanged".utf8).write(to: outside)
        try FileManager.default.removeItem(at: directory.partialTable)
        try FileManager.default.createSymbolicLink(at: directory.partialTable, withDestinationURL: outside)
        script.setMode(.normal)
        await fails("unsafe_path") { _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root) }
        XCTAssertEqual(try Data(contentsOf: outside), Data("unchanged".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.install.path))
    }

    func testProductionRejectsHTTPAndMissingAuthWithoutConnections() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint))
        await fails("invalid_spec") { _ = try await ModelDownloader().fetch(specPath: directory.spec, modelRoot: directory.root) }
        try directory.write(fixture.spec(endpoint: server.endpoint, auth: "env:MISSING_TOKEN"))
        await fails("invalid_spec") { _ = try await testDownloader().fetch(specPath: directory.spec, modelRoot: directory.root) }
        XCTAssertEqual(server.connections, 0)
    }

    func testBearerPresentOnHubButRemovedFromCrossOriginRedirect() async throws {
        let destination = try LoopbackHTTPServer { _ in SocketReply(body: Data("payload".utf8)) }
        defer { destination.stop() }
        let endpoint = destination.endpoint
        let origin = try LoopbackHTTPServer { _ in SocketReply(status: 302, headers: ["Location": endpoint + "/blob?signature=fixture"]) }
        defer { origin.stop() }
        let url = try XCTUnwrap(URL(string: origin.endpoint + "/file"))
        var request = URLRequest(url: url)
        request.setValue("Bearer secret-fixture", forHTTPHeaderField: "Authorization")
        request.setValue("session=private", forHTTPHeaderField: "Cookie")
        let bytes = TestDataCollector()
        try await URLSessionDownloadTransport().execute(request, policy: .loopbackHTTPForTests,
            receiveResponse: { response in
                guard response.status == 200 else { throw SocketFixtureError.fixture }
            }, receiveData: { bytes.append($0) })
        XCTAssertEqual(bytes.data, Data("payload".utf8))
        XCTAssertEqual(origin.requests.first?.headers["authorization"], "Bearer secret-fixture")
        XCTAssertNil(destination.requests.first?.headers["authorization"])
        XCTAssertNil(destination.requests.first?.headers["cookie"])
    }

    func testHTTPFailureNamesRevisionButDoesNotLeakTokenOrServerBody() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let secret = "private-fixture-token"
        let server = try LoopbackHTTPServer { _ in
            SocketReply(status: 403, body: Data(("rejected credential: " + secret).utf8))
        }
        defer { server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint, auth: "env:HF_TOKEN"))
        do {
            _ = try await testDownloader(environment: ["HF_TOKEN": secret]).fetch(specPath: directory.spec, modelRoot: directory.root)
            XCTFail("Expected HTTP failure")
        } catch let error as EmbedANEError {
            XCTAssertEqual(error.code, "transport_error")
            XCTAssertTrue(error.message.contains("source.revision"))
            XCTAssertFalse(error.message.contains(secret))
            XCTAssertFalse(error.message.contains("rejected credential"))
        }
        XCTAssertEqual(server.requests.first?.headers["authorization"], "Bearer " + secret)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.install.path))
    }

    func testLoopbackExceptionDoesNotPermitRedirectToLocalhost() async throws {
        let destination = try LoopbackHTTPServer { _ in SocketReply(body: Data("must not be read".utf8)) }
        defer { destination.stop() }
        let location = "http://localhost:\(destination.port)/file"
        let origin = try LoopbackHTTPServer { _ in SocketReply(status: 302, headers: ["Location": location]) }
        defer { origin.stop() }
        let url = try XCTUnwrap(URL(string: origin.endpoint + "/file"))
        await fails("invalid_spec") {
            try await URLSessionDownloadTransport().execute(URLRequest(url: url), policy: .loopbackHTTPForTests,
                receiveResponse: { _ in }, receiveData: { _ in })
        }
        XCTAssertEqual(destination.connections, 0)
    }

    func testSubprocessSIGKILLAndRelaunchResumesFromPersistedPrefix() async throws {
        let fixture = DownloadBundleFixture(), directory = try DownloadTestDirectory()
        let script = DownloadServerScript(bundle: fixture)
        script.setMode(.slowFirst)
        let server = try LoopbackHTTPServer { script.respond($0) }
        defer { script.gate.open(); server.stop() }
        try directory.write(fixture.spec(endpoint: server.endpoint))
        let first = try launchChild(spec: directory.spec, root: directory.root)
        defer { if first.isRunning { _ = kill(first.processIdentifier, SIGKILL) } }
        try await eventually { script.gate.isReached && self.fileSize(directory.partialTable) > 0 }
        let interruptedSize = fileSize(directory.partialTable)
        XCTAssertGreaterThan(interruptedSize, 0)
        XCTAssertLessThan(interruptedSize, 2 * 1024 * 1024)
        XCTAssertEqual(kill(first.processIdentifier, SIGKILL), 0)
        try await eventually { !first.isRunning }
        XCTAssertEqual(first.terminationReason, .uncaughtSignal)
        XCTAssertEqual(first.terminationStatus, SIGKILL)
        script.gate.open()
        script.setMode(.normal)
        let second = try launchChild(spec: directory.spec, root: directory.root)
        defer { if second.isRunning { _ = kill(second.processIdentifier, SIGKILL) } }
        try await eventually(timeout: 20) { !second.isRunning }
        XCTAssertEqual(second.terminationStatus, 0)
        _ = try BundleVerifier.verify(at: directory.install, expectedID: fixture.modelID)
        let requests = server.requests.filter { $0.path.hasSuffix("/" + DownloadBundleFixture.tablePath) }
        XCTAssertNotNil(requests.last?.headers["range"])
        XCTAssertEqual(server.requests.filter { $0.path.hasSuffix("/tokenizer.json") }.count, 1)
    }

    /// Re-executed by xctest in a genuinely separate process. No executable
    /// product or Package.swift modification is needed for this fixture.
    func testSubprocessFetchEntryPoint() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let spec = environment["EMBED_ANE_S2_CHILD_SPEC"],
              let root = environment["EMBED_ANE_S2_CHILD_ROOT"] else {
            throw XCTSkip("Subprocess fixture entry point; invoked by the SIGKILL/relaunch test")
        }
        _ = try await testDownloader().fetch(specPath: URL(fileURLWithPath: spec), modelRoot: URL(fileURLWithPath: root))
    }

    private func launchChild(spec: URL, root: URL) throws -> Process {
        let process = Process()
        // Resolve the executable, then launch it directly. The PID killed by
        // the parent is therefore the xctest/downloader, not a shell or wrapper.
        let resolver = Process()
        let output = Pipe()
        resolver.executableURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        resolver.arguments = ["--find", "xctest"]
        resolver.standardOutput = output
        resolver.standardError = FileHandle.nullDevice
        try resolver.run()
        let executable = String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        resolver.waitUntilExit()
        guard resolver.terminationStatus == 0, let executable, executable.hasPrefix("/") else {
            throw SocketFixtureError.fixture
        }
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = ["-XCTest",
            "EmbedANEDownloadTests.DownloaderSocketTests/testSubprocessFetchEntryPoint",
            Bundle(for: DownloaderSocketTests.self).bundleURL.path]
        var environment = ProcessInfo.processInfo.environment
        environment["EMBED_ANE_S2_CHILD_SPEC"] = spec.path
        environment["EMBED_ANE_S2_CHILD_ROOT"] = root.path
        environment.removeValue(forKey: "EMBED_ANE_MODEL_ROOT")
        process.environment = environment
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        return process
    }

    private func fileSize(_ url: URL) -> Int {
        ((try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.intValue ?? 0
    }
}

private final class RejectNetworkTransport: DownloadTransport, @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    var calls: Int { lock.withLock { count } }
    func execute(_ request: URLRequest, policy: DownloadTransportPolicy,
                 receiveResponse: @escaping @Sendable (DownloadResponse) throws -> Void,
                 receiveData: @escaping @Sendable (Data) throws -> Void) async throws {
        lock.withLock { count += 1 }
        throw SocketFixtureError.fixture
    }
}

private final class TestDataCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = Data()
    var data: Data { lock.withLock { storage } }
    func append(_ data: Data) { lock.withLock { storage.append(data) } }
}
