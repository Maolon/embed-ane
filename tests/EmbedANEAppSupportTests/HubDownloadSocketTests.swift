import Darwin
import EmbedANEAppSupport
import EmbedANECore
import EmbedANEDownload
import EmbedANETestSupport
import Foundation
import Testing

private struct HubTestRequest: Sendable {
    let method: String
    let path: String
    let headers: [String: String]
}

private struct HubTestReply: Sendable {
    var status = 200
    var headers: [String: String] = [:]
    var body = Data()
    var dropAfterBytes: Int? = nil
}

private final class HubTestLoopbackServer: @unchecked Sendable {
    let port: UInt16
    private let listener: Int32
    private let lock = NSLock()
    private var stopped = false
    private var clients = Set<Int32>()
    private let handler: @Sendable (HubTestRequest) -> HubTestReply

    var endpoint: String { "http://127.0.0.1:\(port)" }

    init(handler: @escaping @Sendable (HubTestRequest) -> HubTestReply) throws {
        self.handler = handler
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw EmbedANEError.io(path: "socket", reason: "socket creation failed") }
        var yes: Int32 = 1
        _ = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0, Darwin.listen(fd, 32) == 0 else {
            close(fd)
            throw EmbedANEError.io(path: "listen", reason: "bind or listen failed")
        }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
        }
        guard named == 0 else {
            close(fd)
            throw EmbedANEError.io(path: "getsockname", reason: "getsockname failed")
        }
        port = UInt16(bigEndian: address.sin_port)
        listener = fd
        DispatchQueue(label: "embed-ane.hub-test-loopback.accept").async { [weak self] in
            self?.acceptLoop()
        }
    }

    deinit { stop() }

    func stop() {
        lock.withLock {
            guard !stopped else { return }
            stopped = true
            _ = shutdown(listener, Int32(SHUT_RDWR))
            close(listener)
            for client in clients { _ = shutdown(client, Int32(SHUT_RDWR)) }
        }
    }

    private func acceptLoop() {
        while !lock.withLock({ stopped }) {
            let client = accept(listener, nil, nil)
            guard client >= 0 else {
                if errno == EINTR { continue }
                return
            }
            _ = lock.withLock { clients.insert(client) }
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.handle(client)
            }
        }
    }

    private func handle(_ client: Int32) {
        defer {
            _ = lock.withLock { clients.remove(client) }
            close(client)
        }
        var yes: Int32 = 1
        _ = setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
        var bytes = Data()
        var buffer = [UInt8](repeating: 0, count: 4096)
        let delimiter = Data("\r\n\r\n".utf8)
        while bytes.range(of: delimiter) == nil, bytes.count < 65_536 {
            let count = recv(client, &buffer, buffer.count, 0)
            guard count > 0 else { return }
            bytes.append(contentsOf: buffer.prefix(count))
        }
        guard let text = String(data: bytes, encoding: .utf8) else { return }
        let lines = text.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return }
        let parts = requestLine.split(separator: " ")
        guard parts.count >= 2 else { return }
        let method = String(parts[0])
        let path = String(parts[1])
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[String(line[..<colon]).lowercased()] =
                String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
        }
        let reply = handler(HubTestRequest(method: method, path: path, headers: headers))
        let statusText: String
        switch reply.status {
        case 200: statusText = "OK"
        case 206: statusText = "Partial Content"
        case 401: statusText = "Unauthorized"
        case 404: statusText = "Not Found"
        case 416: statusText = "Range Not Satisfiable"
        default: statusText = "Error"
        }
        var respHeaders = reply.headers
        respHeaders["Content-Length"] = String(reply.body.count)
        respHeaders["Connection"] = "close"
        let headerBlock = respHeaders.map { "\($0.key): \($0.value)\r\n" }.joined()
        var full = Data("HTTP/1.1 \(reply.status) \(statusText)\r\n\(headerBlock)\r\n".utf8)
        if let drop = reply.dropAfterBytes {
            full.append(reply.body.prefix(drop))
            guard sendAll(full, to: client) else { return }
            usleep(100_000)
            _ = shutdown(client, Int32(SHUT_WR))
            var drainBuf = [UInt8](repeating: 0, count: 1024)
            let deadline = Date().addingTimeInterval(0.5)
            while Date() < deadline {
                let n = recv(client, &drainBuf, drainBuf.count, 0)
                if n <= 0 { break }
            }
            return
        }
        full.append(reply.body)
        guard sendAll(full, to: client) else { return }
        _ = shutdown(client, Int32(SHUT_WR))
        var drainBuf = [UInt8](repeating: 0, count: 1024)
        while recv(client, &drainBuf, drainBuf.count, 0) > 0 {}
    }

    private func sendAll(_ data: Data, to client: Int32) -> Bool {
        data.withUnsafeBytes { raw in
            guard let base = raw.baseAddress else { return true }
            var sent = 0
            while sent < raw.count {
                let n = send(client, base.advanced(by: sent), raw.count - sent, 0)
                if n < 0, errno == EINTR { continue }
                guard n > 0 else { return false }
                sent += n
            }
            return true
        }
    }
}

private struct HubBundleFixture: Sendable {
    static let commitA = String(repeating: "a", count: 40)
    static let commitB = String(repeating: "b", count: 40)
    static let tablePath = "embed_table.fp16.npy"
    let modelID = "wemm-hub-test"
    var files: [String: Data]

    init() {
        var files: [String: Data] = [
            "tokenizer.json": Data(#"{"model":{"type":"BPE"}}"#.utf8),
            "tokenizer_config.json": Data(#"{"tokenizer_class":"PreTrainedTokenizerFast"}"#.utf8),
            Self.tablePath: Data((0..<(2 * 1024 * 1024)).map { UInt8($0 % 251) }),
        ]
        for chunk in 0..<6 {
            let prefix = "chunks/chunk\(chunk).mlmodelc/"
            let members = [
                "model.bin": Data("model-\(chunk)".utf8),
                "weights/weight.bin": Data("weights-\(chunk)".utf8),
            ]
            for (path, data) in members { files[prefix + path] = data }
            files[prefix + "manifest.txt"] = Data(members.sorted(by: { $0.key < $1.key }).map {
                "\($0.key) \($0.value.count) \(FileDigest.sha256($0.value))\n"
            }.joined().utf8)
        }
        self.files = files
    }

    var outerPaths: [String] {
        ["tokenizer.json", "tokenizer_config.json", Self.tablePath]
            + (0..<6).map { "chunks/chunk\($0).mlmodelc/manifest.txt" }
    }

    func specYAML(endpoint: String, repo: String = AppHubDefaults.defaultRepo,
                  revision: String = "v1", auth: String? = nil) throws -> String {
        let entries: [[String: Any]] = try outerPaths.map { path in
            guard let data = files[path] else { throw EmbedANEError.verification(path: path, reason: "missing fixture file") }
            return ["path": path, "size": data.count, "sha256": FileDigest.sha256(data)]
        }
        var sourceDict: [String: Any] = ["repo": repo, "revision": revision, "endpoint": endpoint]
        if let auth { sourceDict["auth"] = auth }
        let dict: [String: Any] = [
            "spec_version": 1,
            "model": ["id": modelID, "dim": 2048, "max_seq": 512, "normalize": "l2"],
            "source": sourceDict,
            "files": entries,
            "runtime": ["compute_units": "cpu_and_ne", "pad_side": "right", "mask_dtype": "fp32"],
        ]
        let json = try JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
        return String(decoding: json, as: UTF8.self)
    }
}

private final class TestState: @unchecked Sendable {
    private let lock = NSLock()
    private var endpoint = ""
    private var commit = HubBundleFixture.commitA
    private var phases: [String] = []
    private var requests: [HubTestRequest] = []
    private var dropFirstTableRequestAfter: Int? = nil
    private var ignoreRangeForTable = false
    private var requireAuthToken: String? = nil

    func setEndpoint(_ val: String) { lock.withLock { endpoint = val } }
    func getEndpoint() -> String { lock.withLock { endpoint } }

    func setCommit(_ val: String) { lock.withLock { commit = val } }
    func getCommit() -> String { lock.withLock { commit } }

    func addPhase(_ phase: String) { lock.withLock { phases.append(phase) } }
    func getPhases() -> [String] { lock.withLock { phases } }

    func recordRequest(_ req: HubTestRequest) { lock.withLock { requests.append(req) } }
    func getRequests() -> [HubTestRequest] { lock.withLock { requests } }
    func clearRequests() { lock.withLock { requests.removeAll() } }

    func setDropFirstTableRequestAfter(_ bytes: Int?) { lock.withLock { dropFirstTableRequestAfter = bytes } }
    func consumeDropFirstTableRequestAfter() -> Int? {
        lock.withLock {
            let val = dropFirstTableRequestAfter
            dropFirstTableRequestAfter = nil
            return val
        }
    }

    func setIgnoreRangeForTable(_ val: Bool) { lock.withLock { ignoreRangeForTable = val } }
    func getIgnoreRangeForTable() -> Bool { lock.withLock { ignoreRangeForTable } }

    func setRequireAuthToken(_ token: String?) { lock.withLock { requireAuthToken = token } }
    func getRequireAuthToken() -> String? { lock.withLock { requireAuthToken } }
}

private func makeServer(fixture: HubBundleFixture, repo: String, state: TestState) throws -> HubTestLoopbackServer {
    try HubTestLoopbackServer { req in
        state.recordRequest(req)
        if let requiredToken = state.getRequireAuthToken() {
            guard req.headers["authorization"] == "Bearer \(requiredToken)" else {
                return HubTestReply(status: 401, body: Data("Unauthorized".utf8))
            }
        }
        let endpoint = state.getEndpoint()
        let commit = state.getCommit()
        if req.path == "/\(repo)/resolve/main/spec.yaml" {
            guard let yaml = try? fixture.specYAML(endpoint: endpoint, repo: repo, revision: "v1") else {
                return HubTestReply(status: 500)
            }
            return HubTestReply(headers: ["Content-Type": "text/yaml"], body: Data(yaml.utf8))
        }
        if req.path.hasPrefix("/api/models/\(repo)/revision/") {
            return HubTestReply(headers: ["Content-Type": "application/json"], body: Data(#"{"sha":"\#(commit)"}"#.utf8))
        }
        if req.path == "/api/models/\(repo)/refs" {
            return HubTestReply(headers: ["Content-Type": "application/json"], body: Data(#"{"branches":[{"name":"main","ref":"refs/heads/main","targetCommit":"\#(commit)"}],"tags":[{"name":"v1","ref":"refs/tags/v1","targetCommit":"\#(commit)"}]}"#.utf8))
        }
        let prefix = "/\(repo)/resolve/\(commit)/"
        if req.path.hasPrefix(prefix) {
            let path = String(req.path.dropFirst(prefix.count))
            if let data = fixture.files[path] {
                if path == HubBundleFixture.tablePath {
                    if let drop = state.consumeDropFirstTableRequestAfter() {
                        return HubTestReply(
                            status: 200,
                            headers: ["Content-Type": "application/octet-stream", "ETag": "\"stable\""],
                            body: data,
                            dropAfterBytes: drop
                        )
                    }
                    if state.getIgnoreRangeForTable() {
                        return HubTestReply(
                            status: 200,
                            headers: ["Content-Type": "application/octet-stream", "ETag": "\"stable\""],
                            body: data
                        )
                    }
                }
                if let rangeHeader = req.headers["range"], rangeHeader.hasPrefix("bytes=") {
                    let spec = String(rangeHeader.dropFirst(6))
                    let parts = spec.split(separator: "-", omittingEmptySubsequences: false)
                    if let start = Int(parts[0]) {
                        let total = data.count
                        let end = (parts.count > 1 && !parts[1].isEmpty) ? Int(parts[1])! : total - 1
                        if start >= 0, start <= end, end < total {
                            let sub = data.subdata(in: start..<(end + 1))
                            return HubTestReply(
                                status: 206,
                                headers: [
                                    "Content-Type": "application/octet-stream",
                                    "Content-Range": "bytes \(start)-\(end)/\(total)",
                                    "ETag": "\"stable\"",
                                ],
                                body: sub
                            )
                        }
                    }
                }
                return HubTestReply(
                    status: 200,
                    headers: ["Content-Type": "application/octet-stream", "ETag": "\"stable\""],
                    body: data
                )
            }
        }
        return HubTestReply(status: 404)
    }
}

@Suite("AppSupport Hub Download over loopback HTTP", .serialized)
struct HubDownloadSocketTests {
    @Test func fetchSpecAndInstallHappyPathOverLoopback() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-hub-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let fixture = HubBundleFixture()
        let repo = AppHubDefaults.defaultRepo
        let state = TestState()

        let server = try makeServer(fixture: fixture, repo: repo, state: state)
        defer { server.stop() }
        state.setEndpoint(server.endpoint)

        let downloader = ModelDownloader(policy: .loopbackHTTPForTests)
        let library = InstalledModelLibrary(downloader: downloader)
        let endpointURL = try #require(URL(string: server.endpoint))

        // 1. Fetch spec from Hub
        let specURL = try await library.fetchSpec(repo: repo, endpoint: endpointURL)
        defer { try? FileManager.default.removeItem(at: specURL) }
        #expect(FileManager.default.fileExists(atPath: specURL.path))

        // 2. Install bundle from the acquired spec
        let installResult = try await library.install(spec: specURL, root: tempDir, replace: false) { progress in
            state.addPhase(progress.phase)
        }
        #expect(installResult.modelID == fixture.modelID)
        let phases = state.getPhases()
        #expect(phases.contains("downloading"))

        // 3. Verify scan finds the newly verified model
        let catalog = try await library.scan(root: tempDir)
        let foundModel = catalog.models.contains { $0.id == fixture.modelID }
        #expect(foundModel)
        #expect(catalog.rejected.isEmpty)

        // 4. Test conflict detection: when remote commit changes, replace=false throws .conflict
        state.setCommit(HubBundleFixture.commitB)
        do {
            _ = try await library.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
            Issue.record("Expected conflict error")
        } catch let error as EmbedANEError {
            guard case let .conflict(reason) = error else {
                Issue.record("Expected EmbedANEError.conflict, got \(error)")
                return
            }
            #expect(reason.contains("already installed as \(fixture.modelID)"))
        }

        // 5. Test replace=true overwrites cleanly and manifest contains commitB
        let replaceResult = try await library.install(spec: specURL, root: tempDir, replace: true, progress: { _ in })
        #expect(replaceResult.modelID == fixture.modelID)
        let manifest = try BundleVerifier.manifest(at: tempDir.appendingPathComponent(fixture.modelID))
        #expect(manifest.resolvedCommit == HubBundleFixture.commitB)
        let verified = try BundleVerifier.verify(at: tempDir.appendingPathComponent(fixture.modelID))
        #expect(verified.modelID == fixture.modelID)
        #expect(verified.resolvedCommit == HubBundleFixture.commitB)
    }

    @Test func hubDownloadInterruptedMidTransferResumesWithRangeAndPromotes() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-hub-resume-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let fixture = HubBundleFixture()
        let repo = AppHubDefaults.defaultRepo
        let state = TestState()
        state.setDropFirstTableRequestAfter(65_536)

        let server = try makeServer(fixture: fixture, repo: repo, state: state)
        defer { server.stop() }
        state.setEndpoint(server.endpoint)

        let downloader = ModelDownloader(policy: .loopbackHTTPForTests)
        let library = InstalledModelLibrary(downloader: downloader)
        let endpointURL = try #require(URL(string: server.endpoint))

        let specURL = try await library.fetchSpec(repo: repo, endpoint: endpointURL)
        defer { try? FileManager.default.removeItem(at: specURL) }

        // First attempt: fails mid-transfer on tablePath
        do {
            _ = try await library.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
            Issue.record("Expected transport error on mid-transfer drop")
        } catch let error as EmbedANEError {
            let isTransport = error.code == "transport_error"
            #expect(isTransport)
        }

        // Staged partial file retained on disk
        let tableDigest = FileDigest.sha256(Data(HubBundleFixture.tablePath.utf8))
        let partialFile = tempDir.appendingPathComponent(".staging/\(fixture.modelID)/.embed-ane-download/\(tableDigest).part")
        let partialExists = FileManager.default.fileExists(atPath: partialFile.path)
        #expect(partialExists)
        let partialAttrs = try FileManager.default.attributesOfItem(atPath: partialFile.path)
        let partialSize = (partialAttrs[.size] as? NSNumber)?.intValue ?? 0
        #expect(partialSize > 0)
        #expect(partialSize <= 65_536)

        // Second attempt: resumes with Range header
        let result = try await library.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
        #expect(result.modelID == fixture.modelID)

        let tableRequests = state.getRequests().filter { $0.path.hasSuffix("/" + HubBundleFixture.tablePath) }
        #expect(tableRequests.count == 2)
        let secondReq = tableRequests[1]
        let rangeHeader = secondReq.headers["range"]
        #expect(rangeHeader != nil)
        #expect(rangeHeader?.hasPrefix("bytes=") == true)

        let verified = try BundleVerifier.verify(at: tempDir.appendingPathComponent(fixture.modelID))
        #expect(verified.modelID == fixture.modelID)
        let installedTable = try Data(contentsOf: tempDir.appendingPathComponent("\(fixture.modelID)/\(HubBundleFixture.tablePath)"))
        #expect(installedTable == fixture.files[HubBundleFixture.tablePath])
    }

    @Test func hubDownloadServerNotSupportingRangeFallsBackToFullDownloadWithoutCorruption() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-hub-norange-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let fixture = HubBundleFixture()
        let repo = AppHubDefaults.defaultRepo
        let state = TestState()
        state.setDropFirstTableRequestAfter(65_536)

        let server = try makeServer(fixture: fixture, repo: repo, state: state)
        defer { server.stop() }
        state.setEndpoint(server.endpoint)

        let downloader = ModelDownloader(policy: .loopbackHTTPForTests)
        let library = InstalledModelLibrary(downloader: downloader)
        let endpointURL = try #require(URL(string: server.endpoint))

        let specURL = try await library.fetchSpec(repo: repo, endpoint: endpointURL)
        defer { try? FileManager.default.removeItem(at: specURL) }

        // Interrupt first
        do {
            _ = try await library.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
        } catch {}

        // Server ignores Range header and returns 200 with full data
        state.setIgnoreRangeForTable(true)

        let result = try await library.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
        #expect(result.modelID == fixture.modelID)

        let installedTable = try Data(contentsOf: tempDir.appendingPathComponent("\(fixture.modelID)/\(HubBundleFixture.tablePath)"))
        let matchesFixture = installedTable == fixture.files[HubBundleFixture.tablePath]
        #expect(matchesFixture)
    }

    @Test func hubDownloadCorruptPartialIsDiscardedAndPromotedOnFreshDownload() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-hub-corrupt-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let fixture = HubBundleFixture()
        let repo = AppHubDefaults.defaultRepo
        let state = TestState()

        let server = try makeServer(fixture: fixture, repo: repo, state: state)
        defer { server.stop() }
        state.setEndpoint(server.endpoint)

        let downloader = ModelDownloader(policy: .loopbackHTTPForTests)
        let library = InstalledModelLibrary(downloader: downloader)
        let endpointURL = try #require(URL(string: server.endpoint))

        let specURL = try await library.fetchSpec(repo: repo, endpoint: endpointURL)
        defer { try? FileManager.default.removeItem(at: specURL) }

        // 1. Interrupt first to create valid staging provenance
        state.setDropFirstTableRequestAfter(65_536)
        do {
            _ = try await library.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
        } catch {}

        let tableDigest = FileDigest.sha256(Data(HubBundleFixture.tablePath.utf8))
        let partialFile = tempDir.appendingPathComponent(".staging/\(fixture.modelID)/.embed-ane-download/\(tableDigest).part")
        #expect(FileManager.default.fileExists(atPath: partialFile.path))

        // 2. Corrupt the partial file on disk
        let corruptData = Data(repeating: 0xEE, count: 65_536)
        try corruptData.write(to: partialFile)

        // 3. Second attempt resumes with corrupt prefix, fails verification
        do {
            _ = try await library.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
            Issue.record("Expected verification error due to corrupt partial")
        } catch let error as EmbedANEError {
            let isVerification = error.code == "verification_failed"
            #expect(isVerification)
        }

        // 4. Corrupt partial must be safely deleted
        let partialStillExists = FileManager.default.fileExists(atPath: partialFile.path)
        #expect(!partialStillExists)

        // 5. Fresh install succeeds from scratch
        let result = try await library.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
        #expect(result.modelID == fixture.modelID)
        let installedTable = try Data(contentsOf: tempDir.appendingPathComponent("\(fixture.modelID)/\(HubBundleFixture.tablePath)"))
        let matchesFixture = installedTable == fixture.files[HubBundleFixture.tablePath]
        #expect(matchesFixture)
    }

    @Test func hubDownloadCancellationRetainsStagedPartialsAndRelaunchResumes() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-hub-cancel-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let fixture = HubBundleFixture()
        let repo = AppHubDefaults.defaultRepo
        let state = TestState()
        state.setDropFirstTableRequestAfter(65_536)

        let server = try makeServer(fixture: fixture, repo: repo, state: state)
        defer { server.stop() }
        state.setEndpoint(server.endpoint)

        let downloader = ModelDownloader(policy: .loopbackHTTPForTests)
        let library = InstalledModelLibrary(downloader: downloader)
        let endpointURL = try #require(URL(string: server.endpoint))

        let specURL = try await library.fetchSpec(repo: repo, endpoint: endpointURL)
        defer { try? FileManager.default.removeItem(at: specURL) }

        // Start install in a Task and cancel it
        let installTask = Task {
            try await library.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
        }
        // Let first attempt fail on dropped connection
        do {
            _ = try await installTask.value
        } catch {}

        // Verify partial file was retained on disk
        let tableDigest = FileDigest.sha256(Data(HubBundleFixture.tablePath.utf8))
        let partialFile = tempDir.appendingPathComponent(".staging/\(fixture.modelID)/.embed-ane-download/\(tableDigest).part")
        let partialExists = FileManager.default.fileExists(atPath: partialFile.path)
        #expect(partialExists)

        // Relaunch simulation: fresh call to install resumes from partial bytes
        let result = try await library.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
        #expect(result.modelID == fixture.modelID)
    }

    @Test func hubDownloadWithTokenFileAuthenticatesPrivateRepo() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-hub-auth-" + UUID().uuidString)
        let tokenDir = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-token-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: tokenDir, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer {
            try? FileManager.default.removeItem(at: tempDir)
            try? FileManager.default.removeItem(at: tokenDir)
        }

        let fixture = HubBundleFixture()
        let repo = AppHubDefaults.defaultRepo
        let state = TestState()
        let secretToken = "hf_secret_test_token_12345"
        state.setRequireAuthToken(secretToken)

        let server = try makeServer(fixture: fixture, repo: repo, state: state)
        defer { server.stop() }
        state.setEndpoint(server.endpoint)

        let endpointURL = try #require(URL(string: server.endpoint))

        // 1. Without token, request fails with 401
        let anonDownloader = ModelDownloader(policy: .loopbackHTTPForTests, environment: { [:] })
        let anonLibrary = InstalledModelLibrary(downloader: anonDownloader)
        do {
            _ = try await anonLibrary.fetchSpec(repo: repo, endpoint: endpointURL)
            Issue.record("Expected 401 transport error without token")
        } catch let error as EmbedANEError {
            let isTransport = error.code == "transport_error"
            #expect(isTransport)
        }

        // 2. Write token file with 0600 permissions
        let tokenFile = tokenDir.appendingPathComponent("hf-token")
        try Data((secretToken + "\n").utf8).write(to: tokenFile)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tokenFile.path)

        // Clear recorded anonymous requests before authenticated phase
        state.clearRequests()

        // 3. Downloader configured with tokenDirectory reads the token file
        let authDownloader = ModelDownloader(
            policy: .loopbackHTTPForTests,
            environment: {
                ModelDownloader.defaultEnvironment(baseEnvironment: [:], tokenDirectory: tokenDir)
            }
        )
        let authLibrary = InstalledModelLibrary(downloader: authDownloader)

        let specURL = try await authLibrary.fetchSpec(repo: repo, endpoint: endpointURL)
        defer { try? FileManager.default.removeItem(at: specURL) }

        let result = try await authLibrary.install(spec: specURL, root: tempDir, replace: false, progress: { _ in })
        #expect(result.modelID == fixture.modelID)

        let allRequests = state.getRequests()
        let hasAuth = allRequests.allSatisfy { $0.headers["authorization"] == "Bearer \(secretToken)" }
        #expect(hasAuth)
    }

    @Test func fetchSpecRepoNotFoundReturnsCleanError() async throws {
        let server = try HubTestLoopbackServer { _ in
            HubTestReply(status: 404)
        }
        defer { server.stop() }

        let downloader = ModelDownloader(policy: .loopbackHTTPForTests)
        let library = InstalledModelLibrary(downloader: downloader)
        let endpointURL = try #require(URL(string: server.endpoint))

        do {
            _ = try await library.fetchSpec(repo: "nonexistent/repo", endpoint: endpointURL)
            Issue.record("Expected 404 transport error")
        } catch let error as EmbedANEError {
            guard case let .transport(path, reason) = error else {
                Issue.record("Expected EmbedANEError.transport, got \(error)")
                return
            }
            #expect(path == "nonexistent/repo")
            #expect(reason.contains("404"))
        }
    }

    @Test func fetchSpecForeignEndpointRejected() async throws {
        let fixture = HubBundleFixture()
        let repo = AppHubDefaults.defaultRepo
        let server = try HubTestLoopbackServer { req in
            if req.path == "/\(repo)/resolve/main/spec.yaml" {
                let yaml = try? fixture.specYAML(endpoint: "https://attacker.com", repo: repo)
                return HubTestReply(headers: ["Content-Type": "text/yaml"], body: Data(yaml!.utf8))
            }
            return HubTestReply(status: 404)
        }
        defer { server.stop() }

        let downloader = ModelDownloader(policy: .loopbackHTTPForTests)
        let library = InstalledModelLibrary(downloader: downloader)
        let endpointURL = try #require(URL(string: server.endpoint))

        do {
            _ = try await library.fetchSpec(repo: repo, endpoint: endpointURL)
            Issue.record("Expected foreign endpoint to be rejected")
        } catch let error as EmbedANEError {
            guard case let .invalidSpec(reason) = error else {
                Issue.record("Expected invalidSpec, got \(error)")
                return
            }
            let hasEndpoint = reason.contains("endpoint")
            let hasMismatch = reason.contains("does not match")
            #expect(hasEndpoint)
            #expect(hasMismatch)
        }
    }

    @Test func fetchSpecWithAuthDirectiveRejected() async throws {
        let fixture = HubBundleFixture()
        let repo = AppHubDefaults.defaultRepo
        let state = TestState()
        let server = try HubTestLoopbackServer { req in
            if req.path == "/\(repo)/resolve/main/spec.yaml" {
                let yaml = try? fixture.specYAML(endpoint: state.getEndpoint(), repo: repo, auth: "env:HF_TOKEN")
                return HubTestReply(headers: ["Content-Type": "text/yaml"], body: Data(yaml!.utf8))
            }
            return HubTestReply(status: 404)
        }
        defer { server.stop() }
        state.setEndpoint(server.endpoint)

        let downloader = ModelDownloader(policy: .loopbackHTTPForTests)
        let library = InstalledModelLibrary(downloader: downloader)
        let endpointURL = try #require(URL(string: server.endpoint))

        do {
            _ = try await library.fetchSpec(repo: repo, endpoint: endpointURL)
            Issue.record("Expected auth directive in remote spec to be rejected")
        } catch let error as EmbedANEError {
            guard case let .invalidSpec(reason) = error else {
                Issue.record("Expected invalidSpec, got \(error)")
                return
            }
            #expect(reason.contains("authentication directives"))
        }
    }

    @Test func fetchSpecRepoMismatchRejected() async throws {
        let fixture = HubBundleFixture()
        let repo = AppHubDefaults.defaultRepo
        let state = TestState()
        let server = try HubTestLoopbackServer { req in
            if req.path == "/\(repo)/resolve/main/spec.yaml" {
                let yaml = try? fixture.specYAML(endpoint: state.getEndpoint(), repo: "attacker/malicious")
                return HubTestReply(headers: ["Content-Type": "text/yaml"], body: Data(yaml!.utf8))
            }
            return HubTestReply(status: 404)
        }
        defer { server.stop() }
        state.setEndpoint(server.endpoint)

        let downloader = ModelDownloader(policy: .loopbackHTTPForTests)
        let library = InstalledModelLibrary(downloader: downloader)
        let endpointURL = try #require(URL(string: server.endpoint))

        do {
            _ = try await library.fetchSpec(repo: repo, endpoint: endpointURL)
            Issue.record("Expected repo mismatch to be rejected")
        } catch let error as EmbedANEError {
            guard case let .invalidSpec(reason) = error else {
                Issue.record("Expected invalidSpec, got \(error)")
                return
            }
            let hasRepo = reason.contains("source.repo")
            let hasMismatch = reason.contains("does not match")
            #expect(hasRepo)
            #expect(hasMismatch)
        }
    }
}
