import Foundation
import EmbedANECore
import EmbedANEDownload

struct DownloadBundleFixture: Sendable {
    static let commitA = String(repeating: "a", count: 40)
    static let commitB = String(repeating: "b", count: 40)
    static let tablePath = "embed_table.fp16.npy"
    let modelID = "fixture-model"
    var files: [String: Data]

    init() {
        var files: [String: Data] = [
            "tokenizer.json": Data("{\"fixture\":1}".utf8),
            "tokenizer_config.json": Data("{\"fixture_config\":1}".utf8),
            Self.tablePath: Data((0..<(2 * 1024 * 1024)).map { UInt8($0 % 251) }),
        ]
        for chunk in 0..<6 {
            let prefix = "chunks/chunk\(chunk).mlmodelc/"
            let members = ["model.bin": Data("model-\(chunk)".utf8),
                           "weights/weight.bin": Data("weights-\(chunk)".utf8)]
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

    func spec(endpoint: String, revision: String = "v1", auth: String? = nil,
              extra: [String] = []) throws -> Data {
        var source = ["repo": "owner/model", "revision": revision, "endpoint": endpoint]
        if let auth { source["auth"] = auth }
        let entries: [[String: Any]] = try (outerPaths + extra).map { path in
            guard let data = files[path] else { throw SocketFixtureError.fixture }
            return ["path": path, "size": data.count, "sha256": FileDigest.sha256(data)]
        }
        // JSON is a YAML 1.2 subset; it exercises the production strict YAML
        // decoder without adding a second YAML implementation to the fixtures.
        return try JSONSerialization.data(withJSONObject: [
            "spec_version": 1,
            "model": ["id": modelID, "dim": 2048, "max_seq": 512, "normalize": "l2"] as [String: Any],
            "source": source, "files": entries,
            "runtime": ["compute_units": "cpu_and_ne", "pad_side": "right", "mask_dtype": "fp32"],
        ], options: [.sortedKeys])
    }
}

final class DownloadTestDirectory {
    let home: URL
    let root: URL
    let spec: URL
    init() throws {
        home = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-s2-" + UUID().uuidString)
        root = home.appendingPathComponent("models")
        spec = home.appendingPathComponent("spec.yaml")
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
    }
    deinit { try? FileManager.default.removeItem(at: home) }
    func write(_ data: Data) throws { try data.write(to: spec, options: .atomic) }
    func stage(_ path: String = "") -> URL { root.appendingPathComponent(".staging/fixture-model/" + path) }
    var install: URL { root.appendingPathComponent("fixture-model") }
    var partialTable: URL {
        stage(".embed-ane-download/" + FileDigest.sha256(Data(DownloadBundleFixture.tablePath.utf8)) + ".part")
    }
}

/// Behavior changes affect only a real listener, never URLSession internals.
final class DownloadServerScript: @unchecked Sendable {
    enum Mode: Sendable {
        case normal, interruptFirst, ignoreRange, range416, changedETag, malformedRange, corrupt
        case slowFirst, pauseLast
    }
    private let lock = NSLock()
    private var mode: Mode = .normal
    private var sha = DownloadBundleFixture.commitA
    private var refsSHA: String?
    private var calls: [String: Int] = [:]
    private var bundle: DownloadBundleFixture
    let gate = SocketGate()
    init(bundle: DownloadBundleFixture) { self.bundle = bundle }
    func setMode(_ mode: Mode) { lock.withLock { self.mode = mode } }
    func setCommit(_ commit: String) { lock.withLock { sha = commit } }
    func setRefsCommit(_ commit: String) { lock.withLock { refsSHA = commit } }
    func setBundle(_ bundle: DownloadBundleFixture) { lock.withLock { self.bundle = bundle } }

    func respond(_ request: SocketRequest) -> SocketReply {
        lock.withLock {
            let decoded = request.path.removingPercentEncoding ?? request.path
            if decoded.hasPrefix("/api/models/owner/model/revision/") {
                return SocketReply(headers: ["Content-Type": "application/json"], body: Data("{\"sha\":\"\(sha)\"}".utf8))
            }
            if decoded == "/api/models/owner/model/refs" {
                let commit = refsSHA ?? sha
                let tags = ["v1", "v2", "release/one"].map {
                    "{\"name\":\"\($0)\",\"ref\":\"refs/tags/\($0)\",\"targetCommit\":\"\(commit)\"}"
                }.joined(separator: ",")
                return SocketReply(body: Data("{\"branches\":[{\"name\":\"main\",\"ref\":\"refs/heads/main\",\"targetCommit\":\"\(sha)\"}],\"tags\":[\(tags)]}".utf8))
            }
            let prefix = "/owner/model/resolve/\(sha)/"
            guard decoded.hasPrefix(prefix) else { return SocketReply(status: 404) }
            let path = String(decoded.dropFirst(prefix.count))
            guard let original = bundle.files[path] else { return SocketReply(status: 404) }
            calls[path, default: 0] += 1
            let first = calls[path] == 1
            var data = original
            if mode == .corrupt, path == DownloadBundleFixture.tablePath, !data.isEmpty { data[0] ^= 0xff }
            let etag = mode == .changedETag && path == DownloadBundleFixture.tablePath ? "\"new\"" : "\"stable\""
            var reply = SocketReply(headers: ["ETag": etag], body: data)
            if let range = request.headers["range"], range.hasPrefix("bytes="),
               let offset = Int(range.dropFirst(6).dropLast()), offset > 0 {
                if mode == .range416, path == DownloadBundleFixture.tablePath {
                    return SocketReply(status: 416, headers: ["Content-Range": "bytes */\(data.count)"])
                }
                if mode != .ignoreRange || path != DownloadBundleFixture.tablePath {
                    guard offset < data.count else { return SocketReply(status: 416) }
                    reply.status = 206
                    let start = mode == .malformedRange ? offset + 1 : offset
                    reply.headers["Content-Range"] = "bytes \(start)-\(data.count - 1)/\(data.count)"
                    reply.body = data.subdata(in: offset..<data.count)
                }
            }
            if path == DownloadBundleFixture.tablePath, first {
                if mode == .interruptFirst { reply.stallAfter = 65_536 }
                if mode == .slowFirst { reply.pauseAfter = 65_536; reply.gate = gate }
            }
            if mode == .pauseLast, path == "chunks/chunk5.mlmodelc/weights/weight.bin" { reply.beforeHeaders = gate }
            return reply
        }
    }
}

func testDownloader(environment: [String: String] = [:]) -> ModelDownloader {
    ModelDownloader(transport: URLSessionDownloadTransport(configuration: {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 5
        configuration.timeoutIntervalForResource = 20
        return configuration
    }), policy: .loopbackHTTPForTests, environment: { environment })
}
