import Darwin
import Dispatch
import EmbedANECore
import EmbedANEHTTP
import EmbedANETestSupport
import Foundation
import Testing

// These tests do not use HummingbirdTesting's in-memory router, URLProtocol,
// or a mock URLSession. Bytes cross a Darwin TCP socket into the real server.
indirect enum WireJSON: Decodable, Sendable, Equatable {
    case object([String: WireJSON]), array([WireJSON]), string(String), number(Double), bool(Bool), null
    init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let value = try? c.decode(Bool.self) { self = .bool(value) }
        else if let value = try? c.decode(Double.self) { self = .number(value) }
        else if let value = try? c.decode(String.self) { self = .string(value) }
        else if let value = try? c.decode([WireJSON].self) { self = .array(value) }
        else { self = .object(try c.decode([String: WireJSON].self)) }
    }
    subscript(_ key: String) -> WireJSON? { if case let .object(value) = self { return value[key] }; return nil }
    var string: String? { if case let .string(value) = self { return value }; return nil }
    var number: Double? { if case let .number(value) = self { return value }; return nil }
    var array: [WireJSON]? { if case let .array(value) = self { return value }; return nil }
    var keys: Set<String> { if case let .object(value) = self { return Set(value.keys) }; return [] }
}

struct SocketReply: Sendable {
    let status: Int
    let headers: [String: String]
    let body: Data
    func json() throws -> WireJSON { try JSONDecoder().decode(WireJSON.self, from: body) }
}

enum SocketFailure: Error { case syscall(String, Int32), malformedHTTP }
enum RequestFraming: Sendable { case length, chunked, declaredOnly(Int) }

enum RealSocket {
    static func request(port: Int, method: String, path: String, body: Data = Data(),
                        headers: [(String, String)] = [], framing: RequestFraming = .length,
                        httpVersion: String = "1.1") async throws -> SocketReply {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    continuation.resume(returning: try blockingRequest(port: port, method: method, path: path,
                        body: body, headers: headers, framing: framing, httpVersion: httpVersion))
                } catch { continuation.resume(throwing: error) }
            }
        }
    }
    private static func blockingRequest(port: Int, method: String, path: String, body: Data,
                                        headers: [(String, String)], framing: RequestFraming,
                                        httpVersion: String) throws -> SocketReply {
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw SocketFailure.syscall("socket", errno) }
        defer { Darwin.close(fd) }
        var noSignal: Int32 = 1
        guard setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &noSignal, socklen_t(MemoryLayout.size(ofValue: noSignal))) == 0 else {
            throw SocketFailure.syscall("SO_NOSIGPIPE", errno)
        }
        var timeout = timeval(tv_sec: 5, tv_usec: 0)
        for option in [SO_RCVTIMEO, SO_SNDTIMEO] {
            guard setsockopt(fd, SOL_SOCKET, option, &timeout, socklen_t(MemoryLayout.size(ofValue: timeout))) == 0 else {
                throw SocketFailure.syscall("timeout", errno)
            }
        }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = UInt16(port).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        let connected = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard connected == 0 else { throw SocketFailure.syscall("connect", errno) }
        var head = "\(method) \(path) HTTP/\(httpVersion)\r\nConnection: close\r\n"
        for (name, value) in headers { head += "\(name): \(value)\r\n" }
        var payload = Data()
        switch framing {
        case .length:
            head += "Content-Length: \(body.count)\r\n\r\n"
            payload = body
        case .declaredOnly(let size):
            head += "Content-Length: \(size)\r\n\r\n"
        case .chunked:
            head += "Transfer-Encoding: chunked\r\n\r\n"
            var offset = 0
            while offset < body.count {
                let end = min(body.count, offset + 16_384)
                payload.append(Data("\(String(end - offset, radix: 16))\r\n".utf8))
                payload.append(body.subdata(in: offset..<end)); payload.append(Data("\r\n".utf8))
                offset = end
            }
            payload.append(Data("0\r\n\r\n".utf8))
        }
        var bytes = Data(head.utf8); bytes.append(payload)
        try bytes.withUnsafeBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                guard let base = buffer.baseAddress else { throw SocketFailure.malformedHTTP }
                let count = Darwin.send(fd, base.advanced(by: offset), buffer.count - offset, 0)
                if count < 0, errno == EINTR { continue }
                // An early 401/413 may close while the client is still sending.
                if count < 0, errno == EPIPE || errno == ECONNRESET { break }
                guard count > 0 else { throw SocketFailure.syscall("send", errno) }
                offset += count
            }
        }
        var received = Data(), buffer = [UInt8](repeating: 0, count: 32_768)
        while true {
            let count = buffer.withUnsafeMutableBytes { Darwin.recv(fd, $0.baseAddress, $0.count, 0) }
            if count == 0 { break }
            if count < 0, errno == EINTR { continue }
            if count < 0, errno == ECONNRESET, !received.isEmpty { break }
            guard count > 0 else { throw SocketFailure.syscall("recv", errno) }
            received.append(contentsOf: buffer.prefix(count))
            guard received.count <= 8 * 1_048_576 else { throw SocketFailure.malformedHTTP }
            if let range = received.range(of: Data("\r\n\r\n".utf8)),
               let header = String(data: received[..<range.lowerBound], encoding: .utf8),
               let length = parseHeaders(header).headers["content-length"].flatMap(Int.init),
               received.count - range.upperBound >= length { break }
        }
        guard let separator = received.range(of: Data("\r\n\r\n".utf8)),
              let headerText = String(data: received[..<separator.lowerBound], encoding: .utf8) else {
            throw SocketFailure.malformedHTTP
        }
        let parsed = parseHeaders(headerText)
        guard let status = parsed.status else { throw SocketFailure.malformedHTTP }
        let responseBody = Data(received[separator.upperBound...])
        if let length = parsed.headers["content-length"].flatMap(Int.init), length != responseBody.count {
            throw SocketFailure.malformedHTTP
        }
        return SocketReply(status: status, headers: parsed.headers, body: responseBody)
    }
    private static func parseHeaders(_ text: String) -> (status: Int?, headers: [String: String]) {
        let lines = text.components(separatedBy: "\r\n")
        let status = lines.first?.split(separator: " ").dropFirst().first.flatMap { Int($0) }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[String(line[..<colon]).lowercased()] = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        }
        return (status, headers)
    }
}

private final class StartupResult: @unchecked Sendable {
    private let lock = NSLock()
    private var port: Int?
    private var error: String?
    func started(_ value: Int) { lock.withLock { port = value } }
    func failed(_ value: String) { lock.withLock { error = value } }
    var finished: Bool { lock.withLock { port != nil || error != nil } }
    var boundPort: Int? { lock.withLock { port } }
}

struct HTTPFixture: Sendable {
    let port: Int
    let server: EmbeddingHTTPServer
    let predictor: MockPredictor
    let preparer: CountingPreparer
    let token: ControlToken
    let store: ConfigurationStore
    let clock: ManualRuntimeClock
    func request(_ method: String, _ path: String, body: Data = Data(), auth: Bool = false,
                 headers extra: [(String, String)] = [], host: String? = nil,
                 framing: RequestFraming = .length) async throws -> SocketReply {
        var headers = [("Host", host ?? "127.0.0.1:\(port)"), ("Content-Type", "application/json")]
        if auth { headers.append(("Authorization", token.authorizationHeader)) }
        headers.append(contentsOf: extra)
        return try await RealSocket.request(port: port, method: method, path: path, body: body, headers: headers, framing: framing)
    }
    func load() async throws {
        let response = try await request("POST", "/control/load", auth: true)
        #expect(response.status == 200)
    }
}

func withHTTPFixture<T: Sendable>(configuration: ServiceConfiguration = .init(),
                                  vision: MockMultimodalPredictor? = nil,
                                  _ body: @Sendable (HTTPFixture) async throws -> T) async throws -> T {
    let home = FileManager.default.temporaryDirectory.appendingPathComponent("embed-ane-http-\(UUID())")
    let store = ConfigurationStore(home: home)
    let predictor = vision?.text ?? MockPredictor(), preparer = CountingPreparer(), clock = ManualRuntimeClock()
    let adapter: any EmbeddingPredictor = vision.map { $0 as any EmbeddingPredictor } ?? predictor
    let server = try EmbeddingHTTPServer(configuration: configuration, predictor: adapter, preparer: preparer,
                                         store: store, portOverride: 0, clock: clock, residentBytes: { 654321 })
    let started = StartupResult()
    let task = Task {
        do { try await server.run { port in started.started(port) } }
        catch { started.failed(String(describing: error)); throw error }
    }
    func cleanup() async {
        await predictor.releaseAll(); await preparer.gate.open(); await vision?.imageGate.open()
        task.cancel(); _ = try? await task.value
        try? FileManager.default.removeItem(at: home)
    }
    do {
        try await eventually { started.finished }
        guard let port = started.boundPort, port > 0 else { throw FixtureError.timeout }
        let token = try ControlToken.loadOrCreate(in: home.appendingPathComponent(".embed-ane"))
        let result = try await body(HTTPFixture(port: port, server: server, predictor: predictor, preparer: preparer,
                                                token: token, store: store, clock: clock))
        await cleanup(); return result
    } catch { await cleanup(); throw error }
}

func jsonData(_ value: [String: Any]) throws -> Data { try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) }
func embeddingBody(_ input: Any = "a", model: String = "m") throws -> Data { try jsonData(["model": model, "input": input]) }

func checkError(_ response: SocketReply, status: Int, code: String, param: WireJSON = .null,
                type: String? = nil) throws {
    #expect(response.status == status)
    #expect(response.headers["content-type"] == "application/json")
    let root = try response.json()
    #expect(root.keys == ["error"])
    let error = try #require(root["error"])
    #expect(error.keys == ["message", "type", "param", "code"])
    #expect(error["code"] == .string(code))
    #expect(error["param"] == param)
    #expect(error["message"]?.string?.isEmpty == false)
    let expectedType = type ?? (status >= 500 ? "internal_error" : "invalid_request_error")
    #expect(error["type"] == .string(expectedType))
}
