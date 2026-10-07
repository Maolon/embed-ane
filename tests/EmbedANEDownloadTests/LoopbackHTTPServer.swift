import Foundation
import Darwin

struct SocketRequest: Sendable {
    let method: String
    let path: String
    let headers: [String: String]
}

struct SocketReply: Sendable {
    var status = 200
    var headers: [String: String] = [:]
    var body = Data()
    var disconnectAfter: Int?
    /// Deterministic interruption: send this prefix, then HOLD the connection
    /// open (no more bytes, no EOF) until the client times out or closes.
    /// Unlike disconnectAfter, URLSession reliably delivers the prefix first.
    var stallAfter: Int?
    var pauseAfter: Int?
    var gate: SocketGate?
    var beforeHeaders: SocketGate?
}

/// Finite waits keep a failed test from wedging the macOS runner.
final class SocketGate: @unchecked Sendable {
    private let semaphore = DispatchSemaphore(value: 0)
    private let lock = NSLock()
    private var reached = false
    var isReached: Bool { lock.withLock { reached } }
    func wait() {
        lock.withLock { reached = true }
        _ = semaphore.wait(timeout: .now() + 15)
    }
    func open() { semaphore.signal() }
}

/// A real HTTP/1.1 listener, bound only to 127.0.0.1:0. No URLProtocol mocks,
/// fixed-port races, Python dependency, or framework's in-memory test client.
final class LoopbackHTTPServer: @unchecked Sendable {
    let port: UInt16
    private let listener: Int32
    private let lock = NSLock()
    private var stopped = false
    private var clients = Set<Int32>()
    private var log: [SocketRequest] = []
    private var accepts = 0
    private let handler: @Sendable (SocketRequest) -> SocketReply
    private let workers = DispatchQueue(label: "embed-ane.tests.socket-clients", attributes: .concurrent)

    var endpoint: String { "http://127.0.0.1:\(port)" }
    var requests: [SocketRequest] { lock.withLock { log } }
    var connections: Int { lock.withLock { accepts } }

    init(handler: @escaping @Sendable (SocketRequest) -> SocketReply) throws {
        self.handler = handler
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { throw SocketFixtureError.socket }
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
        guard bound == 0, listen(fd, 32) == 0 else { close(fd); throw SocketFixtureError.socket }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let named = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(fd, $0, &length) }
        }
        guard named == 0 else { close(fd); throw SocketFixtureError.socket }
        port = UInt16(bigEndian: address.sin_port)
        listener = fd
        DispatchQueue(label: "embed-ane.tests.socket-accept").async { [weak self] in self?.acceptConnections() }
    }

    deinit { stop() }
    func stop() {
        lock.withLock {
            guard !stopped else { return }
            stopped = true
            _ = shutdown(listener, Int32(SHUT_RDWR))
            close(listener)
            // Per-client workers own close(). Shutdown wakes any blocked recv.
            for client in clients { _ = shutdown(client, Int32(SHUT_RDWR)) }
        }
    }

    private func acceptConnections() {
        while !lock.withLock({ stopped }) {
            let client = accept(listener, nil, nil)
            if client < 0 {
                if errno == EINTR { continue }
                return
            }
            var yes: Int32 = 1
            _ = setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &yes, socklen_t(MemoryLayout<Int32>.size))
            var timeout = timeval(tv_sec: 10, tv_usec: 0)
            _ = setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            _ = setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
            let accepted = lock.withLock {
                guard !stopped else { return false }
                clients.insert(client)
                accepts += 1
                return true
            }
            guard accepted else { close(client); return }
            workers.async { [self] in
                defer {
                    lock.withLock { _ = clients.remove(client) }
                    close(client)
                }
                serve(client)
            }
        }
    }

    private func serve(_ client: Int32) {
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
        guard let first = lines.first else { return }
        let fields = first.split(separator: " ")
        guard fields.count == 3 else { return }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            guard let colon = line.firstIndex(of: ":") else { continue }
            headers[String(line[..<colon]).lowercased()] = String(line[line.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
        }
        let request = SocketRequest(method: String(fields[0]), path: String(fields[1]), headers: headers)
        lock.withLock { log.append(request) }
        let reply = handler(request)
        reply.beforeHeaders?.wait()
        var outgoing = reply.headers
        if outgoing.keys.allSatisfy({ $0.lowercased() != "content-length" }) {
            outgoing["Content-Length"] = String(reply.body.count)
        }
        outgoing["Connection"] = "close"
        let header = "HTTP/1.1 \(reply.status) Fixture\r\n"
            + outgoing.sorted(by: { $0.key < $1.key }).map { "\($0.key): \($0.value)\r\n" }.joined() + "\r\n"
        guard sendAll(Data(header.utf8), to: client) else { return }
        let count = min(reply.disconnectAfter ?? reply.stallAfter ?? reply.body.count, reply.body.count)
        if let pause = reply.pauseAfter, let gate = reply.gate {
            let first = min(pause, count)
            guard sendAll(reply.body.prefix(first), to: client) else { return }
            gate.wait()
            guard sendAll(reply.body.subdata(in: first..<count), to: client) else { return }
        } else {
            guard sendAll(reply.body.prefix(count), to: client) else { return }
        }
        if let _ = reply.stallAfter {
            // Hold the connection open, sending nothing further, until the peer
            // gives up (client request timeout closes its side) or the cap hits.
            drain(client, timeoutMs: 12_000)
        } else if reply.disconnectAfter != nil {
            // FIN after the prefix, then drain so the prefix is delivered before
            // close(). (Legacy abrupt path — kept for completeness.)
            _ = shutdown(client, Int32(SHUT_WR))
            drain(client, timeoutMs: 3000)
        }
    }

    private func drain(_ client: Int32, timeoutMs: Int) {
        var buf = [UInt8](repeating: 0, count: 4096)
        let deadline = Date().addingTimeInterval(TimeInterval(timeoutMs) / 1000)
        while Date() < deadline {
            let n = recv(client, &buf, buf.count, 0)
            if n <= 0 {
                if n == 0 { return }              // client closed its side
                if errno == EINTR { continue }
                return                            // RST or error: stop draining
            }
        }
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

enum SocketFixtureError: Error { case socket, timedOut, fixture }
