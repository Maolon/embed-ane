import Foundation
import EmbedANECore

/// Production permits HTTPS only. HTTP is an explicit test capability, limited
/// to numeric IPv4 loopback; it is not inferred from a model specification.
public enum DownloadTransportPolicy: Sendable {
    case httpsOnly
    case loopbackHTTPForTests

    public func validate(_ url: URL) throws {
        guard let c = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let host = c.host, !host.isEmpty, c.user == nil, c.password == nil,
              c.fragment == nil else {
            throw EmbedANEError.invalidSpec("Invalid download URL")
        }
        if c.scheme == "https" { return }
        if case .loopbackHTTPForTests = self, c.scheme == "http", host == "127.0.0.1",
           let port = c.port, (1...65535).contains(port) { return }
        throw EmbedANEError.invalidSpec("Downloads require HTTPS; loopback HTTP is test-only")
    }
}

public struct DownloadResponse: Sendable {
    public let status: Int
    public let headers: [String: String]
    public init(status: Int, headers: [String: String]) {
        self.status = status
        self.headers = headers.reduce(into: [:]) { $0[$1.key.lowercased()] = $1.value }
    }
    public subscript(_ name: String) -> String? { headers[name.lowercased()] }
}

/// Header validation precedes delivery of any bytes. Implementations must call
/// callbacks serially, await their completion, and stop after a callback throws.
/// A completed execute has no remaining callbacks. Bodies are streamed, not
/// accumulated. Cancellation must terminate the transfer and preserve sink data.
public protocol DownloadTransport: Sendable {
    func execute(
        _ request: URLRequest,
        policy: DownloadTransportPolicy,
        receiveResponse: @escaping @Sendable (DownloadResponse) throws -> Void,
        receiveData: @escaping @Sendable (Data) throws -> Void
    ) async throws
}

/// A new ephemeral session per transfer avoids shared cookie/credential/cache
/// state. The configuration factory is injectable; no session is constructed
/// until execute(), so offline verification cannot accidentally open a socket.
public struct URLSessionDownloadTransport: DownloadTransport {
    private let configuration: @Sendable () -> URLSessionConfiguration

    public init(configuration: @escaping @Sendable () -> URLSessionConfiguration = { .ephemeral }) {
        self.configuration = configuration
    }

    public func execute(
        _ request: URLRequest,
        policy: DownloadTransportPolicy,
        receiveResponse: @escaping @Sendable (DownloadResponse) throws -> Void,
        receiveData: @escaping @Sendable (Data) throws -> Void
    ) async throws {
        guard let url = request.url else { throw EmbedANEError.invalidSpec("Missing download URL") }
        try policy.validate(url)
        let delegate = StreamingDelegate(
            originalURL: url, policy: policy,
            receiveResponse: receiveResponse, receiveData: receiveData
        )
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let config = configuration()
                config.urlCache = nil
                config.requestCachePolicy = .reloadIgnoringLocalCacheData
                config.httpCookieStorage = nil
                config.httpShouldSetCookies = false
                config.urlCredentialStorage = nil
                // Configuration injection must not smuggle default credentials.
                config.httpAdditionalHeaders = nil
                config.waitsForConnectivity = false
                let queue = OperationQueue()
                queue.maxConcurrentOperationCount = 1
                queue.name = "embed-ane.download.transport"
                delegate.start(request, configuration: config, queue: queue, continuation: continuation)
            }
        } onCancel: {
            delegate.cancel()
        }
    }
}

/// URLSession callbacks use one serial queue. The lock additionally protects the
/// cancellation/start race and continuation ownership across the caller's task.
private final class StreamingDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let lock = NSLock()
    private let originalURL: URL
    private let policy: DownloadTransportPolicy
    private let receiveResponse: @Sendable (DownloadResponse) throws -> Void
    private let receiveData: @Sendable (Data) throws -> Void
    private var continuation: CheckedContinuation<Void, any Error>?
    private var session: URLSession?
    private var task: URLSessionDataTask?
    private var failure: (any Error)?
    private var cancelled = false
    private var redirects = 0

    init(originalURL: URL, policy: DownloadTransportPolicy,
         receiveResponse: @escaping @Sendable (DownloadResponse) throws -> Void,
         receiveData: @escaping @Sendable (Data) throws -> Void) {
        self.originalURL = originalURL
        self.policy = policy
        self.receiveResponse = receiveResponse
        self.receiveData = receiveData
    }

    func start(_ request: URLRequest, configuration: URLSessionConfiguration,
               queue: OperationQueue, continuation: CheckedContinuation<Void, any Error>) {
        lock.withLock {
            guard !cancelled else { continuation.resume(throwing: CancellationError()); return }
            self.continuation = continuation
            let session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
            self.session = session
            let task = session.dataTask(with: request)
            self.task = task
            task.resume()
        }
    }

    func cancel() {
        lock.withLock { cancelled = true; task?.cancel() }
    }

    private func fail(_ error: any Error) {
        lock.withLock { if failure == nil { failure = error }; task?.cancel() }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask,
                    didReceive response: URLResponse,
                    completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void) {
        do {
            guard let response = response as? HTTPURLResponse else {
                throw EmbedANEError.transport(path: "response", reason: "Non-HTTP response")
            }
            var headers: [String: String] = [:]
            for (key, value) in response.allHeaderFields {
                headers[String(describing: key)] = String(describing: value)
            }
            try receiveResponse(DownloadResponse(status: response.statusCode, headers: headers))
            completionHandler(.allow)
        } catch {
            fail(error)
            completionHandler(.cancel)
        }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard lock.withLock({ failure == nil && !cancelled }) else { return }
        do { try receiveData(data) } catch { fail(error) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask,
                    willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
                    completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        do {
            redirects += 1
            guard redirects <= 10, let url = request.url else {
                throw EmbedANEError.transport(path: "redirect", reason: "Redirect limit or invalid destination")
            }
            try policy.validate(url)
            var safe = request
            // Never forward a Hub bearer token to an LFS/CDN origin. Also avoid
            // reattaching it after any hop which has already removed it.
            if !Self.sameOrigin(originalURL, url) {
                safe.setValue(nil, forHTTPHeaderField: "Authorization")
            }
            safe.setValue(nil, forHTTPHeaderField: "Cookie")
            safe.setValue(nil, forHTTPHeaderField: "Proxy-Authorization")
            completionHandler(safe)
        } catch {
            fail(error)
            completionHandler(nil)
        }
    }

    private static func sameOrigin(_ lhs: URL, _ rhs: URL) -> Bool {
        lhs.scheme?.lowercased() == rhs.scheme?.lowercased()
            && lhs.host?.lowercased() == rhs.host?.lowercased()
            && (lhs.port ?? (lhs.scheme == "https" ? 443 : 80))
                == (rhs.port ?? (rhs.scheme == "https" ? 443 : 80))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        let completion: (CheckedContinuation<Void, any Error>?, (any Error)?) = lock.withLock {
            let result: (any Error)? = failure ?? (cancelled ? CancellationError() : error)
            let continuation = self.continuation
            self.continuation = nil
            self.task = nil
            self.session = nil
            return (continuation, result)
        }
        session.finishTasksAndInvalidate()
        if let error = completion.1 { completion.0?.resume(throwing: error) }
        else { completion.0?.resume() }
    }
}
