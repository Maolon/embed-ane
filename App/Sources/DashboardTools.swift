import Foundation

/// The playground is a client of the existing listener, never another runtime.
/// Use the listening port and effective model, not values pending a restart.
struct DashboardEndpoint: Equatable, Sendable {
    let port: Int
    let modelID: String
    var address: String { "http://127.0.0.1:\(port)" }
    var embeddingsAddress: String { address + "/v1/embeddings" }

    func request(text: String) throws -> URLRequest {
        guard (1...65535).contains(port), !modelID.isEmpty,
              let url = URL(string: embeddingsAddress) else { throw PlaygroundError.invalidEndpoint }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try body(text: text)
        request.timeoutInterval = 120
        return request
    }

    func examples(text: String) throws -> (curl: String, python: String) {
        let request = try request(text: text)
        guard let data = request.httpBody, let json = String(data: data, encoding: .utf8) else {
            throw PlaygroundError.invalidResponse
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        // A JSON-quoted string is also a safe Python string literal here. Quote
        // the entire JSON document again: booleans/null in a payload would stay
        // JSON, and quotes, backslashes and control characters cannot become code.
        let pythonBody = String(decoding: try encoder.encode(json), as: UTF8.self)
        let curl = """
        curl --silent --show-error \(shellQuote(embeddingsAddress)) \\
          -H 'Content-Type: application/json' \\
          --data-raw \(shellQuote(json))
        """
        let python = """
        # Python 3 — no packages to install.
        import json
        from urllib.request import Request, urlopen

        body = \(pythonBody).encode("utf-8")
        request = Request(
            "\(embeddingsAddress)",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=120) as response:
            print(json.dumps(json.load(response), indent=2))
        """
        return (curl, python)
    }

    private func body(text: String) throws -> Data {
        struct Payload: Encodable {
            let model: String
            let input: String
            let encoding_format = "float"
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(Payload(model: modelID, input: text))
    }
    private func shellQuote(_ text: String) -> String {
        "'" + text.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }
}

struct EmbeddingPreview: Sendable {
    let dimension: Int
    let norm: Double
    let firstValues: [Double]

    static func decode(_ data: Data) throws -> EmbeddingPreview {
        struct Response: Decodable {
            struct Item: Decodable {
                let object: String
                let index: Int
                let embedding: [Double]
            }
            let object: String
            let data: [Item]
        }
        let response = try JSONDecoder().decode(Response.self, from: data)
        guard response.object == "list", response.data.count == 1,
              let item = response.data.first, item.object == "embedding", item.index == 0,
              !item.embedding.isEmpty, item.embedding.allSatisfy(\.isFinite) else {
            throw PlaygroundError.invalidResponse
        }
        let norm = sqrt(item.embedding.reduce(0) { $0 + $1 * $1 })
        guard norm.isFinite, norm > 0 else { throw PlaygroundError.invalidResponse }
        return EmbeddingPreview(dimension: item.embedding.count, norm: norm, firstValues: Array(item.embedding.prefix(4)))
    }
}

enum PlaygroundError: LocalizedError {
    case invalidEndpoint, invalidResponse, rejected(String)
    var errorDescription: String? {
        switch self {
        case .invalidEndpoint: "The local server is not available. Check its status and try again."
        case .invalidResponse: "The server returned an unexpected result. Open Logs for details."
        case .rejected(let message): message
        }
    }

    static func message(for error: any Error) -> String {
        if let known = error as? PlaygroundError { return known.errorDescription ?? "The request failed." }
        if let network = error as? URLError {
            if network.code == .timedOut {
                return "This request timed out. The model may still be working; check its status before trying again."
            }
            return "Could not reach the local server. Check its status and try again."
        }
        return "Could not read an embedding. Check the server status or open Logs."
    }

    static func rejection(data: Data, status: Int) -> PlaygroundError {
        struct Envelope: Decodable {
            struct Detail: Decodable { let code: String? }
            let error: Detail
        }
        let code = (try? JSONDecoder().decode(Envelope.self, from: data))?.error.code
        let message: String
        switch code {
        case "input_too_long": message = "That text is too long. Try a shorter sentence."
        case "empty_input": message = "Enter a sentence to embed."
        case "loading": message = "The model is still loading. Try again when it is ready."
        case "model_not_loaded": message = "Load a verified model, then try again."
        case "overloaded", "busy": message = "The server is busy. Wait a moment and try again."
        default: message = "The server could not finish this request (HTTP \(status)). Check its status or open Logs."
        }
        return .rejected(message)
    }
}

enum PlaygroundClient {
    static func embed(text: String, at endpoint: DashboardEndpoint) async throws -> EmbeddingPreview {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.httpShouldSetCookies = false
        configuration.connectionProxyDictionary = [:] // Never send local text through a configured proxy.
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.timeoutIntervalForResource = 120
        let session = URLSession(configuration: configuration)
        defer { session.invalidateAndCancel() }
        let (data, response) = try await session.data(for: endpoint.request(text: text), delegate: NoRedirects())
        guard let http = response as? HTTPURLResponse else { throw PlaygroundError.invalidResponse }
        guard http.statusCode == 200 else { throw PlaygroundError.rejection(data: data, status: http.statusCode) }
        return try EmbeddingPreview.decode(data)
    }

    /// Never forward entered text to a redirect destination, even if another
    /// process has taken the local port since the last UI snapshot.
    private final class NoRedirects: NSObject, URLSessionTaskDelegate, Sendable {
        func urlSession(_ session: URLSession, task: URLSessionTask,
                        willPerformHTTPRedirection response: HTTPURLResponse,
                        newRequest request: URLRequest) async -> URLRequest? { nil }
    }
}
