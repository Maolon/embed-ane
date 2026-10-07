import EmbedANECore
import Foundation

private final class NoStatusRedirect: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}

enum StatusClient {
    static func read(port: Int) async throws -> Data {
        guard (1...65535).contains(port), let url = URL(string: "http://127.0.0.1:\(port)/health") else {
            throw CLIUsageError("Invalid status port.")
        }
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 5; config.timeoutIntervalForResource = 10
        config.waitsForConnectivity = false; config.connectionProxyDictionary = [:]
        let session = URLSession(configuration: config, delegate: NoStatusRedirect(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"; request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (stream, response) = try await session.bytes(for: request)
        guard let response = response as? HTTPURLResponse, response.statusCode == 200 else {
            throw EmbedANEError.transport(path: url.absoluteString, reason: "Status did not return HTTP 200 (redirects are refused).")
        }
        guard response.expectedContentLength <= 1_048_576 else { throw EmbedANEError.bodyTooLarge }
        var body = Data()
        for try await byte in stream {
            guard body.count < 1_048_576 else { throw EmbedANEError.bodyTooLarge }
            body.append(byte)
        }
        let health: StatusHealth
        do { health = try JSONDecoder().decode(StatusHealth.self, from: body) }
        catch { throw EmbedANEError.transport(path: url.absoluteString, reason: "Status response is not the health JSON contract.") }
        guard health.status == "ok", health.uptimeS.isFinite, health.uptimeS >= 0,
              health.model.queueDepth >= 0 else {
            throw EmbedANEError.transport(path: url.absoluteString, reason: "Invalid health values.")
        }
        if body.last != 10 { body.append(10) }
        return body
    }
}
private struct StatusHealth: Decodable {
    struct Model: Decodable {
        let id: String
        let state: LifecycleState
        let queueDepth: Int
        let residentBytes: UInt64
        enum CodingKeys: String, CodingKey { case id, state, queueDepth = "queue_depth", residentBytes = "resident_bytes" }
    }
    let status: String
    let model: Model
    let uptimeS: Double
    enum CodingKeys: String, CodingKey { case status, model, uptimeS = "uptime_s" }
}
