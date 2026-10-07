// App/Sources is an Xcode target, not an SPM test dependency. Run the focused
// client/clipboard checks with: python3 tests/AppWindowChecks/check_dashboard_tools.py
import Foundation

private struct CheckFailure: Error { let message: String }
private func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw CheckFailure(message: message) }
}

@main struct DashboardToolsChecks {
    static func main() async throws {
        guard CommandLine.arguments.count == 2, let port = Int(CommandLine.arguments[1]) else {
            throw CheckFailure(message: "Expected the loopback fixture port")
        }
        let endpoint = DashboardEndpoint(port: port, modelID: "fixture-valid")
        try require(endpoint.address == "http://127.0.0.1:\(port)", "Port must not use localized grouping")
        let samples = [
            "A small step, a new idea.",
            "It's \"quoted\" and backslashed: \\ \nnext line",
            "你好 🙂 café\u{2028}next",
            "$(printf injected) `whoami` $HOME; '\\",
            "\u{0}\u{1}\t\r\n",
            "''' \"\"\" \\u1234 \\\\",
        ]
        struct Example: Encodable { let text: String; let curl: String; let python: String }
        var examples: [Example] = []
        for text in samples {
            let request = try endpoint.request(text: text)
            let body = try JSONSerialization.jsonObject(with: request.httpBody ?? Data()) as? [String: String]
            try require(request.httpMethod == "POST", "Method")
            try require(request.url?.absoluteString == endpoint.embeddingsAddress, "Route")
            try require(request.value(forHTTPHeaderField: "Content-Type") == "application/json", "JSON header")
            try require(body == ["model": "fixture-valid", "input": text, "encoding_format": "float"], "Lossless request body")
            let snippets = try endpoint.examples(text: text)
            examples.append(Example(text: text, curl: snippets.curl, python: snippets.python))
        }
        for badPort in [0, -1, 65536] {
            do {
                _ = try DashboardEndpoint(port: badPort, modelID: "fixture-valid").request(text: "test")
                throw CheckFailure(message: "Invalid port accepted")
            } catch PlaygroundError.invalidEndpoint {}
        }
        let valid = Data(#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.6,0.8,0,0]}]}"#.utf8)
        let preview = try EmbeddingPreview.decode(valid)
        try require(preview.dimension == 4 && abs(preview.norm - 1) < 0.000001, "Dimension/norm")
        try require(preview.firstValues == [0.6, 0.8, 0, 0], "Preview values")
        let invalid = [
            #"{"object":"wrong","data":[]}"#,
            #"{"object":"list","data":[{"object":"embedding","index":1,"embedding":[1]}]}"#,
            #"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[]}]}"#,
            #"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0,0]}]}"#,
            #"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[1e308]}]}"#,
        ]
        for value in invalid {
            do {
                _ = try EmbeddingPreview.decode(Data(value.utf8))
                throw CheckFailure(message: "Malformed preview accepted")
            } catch PlaygroundError.invalidResponse {}
        }
        try require(PlaygroundError.message(for: URLError(.timedOut)).contains("still be working"), "Timeout guidance")
        let live = try await PlaygroundClient.embed(text: "swift-client", at: endpoint)
        try require(live.dimension == 4 && abs(live.norm - 1) < 0.000001, "Actual client decode")
        for (model, expected) in [("fixture-loading", "still loading"), ("fixture-redirect", "HTTP 307"), ("fixture-malformed", "unexpected result")] {
            do {
                _ = try await PlaygroundClient.embed(text: "swift-client", at: DashboardEndpoint(port: port, modelID: model))
                throw CheckFailure(message: "Expected a client error: \(model)")
            } catch let error as PlaygroundError {
                let message = PlaygroundError.message(for: error)
                try require(message.contains(expected), "Actionable error: \(message)")
                try require(!message.contains("raw-secret-error"), "Do not expose raw server errors")
            }
        }
        print(String(decoding: try JSONEncoder().encode(examples), as: UTF8.self))
    }
}
