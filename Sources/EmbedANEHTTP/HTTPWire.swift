import EmbedANECore
import Foundation
import Hummingbird

struct EmbeddingsRequest: Decodable {
    let model: String
    let input: EmbeddingInput
    let encodingFormat: String?
    let dimensions: Int?
    let user: String? // Accepted for SDK compatibility; never logged or used for inference.
    enum CodingKeys: String, CodingKey, CaseIterable {
        case model, input, encodingFormat = "encoding_format", dimensions, user
    }
    init(from decoder: any Swift.Decoder) throws {
        do { try StrictCoding.rejectUnknown(decoder, allowed: CodingKeys.allCases.map(\.rawValue)) }
        catch { throw EmbedANEError.invalidRequest("Unknown embeddings request field.", param: nil) }
        let c = try decoder.container(keyedBy: CodingKeys.self)
        model = try c.decode(String.self, forKey: .model)
        // All schema checks precede every semantic validation, regardless of
        // the order keys occur in the incoming JSON object.
        input = try c.decode(EmbeddingInput.self, forKey: .input)
        encodingFormat = try c.contains(.encodingFormat) ? c.decode(String.self, forKey: .encodingFormat) : nil
        dimensions = try c.contains(.dimensions) ? c.decode(Int.self, forKey: .dimensions) : nil
        user = try c.contains(.user) ? c.decode(String.self, forKey: .user) : nil
    }
    func validate(maxTexts: Int) throws {
        if let encodingFormat, encodingFormat != "float" && encodingFormat != "base64" { throw EmbedANEError.unsupportedEncodingFormat }
        if let dimensions, !(1...ModelABI.dimension).contains(dimensions) { throw EmbedANEError.unsupportedDimension }
        try input.validate(maxTexts: maxTexts)
    }
}

struct EmbeddingsResponse: Encodable {
    struct Item: Encodable {
        let object = "embedding"
        let index: Int
        let embedding: WireEmbedding
    }
    struct Usage: Encodable {
        let promptTokens: Int
        let totalTokens: Int
        enum CodingKeys: String, CodingKey { case promptTokens = "prompt_tokens", totalTokens = "total_tokens" }
    }
    let object = "list"
    let data: [Item]
    let model: String
    let usage: Usage
    init(model: String, execution: EmbeddingExecution, dimensions: Int?, encodingFormat: String?) throws {
        self.model = model
        data = try execution.embeddings.enumerated().map {
            Item(index: $0.offset, embedding: try WireEmbedding(vector: $0.element, dimensions: dimensions,
                                                               base64: encodingFormat == "base64"))
        }
        usage = Usage(promptTokens: execution.promptTokens, totalTokens: execution.promptTokens)
    }
}

/// Shortening is an explicitly requested output transform, not normalization of
/// the default graph output or a claim of trained Matryoshka quality.
struct WireEmbedding: Encodable {
    let vector: [Float]
    let base64: Bool
    init(vector: [Float], dimensions: Int?, base64: Bool) throws {
        if let dimensions, dimensions < vector.count {
            let prefix = Array(vector.prefix(dimensions))
            let norm = sqrt(prefix.reduce(0.0) { $0 + Double($1) * Double($1) })
            guard norm.isFinite, norm > 0 else {
                throw EmbedANEError.invalidRequest("The requested embedding prefix has zero norm; choose more dimensions.", param: "dimensions")
            }
            self.vector = prefix.map { Float(Double($0) / norm) }
        } else { self.vector = vector } // Omitted and explicit 2048 preserve every bit.
        self.base64 = base64
    }
    func encode(to encoder: any Swift.Encoder) throws {
        var c = encoder.singleValueContainer()
        if base64 {
            var bytes = Data(capacity: vector.count * 4)
            for value in vector {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) }
            }
            try c.encode(bytes.base64EncodedString())
        } else { try c.encode(vector) }
    }
}

public struct HealthResponse: Encodable, Sendable {
    public struct Model: Encodable, Sendable {
        public let id: String
        public let state: LifecycleState
        public let queueDepth: Int
        public let residentBytes: UInt64
        enum CodingKeys: String, CodingKey { case id, state, queueDepth = "queue_depth", residentBytes = "resident_bytes" }
        public init(id: String, state: LifecycleState, queueDepth: Int = 0, residentBytes: UInt64 = 0) {
            self.id = id; self.state = state; self.queueDepth = queueDepth; self.residentBytes = residentBytes
        }
    }
    public let status = "ok"
    public let model: Model
    public let uptimeS: Double
    enum CodingKeys: String, CodingKey { case status, model, uptimeS = "uptime_s" }
    public init(model: Model, uptimeS: Double) {
        self.model = model; self.uptimeS = uptimeS
    }
}

public struct ModelsResponse: Encodable, Sendable {
    public struct Model: Encodable, Sendable {
        public let id: String
        public let object = "model"
        public let ownedBy = "embed-ane"
        enum CodingKeys: String, CodingKey { case id, object, ownedBy = "owned_by" }
        public init(id: String) { self.id = id }
    }
    public let object = "list"
    public let data: [Model]
    public init(data: [Model]) { self.data = data }
}

public struct LoadResponse: Encodable, Sendable {
    public let state: LifecycleState
    public let report: LoadReport
    public let residentBytesScope = "process"
    enum CodingKeys: String, CodingKey { case state, report, residentBytesScope = "resident_bytes_scope" }
    public init(state: LifecycleState, report: LoadReport) {
        self.state = state; self.report = report
    }
}

public struct UnloadResponse: Encodable, Sendable {
    public let state: LifecycleState
    public let report: UnloadReport
    public let residentBytesScope = "process"
    enum CodingKeys: String, CodingKey { case state, report, residentBytesScope = "resident_bytes_scope" }
    public init(state: LifecycleState, report: UnloadReport) {
        self.state = state; self.report = report
    }
}

public struct SupervisorHealthResponse: Encodable, Sendable {
    public struct Model: Encodable, Sendable {
        public let id: String
        public let state: LifecycleState
        public let queueDepth: Int
        public let residentBytes: UInt64
        enum CodingKeys: String, CodingKey { case id, state, queueDepth = "queue_depth", residentBytes = "resident_bytes" }
        public init(id: String, state: LifecycleState, queueDepth: Int = 0, residentBytes: UInt64 = 0) {
            self.id = id; self.state = state; self.queueDepth = queueDepth; self.residentBytes = residentBytes
        }
    }
    public let status = "ok"
    public let worker: String
    public let model: Model
    public let uptimeS: Double
    enum CodingKeys: String, CodingKey { case status, worker, model, uptimeS = "uptime_s" }
    public init(worker: String, model: Model, uptimeS: Double) {
        self.worker = worker; self.model = model; self.uptimeS = uptimeS
    }
}

public struct HTTPErrorEnvelope: Encodable, Sendable {
    public struct Body: Encodable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String
        enum CodingKeys: String, CodingKey { case message, type, param, code }
        public init(message: String, type: String, param: String?, code: String) {
            self.message = message; self.type = type; self.param = param; self.code = code
        }
        public func encode(to encoder: any Swift.Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(message, forKey: .message); try c.encode(type, forKey: .type)
            try c.encode(code, forKey: .code)
            if let param { try c.encode(param, forKey: .param) }
            else { try c.encodeNil(forKey: .param) }
        }
    }
    public let error: Body
    public init(error: Body) { self.error = error }
}

public enum HTTPWire {
    public static func json<T: Encodable>(_ value: T, status: HTTPResponse.Status = .ok) throws -> Response {
        let data = try JSONEncoder().encode(value)
        return Response(status: status, headers: [.contentType: "application/json", .cacheControl: "no-store"],
                        body: .init(byteBuffer: ByteBuffer(bytes: data)))
    }
    public static func decode<T: Decodable>(_ type: T.Type, data: Data) throws -> T {
        do { return try JSONDecoder().decode(type, from: data) }
        catch let error as EmbedANEError {
            if case .invalidSpec = error { throw EmbedANEError.invalidRequest("Unknown or invalid request field.", param: nil) }
            throw error
        } catch let error as DecodingError {
            let path: [any CodingKey]
            switch error {
            case let .keyNotFound(key, context): path = context.codingPath + [key]
            case let .typeMismatch(_, context), let .valueNotFound(_, context), let .dataCorrupted(context): path = context.codingPath
            @unknown default: path = []
            }
            let param = path.isEmpty ? nil : path.map(\.stringValue).joined(separator: ".")
            throw EmbedANEError.invalidRequest("Request JSON does not match the route schema.", param: param)
        } catch { throw EmbedANEError.invalidRequest("Malformed JSON request.", param: nil) }
    }
    public static func error(
        message: String,
        type: String,
        code: String,
        param: String? = nil,
        status: Int = 503,
        retryAfter: String? = "20"
    ) -> Response {
        let body = HTTPErrorEnvelope.Body(message: message, type: type, param: param, code: code)
        let fallback = "{\"error\":{\"message\":\"\(message)\",\"type\":\"\(type)\",\"param\":null,\"code\":\"\(code)\"}}"
        var response = (try? json(HTTPErrorEnvelope(error: body), status: .init(code: status)))
            ?? Response(status: .init(code: status), headers: [.contentType: "application/json"],
                        body: .init(byteBuffer: ByteBuffer(string: fallback)))
        if let retryAfter { response.headers[.retryAfter] = retryAfter }
        return response
    }
    public static func error(_ error: any Error) -> Response {
        let body: HTTPErrorEnvelope.Body
        let status: Int
        if let typed = error as? EmbedANEError {
            status = typed.httpStatus
            // Do not expose arbitrary adapter error descriptions, paths, or
            // secrets as public 5xx messages. Stable error codes are retained.
            let safeMessage: String
            switch typed {
            case .overloaded, .loading, .modelNotLoaded, .visionNotConfigured: safeMessage = typed.message
            case .failed: safeMessage = "The model is in failed/backoff state."
            default: safeMessage = status >= 500 ? "Internal server error." : typed.message
            }
            body = .init(message: safeMessage, type: typed.envelope.error.type, param: typed.param, code: typed.code)
        } else {
            status = 500
            body = .init(message: "Internal server error.", type: "internal_error", param: nil, code: "internal_error")
        }
        let fallback = "{\"error\":{\"message\":\"Internal server error.\",\"type\":\"internal_error\",\"param\":null,\"code\":\"internal_error\"}}"
        var response = (try? json(HTTPErrorEnvelope(error: body), status: .init(code: status)))
            ?? Response(status: .internalServerError, headers: [.contentType: "application/json"],
                        body: .init(byteBuffer: ByteBuffer(string: fallback)))
        if body.code == "overloaded" || body.code == "loading" { response.headers[.retryAfter] = "5" }
        if status == 401 { response.headers[.wwwAuthenticate] = "Bearer" }
        if status == 413 || status == 401 { response.headers[.connection] = "close" }
        return response
    }
}
