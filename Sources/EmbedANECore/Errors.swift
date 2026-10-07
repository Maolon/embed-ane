import Foundation

/// Stable errors shared by the CLI, HTTP boundary and model worker.
public enum EmbedANEError: Error, Sendable, Equatable, LocalizedError {
    case invalidRequest(String, param: String?)
    case inputTooLong(index: Int, contentTokens: Int)
    case emptyInput(index: Int?)
    case batchTooLarge(Int)
    case unsupportedDimension
    case unsupportedEncodingFormat
    case modelNotLoaded
    case visionNotConfigured
    case overloaded
    case loading
    case busy
    case failed(String)
    case conflict(String)
    case unauthorized
    case bodyTooLarge
    case invalidSpec(String)
    case unsafePath(String)
    case verification(path: String, reason: String)
    case movingBranch(String)
    case transport(path: String, reason: String)
    case invalidNPY(String)
    case abiMismatch(chunk: Int, reason: String)
    case io(path: String, reason: String)
    case cancelled

    public var code: String {
        switch self {
        case .invalidRequest: "invalid_request"
        case .inputTooLong: "input_too_long"
        case .emptyInput: "empty_input"
        case .batchTooLarge: "batch_too_large"
        case .unsupportedDimension: "unsupported_dimension"
        case .unsupportedEncodingFormat: "unsupported_encoding_format"
        case .modelNotLoaded: "model_not_loaded"
        case .visionNotConfigured: "vision_not_configured"
        case .overloaded: "overloaded"
        case .loading: "loading"
        case .busy: "busy"
        case .failed: "failed"
        case .conflict: "conflict"
        case .unauthorized: "unauthorized"
        case .bodyTooLarge: "body_too_large"
        case .invalidSpec: "invalid_spec"
        case .unsafePath: "unsafe_path"
        case .verification: "verification_failed"
        case .movingBranch: "moving_branch"
        case .transport: "transport_error"
        case .invalidNPY: "invalid_npy"
        case .abiMismatch: "abi_mismatch"
        case .io: "io_error"
        case .cancelled: "cancelled"
        }
    }

    public var httpStatus: Int {
        switch self {
        case .invalidRequest, .inputTooLong, .emptyInput, .batchTooLarge,
             .unsupportedDimension, .unsupportedEncodingFormat, .invalidSpec, .movingBranch: 400
        case .visionNotConfigured: 501
        case .unauthorized: 401
        case .busy, .conflict: 409
        case .bodyTooLarge: 413
        case .modelNotLoaded, .overloaded, .loading, .failed: 503
        default: 500
        }
    }

    public var message: String {
        switch self {
        case let .invalidRequest(message, _): message
        case let .inputTooLong(index, count): "input[\(index)] has \(count) tokens; text and token inputs may use at most \(ModelABI.textContentLimit)."
        case let .emptyInput(index): index.map { "input[\($0)] is empty." } ?? "input must not be empty."
        case let .batchTooLarge(count): "Batch contains \(count) texts; the configured batch limit was exceeded."
        case .unsupportedDimension: "dimensions must be an integer in 1...2048."
        case .unsupportedEncodingFormat: "encoding_format must be float or base64."
        case .modelNotLoaded: "The model is not loaded."
        case .visionNotConfigured: "Images and videos need a multimodal model. Install one (it includes vision_tower.mlmodelc), or set vision_tower_path, vision_extrope_chunks_dir and vision_position_table_path, then restart the server."
        case .overloaded: "The bounded admission queue is full."
        case .loading: "The shared model load is in progress."
        case .busy: "Queued, preparing, or in-flight work prevents unloading."
        case let .failed(reason): "The model is in failed/backoff state: \(reason)"
        case let .conflict(reason): reason
        case .unauthorized: "Control authorization was rejected."
        case .bodyTooLarge: "The request body exceeds 1 MiB."
        case let .invalidSpec(reason): "Invalid specification: \(reason)"
        case let .unsafePath(path): "Unsafe path: \(path)"
        case let .verification(path, reason): "Verification failed for \(path): \(reason)"
        case let .movingBranch(branch): "Moving branch '\(branch)' is not an immutable model revision."
        case let .transport(path, reason): "Transfer failed for \(path): \(reason)"
        case let .invalidNPY(reason): "Invalid NPY embedding table: \(reason)"
        case let .abiMismatch(chunk, reason): "CoreML ABI mismatch in chunk \(chunk): \(reason)"
        case let .io(path, reason): "I/O failed for \(path): \(reason)"
        case .cancelled: "Request cancelled before admission."
        }
    }

    public var param: String? {
        switch self {
        case let .invalidRequest(_, param): param
        case let .inputTooLong(index, _): "input[\(index)]"
        case let .emptyInput(index): index.map { "input[\($0)]" } ?? "input"
        case .batchTooLarge: "input"
        case .unsupportedDimension: "dimensions"
        case .unsupportedEncodingFormat: "encoding_format"
        default: nil
        }
    }

    public var errorDescription: String? { message }
    public var isVerificationFailure: Bool {
        switch self { case .verification, .unsafePath, .invalidNPY, .abiMismatch: true; default: false }
    }

    public struct Envelope: Codable, Sendable, Equatable {
        public struct Body: Codable, Sendable, Equatable {
            public let message: String
            public let type: String
            public let param: String?
            public let code: String
        }
        public let error: Body
    }

    public var envelope: Envelope {
        let type: String
        switch self {
        case .overloaded, .loading: type = "overloaded_error"
        default: type = httpStatus >= 500 ? "internal_error" : "invalid_request_error"
        }
        return Envelope(error: .init(message: message, type: type, param: param, code: code))
    }
}

public enum ModelABI {
    public static let vocabulary = 248_078
    public static let dimension = 2_048
    public static let sequenceLength = 512
    public static let embeddingToken = 248_077
    public static let imStartToken = 248_045
    public static let imEndToken = 248_046
    /// "user" then "\n", as the reference tokenizer encodes the chat-template role line.
    public static let userRoleTokens = [846]
    public static let newlineToken = 198
    /// Content tokens a text input may use: 512 minus `<embedding>` and the
    /// four chat-template tokens around it.
    public static let textContentLimit = 507
    public static let tablePayloadBytes: Int64 = 1_016_127_488
    public static let defaultModelID = "wemm-embedding-2b-ane"
}
