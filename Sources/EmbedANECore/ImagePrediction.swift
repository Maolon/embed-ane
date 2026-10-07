import Foundation

/// The visual part of a joint request: encoded image bytes, or a local video file.
public enum VisualMedia: Sendable, Equatable {
    case image(Data)
    case video(URL)
}

/// A single joint image/text or video/text item, admitted on the same FIFO as
/// text batches.
public struct ImageEmbeddingRequest: Sendable {
    public let media: VisualMedia
    public let text: String
    public var imageData: Data? { if case let .image(data) = media { data } else { nil } }
    public var videoURL: URL? { if case let .video(url) = media { url } else { nil } }

    public init(imageData: Data, text: String) throws {
        guard !imageData.isEmpty, imageData.count <= 64 * 1_024 * 1_024 else {
            throw EmbedANEError.invalidRequest("Image must contain between 1 byte and 64 MiB of encoded data.", param: "input")
        }
        media = .image(imageData); self.text = text
    }
    public init(videoURL: URL, text: String) throws {
        guard videoURL.isFileURL else {
            throw EmbedANEError.invalidRequest("Videos must be local file URLs.", param: "input")
        }
        media = .video(videoURL); self.text = text
    }
}

public protocol MultimodalEmbeddingPredictor: EmbeddingPredictor {
    func predictImage(_ request: ImageEmbeddingRequest) async throws -> ImageEmbeddingResult
}

public struct ImageEmbeddingResult: Sendable {
    public let prediction: PredictResult
    public let promptTokens: Int
    public let tokenizeNS: UInt64
    public init(embedding: [Float], promptTokens: Int, tokenizeNS: UInt64 = 0,
                tableLookupNS: UInt64 = 0, perChunkNS: [UInt64] = Array(repeating: 0, count: 6)) {
        prediction = .init(embeddings: [embedding], tableLookupNS: tableLookupNS, perChunkNS: perChunkNS)
        self.promptTokens = promptTokens; self.tokenizeNS = tokenizeNS
    }
    func validate() throws {
        guard (1...ModelABI.sequenceLength).contains(promptTokens), prediction.embeddings.count == 1,
              let vector = prediction.embeddings.first, vector.count == ModelABI.dimension,
              vector.allSatisfy(\.isFinite), vector.contains(where: { $0 != 0 }), prediction.perChunkNS.count == 6 else {
            throw EmbedANEError.abiMismatch(chunk: 5, reason: "Invalid image embedding, token usage, or timings.")
        }
    }
}

extension MultimodalPredictor: MultimodalEmbeddingPredictor {
    public func predictImage(_ request: ImageEmbeddingRequest) async throws -> ImageEmbeddingResult {
        let result: MultimodalPrediction
        switch request.media {
        case let .image(data): result = try await embed(imageData: data, text: request.text)
        case let .video(url): result = try await embed(videoURL: url, text: request.text)
        }
        return .init(embedding: result.embedding, promptTokens: result.promptTokens, tokenizeNS: result.tokenizeNS,
                     tableLookupNS: result.tableLookupNS, perChunkNS: result.perChunkNS)
    }
}
