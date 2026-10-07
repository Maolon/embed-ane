import EmbedANECore
import Foundation

public actor MockMultimodalPredictor: MultimodalEmbeddingPredictor {
    public nonisolated let text = MockPredictor()
    public let imageGate = AsyncGate()
    public private(set) var imageRequests: [ImageEmbeddingRequest] = []
    private var failure: EmbedANEError?
    private var invalidOutput = false
    public init() {}
    public func setFailure(_ error: EmbedANEError?) { failure = error }
    public func setInvalidOutput(_ value: Bool) { invalidOutput = value }
    public func load() async throws -> LoadReport { try await text.load() }
    public func predict(_ request: PredictRequest) async throws -> PredictResult { try await text.predict(request) }
    public func predictImage(_ request: ImageEmbeddingRequest) async throws -> ImageEmbeddingResult {
        imageRequests.append(request)
        await imageGate.wait()
        if let failure { throw failure }
        return .init(embedding: invalidOutput ? [1] : [3, 4] + Array(repeating: 0, count: 2046),
                     promptTokens: request.text.utf8.count + 64 + 1, tokenizeNS: 7, tableLookupNS: 5,
                     perChunkNS: [1, 2, 3, 4, 5, 6])
    }
    public func unload() async throws -> UnloadReport { try await text.unload() }
}
