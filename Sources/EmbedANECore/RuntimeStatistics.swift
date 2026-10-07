import Foundation

public enum LifecycleState: String, Codable, Sendable {
    case unloaded, loading, ready, unloading, failed
}

public struct RequestTiming: Codable, Sendable, Equatable {
    public let queueWaitNS: UInt64
    public let tokenizeNS: UInt64
    public let tableLookupNS: UInt64
    public let perChunkNS: [UInt64]
    public let totalNS: UInt64
    enum CodingKeys: String, CodingKey {
        case queueWaitNS = "queue_wait_ns", tokenizeNS = "tokenize_ns"
        case tableLookupNS = "table_lookup_ns", perChunkNS = "per_chunk_ns", totalNS = "total_ns"
    }
    public init(queueWaitNS: UInt64, tokenizeNS: UInt64, tableLookupNS: UInt64,
                perChunkNS: [UInt64], totalNS: UInt64) {
        self.queueWaitNS = queueWaitNS; self.tokenizeNS = tokenizeNS
        self.tableLookupNS = tableLookupNS; self.perChunkNS = perChunkNS; self.totalNS = totalNS
    }
}

public struct TimingPercentiles: Codable, Sendable, Equatable {
    public let p50NS: UInt64
    public let p95NS: UInt64
    enum CodingKeys: String, CodingKey { case p50NS = "p50_ns", p95NS = "p95_ns" }
    public init(samples: [UInt64]) {
        let sorted = samples.sorted()
        func rank(_ percent: Int) -> UInt64 {
            guard !sorted.isEmpty else { return 0 }
            // nearest rank: ceil(p * n), one-based. No interpolation.
            return sorted[(sorted.count * percent + 99) / 100 - 1]
        }
        p50NS = rank(50); p95NS = rank(95)
    }
}

public struct RuntimeSnapshot: Codable, Sendable {
    public let state: LifecycleState
    public let queueDepth: Int
    public let inFlight: Int
    public let preparing: Int
    public let residentBytes: UInt64
    public let retryAtNS: UInt64?
    enum CodingKeys: String, CodingKey {
        case state, queueDepth = "queue_depth", inFlight = "in_flight", preparing
        case residentBytes = "resident_bytes", retryAtNS = "retry_at_ns"
    }
    public init(
        state: LifecycleState,
        queueDepth: Int,
        inFlight: Int,
        preparing: Int,
        residentBytes: UInt64,
        retryAtNS: UInt64? = nil
    ) {
        self.state = state
        self.queueDepth = queueDepth
        self.inFlight = inFlight
        self.preparing = preparing
        self.residentBytes = residentBytes
        self.retryAtNS = retryAtNS
    }
}

public struct RuntimeStatistics: Codable, Sendable {
    public let runtime: RuntimeSnapshot
    public let admittedRequests: UInt64
    public let completedRequests: UInt64
    public let failedRequests: UInt64
    public let windowCount: Int
    public let p50NS: UInt64
    public let p95NS: UInt64
    public let queueWait: TimingPercentiles
    public let tokenization: TimingPercentiles
    public let tableLookup: TimingPercentiles
    public let perChunk: [TimingPercentiles]
    public let residentBytesScope: String
    public let lastError: String?
    enum CodingKeys: String, CodingKey {
        case runtime, admittedRequests = "admitted_requests", completedRequests = "completed_requests"
        case failedRequests = "failed_requests", windowCount = "window_count"
        case p50NS = "p50_ns", p95NS = "p95_ns", queueWait = "queue_wait"
        case tokenization, tableLookup = "table_lookup", perChunk = "per_chunk"
        case residentBytesScope = "resident_bytes_scope", lastError = "last_error"
    }
    public init(
        runtime: RuntimeSnapshot,
        admittedRequests: UInt64,
        completedRequests: UInt64,
        failedRequests: UInt64,
        windowCount: Int,
        p50NS: UInt64,
        p95NS: UInt64,
        queueWait: TimingPercentiles,
        tokenization: TimingPercentiles,
        tableLookup: TimingPercentiles,
        perChunk: [TimingPercentiles],
        residentBytesScope: String = "process",
        lastError: String? = nil
    ) {
        self.runtime = runtime
        self.admittedRequests = admittedRequests
        self.completedRequests = completedRequests
        self.failedRequests = failedRequests
        self.windowCount = windowCount
        self.p50NS = p50NS
        self.p95NS = p95NS
        self.queueWait = queueWait
        self.tokenization = tokenization
        self.tableLookup = tableLookup
        self.perChunk = perChunk
        self.residentBytesScope = residentBytesScope
        self.lastError = lastError
    }
}

public struct EmbeddingExecution: Sendable {
    public let embeddings: [[Float]]
    public let promptTokens: Int
    public let timing: RequestTiming
}
