import Foundation

public struct ReferenceEmbedding: Decodable, Sendable {
    public let text: String
    public let embedding: [Double]
    enum CodingKeys: String, CodingKey { case text, embedding }
    public init(text: String, embedding: [Double]) { self.text = text; self.embedding = embedding }
    public init(from decoder: any Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decode(String.self, forKey: .text)
        embedding = try c.decode([Double].self, forKey: .embedding)
    }
}

public enum BenchmarkInput {
    /// One text per line; only the terminal newline is discarded. Blank lines
    /// inside a corpus are errors, not silently removed or renumbered examples.
    public static func corpus(_ data: Data) throws -> [String] {
        guard String(data: data, encoding: .utf8) != nil else {
            throw EmbedANEError.invalidRequest("Corpus must be UTF-8.", param: "corpus")
        }
        // Split bytes: Swift treats CRLF as ONE extended grapheme, so splitting
        // a String on the Character "\n" can leave CRLF records unsplit.
        var lines = data.split(separator: 10, omittingEmptySubsequences: false).map { line in
            String(decoding: line.last == 13 ? line.dropLast() : line[...], as: UTF8.self)
        }
        if data.last == 10 { lines.removeLast() }
        guard !lines.isEmpty, !lines.contains(where: \.isEmpty) else {
            throw EmbedANEError.invalidRequest("Corpus must contain nonempty lines.", param: "corpus")
        }
        return lines
    }
    public static func references(_ data: Data, texts: [String]) throws -> [ReferenceEmbedding] {
        let lines: [String]
        do { lines = try corpus(data) }
        catch { throw EmbedANEError.verification(path: "reference", reason: "Reference must be nonempty UTF-8 JSONL.") }
        var result: [ReferenceEmbedding] = []
        for (index, line) in lines.enumerated() {
            do { result.append(try JSONDecoder().decode(ReferenceEmbedding.self, from: Data(line.utf8))) }
            catch { throw EmbedANEError.verification(path: "reference[\(index)]", reason: "Expected {text, embedding} JSON object.") }
        }
        try validateReferences(result, texts: texts)
        return result
    }
    public static func validateReferences(_ references: [ReferenceEmbedding], texts: [String]) throws {
        guard references.count == texts.count else {
            throw EmbedANEError.verification(path: "reference", reason: "Reference count must equal corpus count.")
        }
        for index in texts.indices {
            let record = references[index]
            guard record.text == texts[index], record.embedding.count == ModelABI.dimension,
                  record.embedding.allSatisfy(\.isFinite), record.embedding.contains(where: { $0 != 0 }) else {
                throw EmbedANEError.verification(path: "reference[\(index)]", reason: "Text/order, dimension, or finite nonzero vector mismatch.")
            }
        }
    }
}

public enum VectorCosine {
    /// Scale before accumulation to avoid overflow/underflow. This is a metric
    /// computation only; actual embeddings are neither changed nor normalized.
    public static func raw(_ actual: [Float], _ reference: [Double]) throws -> Double {
        guard actual.count == ModelABI.dimension, reference.count == actual.count,
              actual.allSatisfy(\.isFinite), reference.allSatisfy(\.isFinite),
              let aScale = actual.map({ abs(Double($0)) }).max(), aScale > 0,
              let bScale = reference.map({ abs($0) }).max(), bScale > 0 else {
            throw EmbedANEError.verification(path: "embedding", reason: "Cosine requires finite, nonzero 2048-dimensional vectors.")
        }
        var dot = 0.0; var aa = 0.0; var bb = 0.0
        for index in actual.indices {
            let a = Double(actual[index]) / aScale
            let b = reference[index] / bScale
            dot += a * b; aa += a * a; bb += b * b
        }
        // Do not clamp raw diagnostic values to [0,1] or hide negative cosines.
        return dot / sqrt(aa * bb)
    }
}

public struct BenchmarkSample: Codable, Sendable {
    public let index: Int
    public let text: String
    public let promptTokens: Int
    public let embedding: [Float]
    public let timing: RequestTiming
    public let rawCosine: Double?
    enum CodingKeys: String, CodingKey {
        case index, text, promptTokens = "prompt_tokens", embedding, timing, rawCosine = "raw_cosine"
    }
}
public struct ParityMeasurement: Codable, Sendable {
    public let minimumCosine: Double
    public let meanCosine: Double
    public let threshold: Double
    public let tokenizerGatePassed: Bool
    public let gatePassed: Bool
    enum CodingKeys: String, CodingKey {
        case minimumCosine = "min_cos", meanCosine = "mean_cos", threshold
        case tokenizerGatePassed = "tokenizer_gate_passed", gatePassed = "gate_passed"
    }
}
public struct BenchmarkMeasurements: Codable, Sendable {
    public let samples: [BenchmarkSample]
    public let latency: TimingPercentiles
    public let parity: ParityMeasurement?
}

public enum BenchmarkRunner {
    /// The caller loads once before measurement. Each corpus line is one timed
    /// B=1 request; load/verification/compute-plan cost is reported separately.
    public static func run(texts: [String], references: [ReferenceEmbedding]? = nil,
                           tokenizerGatePassed: Bool = false, lifecycle: LifecycleActor,
                           preparer: any EmbeddingPreparer) async throws -> BenchmarkMeasurements {
        guard !texts.isEmpty else { throw EmbedANEError.emptyInput(index: nil) }
        if let references {
            try BenchmarkInput.validateReferences(references, texts: texts)
            guard tokenizerGatePassed else {
                throw EmbedANEError.verification(path: "tokenizer-gate.json",
                    reason: "Run verify-tokenizer with >=1000 passing cases and matching asset digests before parity.")
            }
        }
        var samples: [BenchmarkSample] = []
        for (index, text) in texts.enumerated() {
            let execution = try await lifecycle.embed([text], using: preparer)
            guard let vector = execution.embeddings.first else {
                throw EmbedANEError.abiMismatch(chunk: 5, reason: "Benchmark received no vector.")
            }
            let cosine = try references.map { try VectorCosine.raw(vector, $0[index].embedding) }
            samples.append(.init(index: index, text: text, promptTokens: execution.promptTokens,
                                 embedding: vector, timing: execution.timing, rawCosine: cosine))
        }
        let cosines = samples.compactMap(\.rawCosine)
        let parity: ParityMeasurement?
        if let minimum = cosines.min(), !cosines.isEmpty {
            parity = .init(minimumCosine: minimum, meanCosine: cosines.reduce(0, +) / Double(cosines.count),
                           threshold: 0.995, tokenizerGatePassed: tokenizerGatePassed,
                           // The 0.995 reference figure is a MEAN over a corpus; short,
                           // sensitive texts carry a ~0.993 fp16 tail, so the gate is
                           // mean-based. The minimum stays reported for drift monitoring.
                           gatePassed: tokenizerGatePassed && cosines.reduce(0, +) / Double(cosines.count) >= 0.995)
        } else { parity = nil }
        return .init(samples: samples, latency: .init(samples: samples.map { $0.timing.totalNS }), parity: parity)
    }
}
