import EmbedANETestSupport
import Foundation
import Testing
@testable import EmbedANECore

struct BenchmarkContractTests {
    @Test func orderedRawSamplesAndNearestRankUsingMockPredictor() async throws {
        let clock = ManualRuntimeClock(); let predictor = MockPredictor(clock: clock)
        let lifecycle = try LifecycleActor(predictor: predictor, clock: clock)
        _ = try await lifecycle.load()
        // Mock rows are [first content id, n_tokens]; one byte + 4 chat-template ids + <embedding> = 6.
        let references = [ReferenceEmbedding(text: "b", embedding: [98, 6] + Array(repeating: 0, count: 2046)),
                          ReferenceEmbedding(text: "a", embedding: [97, 6] + Array(repeating: 0, count: 2046))]
        let result = try await BenchmarkRunner.run(texts: ["b", "a"], references: references,
            tokenizerGatePassed: true, lifecycle: lifecycle, preparer: CountingPreparer())
        #expect(result.samples.map(\.index) == [0, 1])
        #expect(result.samples.map(\.promptTokens) == [6, 6])
        #expect(result.samples.map { $0.embedding[0] } == [98, 97])
        #expect(result.latency.p50NS == 98); #expect(result.latency.p95NS == 99)
        #expect(result.parity?.gatePassed == true)
        #expect(await predictor.predictCount == 2)
        let decoded = try JSONDecoder().decode(BenchmarkMeasurements.self, from: JSONEncoder().encode(result))
        #expect(decoded.samples[0].embedding == result.samples[0].embedding)
    }
    @Test func parityFailureReportsRawNegativeCosineRatherThanHidingIt() async throws {
        let predictor = MockPredictor(); let lifecycle = try LifecycleActor(predictor: predictor)
        _ = try await lifecycle.load()
        let reference = ReferenceEmbedding(text: "a", embedding: [-97, -6] + Array(repeating: 0, count: 2046))
        let result = try await BenchmarkRunner.run(texts: ["a"], references: [reference], tokenizerGatePassed: true,
            lifecycle: lifecycle, preparer: CountingPreparer())
        #expect(result.parity?.gatePassed == false)
        let minimum = try #require(result.parity?.minimumCosine)
        #expect(abs(minimum + 1) < 1e-12)
        #expect(result.parity?.meanCosine == result.parity?.minimumCosine)
        #expect(result.samples[0].embedding[0] == 97)
    }
    @Test func absentGateAndMismatchedReferencesRunNoPrediction() async throws {
        let predictor = MockPredictor(); let lifecycle = try LifecycleActor(predictor: predictor)
        _ = try await lifecycle.load()
        let reference = ReferenceEmbedding(text: "a", embedding: Array(repeating: 1, count: 2048))
        await #expect(throws: EmbedANEError.self) {
            try await BenchmarkRunner.run(texts: ["a"], references: [reference], lifecycle: lifecycle, preparer: CountingPreparer())
        }
        await #expect(throws: EmbedANEError.self) {
            try await BenchmarkRunner.run(texts: ["b"], references: [reference], tokenizerGatePassed: true,
                lifecycle: lifecycle, preparer: CountingPreparer())
        }
        #expect(await predictor.predictCount == 0)
    }
    @Test func latencyOnlyNeverClaimsParityPass() async throws {
        let predictor = MockPredictor(); let lifecycle = try LifecycleActor(predictor: predictor)
        _ = try await lifecycle.load()
        let result = try await BenchmarkRunner.run(texts: ["a"], lifecycle: lifecycle, preparer: CountingPreparer())
        #expect(result.parity == nil); #expect(result.samples[0].rawCosine == nil)
    }
    @Test func corpusPreservesWhitespaceUnicodeAndCRLFOrder() throws {
        #expect(try BenchmarkInput.corpus(Data(" first \r\n中文\r\n".utf8)) == [" first ", "中文"])
        #expect(throws: EmbedANEError.self) { try BenchmarkInput.corpus(Data("a\n\nb\n".utf8)) }
        #expect(throws: EmbedANEError.self) { try BenchmarkInput.corpus(Data()) }
        #expect(throws: EmbedANEError.self) { try BenchmarkInput.corpus(Data([0xff])) }
    }
    @Test func referenceSchemaCountOrderAndDimensionAreChecked() throws {
        #expect(throws: EmbedANEError.self) { try BenchmarkInput.references(Data("not json".utf8), texts: ["a"]) }
        #expect(throws: EmbedANEError.self) { try BenchmarkInput.references(Data("{\"text\":\"a\",\"embedding\":[1]}".utf8), texts: ["a"]) }
        #expect(throws: EmbedANEError.self) { try BenchmarkInput.validateReferences([], texts: ["a"]) }
    }
    @Test func cosineHandlesScaleWithoutOverflowAndRejectsInvalidVectors() throws {
        let actual: [Float] = [1e-38] + Array(repeating: 0, count: 2047)
        let reference = [Double(1e300)] + Array(repeating: 0, count: 2047)
        #expect(try VectorCosine.raw(actual, reference) == 1)
        #expect(throws: EmbedANEError.self) { try VectorCosine.raw(Array(repeating: 0, count: 2048), reference) }
        #expect(throws: EmbedANEError.self) { try VectorCosine.raw([.nan] + Array(repeating: 0, count: 2047), reference) }
        #expect(throws: EmbedANEError.self) { try VectorCosine.raw([1], [1]) }
    }
}
