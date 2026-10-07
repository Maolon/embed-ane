import Darwin
import EmbedANECore
import EmbedANEHTTP
import EmbedANETestSupport
import Foundation
import Testing

// A generated 1x1 PNG; no real-model execution is involved in these wire tests.
private let pixelPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
private let pixelURI = "data:image/png;base64," + pixelPNG
private func imageBody(url: String = pixelURI, text: String? = "a") throws -> Data {
    var parts: [[String: Any]] = [["type": "image_url", "image_url": ["url": url]]]
    if let text { parts.append(["type": "text", "text": text]) }
    return try embeddingBody(parts)
}
private func videoBody(url: String, text: String? = "a") throws -> Data {
    var parts: [[String: Any]] = [["type": "video_url", "video_url": ["url": url]]]
    if let text { parts.append(["type": "text", "text": text]) }
    return try embeddingBody(parts)
}
private func decodeBase64Vector(_ value: WireJSON?) throws -> [Float] {
    let string = try #require(value?.string)
    let data = try #require(Data(base64Encoded: string))
    #expect(data.count.isMultiple(of: 4))
    return stride(from: 0, to: data.count, by: 4).map { offset in
        let bits = (0..<4).reduce(UInt32(0)) { $0 | UInt32(data[offset + $1]) << ($1 * 8) }
        return Float(bitPattern: bits)
    }
}

@Suite("Embeddings wire extensions", .serialized)
struct ExtendedEmbeddingsTests {
    @Test func tokenFormsAreCanonicalOrderedAndNeverRetokenized() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            // Token arrays are content: wrapped as <|im_start|> user \n ids <|im_end|>, then <embedding>.
            let single = try await f.request("POST", "/v1/embeddings", body: embeddingBody([7, 0, 248077])).json()
            #expect(single["usage"]?["prompt_tokens"] == .number(8))
            #expect(single["data"]?.array?.first?["embedding"]?.array?.prefix(2) == [.number(7), .number(8)])
            #expect(await f.predictor.recordedInputIDs.last == [248045, 846, 198, 7, 0, 248077, 248046, 248077])
            let batch = try await f.request("POST", "/v1/embeddings", body: embeddingBody([[8], [9, 10]])).json()
            #expect(batch["usage"]?["total_tokens"] == .number(13))
            #expect(batch["data"]?.array?.map { $0["index"] } == [.number(0), .number(1)])
            #expect(batch["data"]?.array?.map { $0["embedding"]?.array?.first } == [.number(8), .number(9)])
            #expect(await f.preparer.calls == 0)
            let maximum = Array(repeating: Array(repeating: 0, count: 507), count: 8)
            #expect(try await f.request("POST", "/v1/embeddings", body: embeddingBody(maximum)).json()["usage"]?["prompt_tokens"] == .number(4096))
        }
    }

    @Test func invalidTokenBatchNeverAdmitsPrefix() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            for ids in [[-1], [248078]] {
                try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody([[1], ids])),
                               status: 400, code: "invalid_request", param: .string("input[1]"))
            }
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody([[1], []])),
                           status: 400, code: "empty_input", param: .string("input[1]"))
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody([[1], Array(repeating: 0, count: 508)])), // 507 + wrapper fills 512
                           status: 400, code: "input_too_long", param: .string("input[1]"))
            for raw in ["[1.5]", "[true]", "[1,\"a\"]", "[[1],2]", "null", "[null]"] {
                let response = try await f.request("POST", "/v1/embeddings", body: Data("{\"model\":\"m\",\"input\":\(raw)}".utf8))
                #expect(response.status == 400)
                #expect(try response.json()["error"]?["code"] == .string("invalid_request"))
            }
            #expect(await f.server.lifecycle.statistics().admittedRequests == 0)
            #expect(await f.predictor.predictCount == 0)
        }
    }

    @Test func base64IsLittleEndianFloat32AndFullDimensionOutputIsUnchanged() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            for dimensions in [nil, 2048] as [Int?] {
                var body: [String: Any] = ["model": "m", "input": [[97], [98, 99]], "encoding_format": "base64", "user": "sdk-user"]
                if let dimensions { body["dimensions"] = dimensions }
                let response = try await f.request("POST", "/v1/embeddings", body: jsonData(body))
                #expect(response.status == 200)
                let json = try response.json()
                let rows = try #require(json["data"]?.array)
                #expect(try decodeBase64Vector(rows[0]["embedding"]) == [97, 6] + Array(repeating: 0, count: 2046))
                #expect(try decodeBase64Vector(rows[1]["embedding"]) == [98, 7] + Array(repeating: 0, count: 2046))
            }
        }
    }

    @Test func explicitShorteningNormalizesOnlyPrefixInBothFormats() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            for dimension in [1, 2, 17, 2047] {
                for format in ["float", "base64"] {
                    // Mock row [9, n_tokens 12]: 7 ids + 4 chat-template ids + <embedding>; normalizes to 0.6/0.8.
                    let body = try jsonData(["model": "m", "input": [9, 1, 1, 1, 1, 1, 1], "dimensions": dimension, "encoding_format": format])
                    let response = try await f.request("POST", "/v1/embeddings", body: body)
                    #expect(response.status == 200)
                    let embedding = try response.json()["data"]?.array?.first?["embedding"]
                    let vector: [Float]
                    if format == "base64" { vector = try decodeBase64Vector(embedding) }
                    else { vector = try #require(embedding?.array).compactMap { $0.number.map(Float.init) } }
                    #expect(vector.count == dimension)
                    let squaredSum: Double = vector.reduce(0.0) { $0 + Double($1) * Double($1) }
                    #expect(abs(squaredSum - 1) < 1e-6)
                    let expectedFirst: Float = dimension == 1 ? 1 : 0.6
                    #expect(abs(vector[0] - expectedFirst) < 1e-6)
                    if dimension >= 2 {
                        let secondDelta: Double = Double(vector[1] - 0.8)
                        #expect(abs(secondDelta) < 1e-6)
                    }
                    #expect(vector.dropFirst(2).allSatisfy { $0 == 0 })
                }
            }
            // A zero prefix cannot be normalized; do not return NaNs or poison the model.
            let zero = try jsonData(["model": "m", "input": [0], "dimensions": 1])
            try checkError(await f.request("POST", "/v1/embeddings", body: zero), status: 400,
                           code: "invalid_request", param: .string("dimensions"))
            #expect(await f.server.lifecycle.snapshot().state == .ready)
            for dimension in [0, -1, 2049] {
                try checkError(await f.request("POST", "/v1/embeddings", body: jsonData(["model": "m", "input": "a", "dimensions": dimension])),
                               status: 400, code: "unsupported_dimension", param: .string("dimensions"))
            }
        }
    }

    @Test func contentPartsProduceOneJointVectorAndHonestUsage() async throws {
        let vision = MockMultimodalPredictor()
        try await withHTTPFixture(vision: vision) { f in
            try await f.load()
            let response = try await f.request("POST", "/v1/embeddings", body: imageBody(text: "hi"))
            #expect(response.status == 200)
            let json = try response.json()
            #expect(json["data"]?.array?.count == 1)
            #expect(json["usage"] == .object(["prompt_tokens": .number(67), "total_tokens": .number(67)]))
            #expect(json["data"]?.array?.first?["embedding"]?.array?.prefix(2) == [.number(3), .number(4)])
            let captured = try #require(await vision.imageRequests.first)
            #expect(captured.imageData == Data(base64Encoded: pixelPNG)); #expect(captured.text == "hi")
            #expect(await f.predictor.predictCount == 0)
            #expect(try await f.request("POST", "/v1/embeddings", body: imageBody(text: nil)).json()["usage"]?["prompt_tokens"] == .number(65))
            let textParts = [["type": "text", "text": "a"], ["type": "text", "text": "b"]]
            let plain = try await f.request("POST", "/v1/embeddings", body: embeddingBody(textParts)).json()
            #expect(plain["usage"]?["prompt_tokens"] == .number(7)) // "ab" + 4 chat-template ids + <embedding>
            #expect(await vision.imageRequests.count == 2)
            #expect(await f.predictor.predictCount == 1)
        }
    }

    @Test func visionUnconfiguredAndMalformedPartsHaveStableErrors() async throws {
        try await withHTTPFixture { f in
            let unavailable = try await f.request("POST", "/v1/embeddings", body: imageBody())
            try checkError(unavailable, status: 501, code: "vision_not_configured")
            let unavailableJSON = try unavailable.json()
            let message = try #require(unavailableJSON["error"]?["message"]?.string)
            for key in ["vision_tower_path", "vision_extrope_chunks_dir", "vision_position_table_path"] { #expect(message.contains(key)) }
            for url in ["https://example.com/p.png", "http://127.0.0.1:1/p.png"] {
                let response = try await f.request("POST", "/v1/embeddings", body: imageBody(url: url))
                try checkError(response, status: 400, code: "invalid_request", param: .string("input"))
                #expect(try response.json()["error"]?["message"]?.string?.contains("Remote image fetch not supported") == true)
            }
            for url in ["/local.png", "file://host/p.png", "file:relative", "file:///a/../b.png", "file:///a%00.png",
                        "file:///a.png?q=1", "data:text/plain;base64,YQ==", "data:image/png;base64,%%%", "data:image/png;base64,"] {
                try checkError(await f.request("POST", "/v1/embeddings", body: imageBody(url: url)), status: 400,
                               code: "invalid_request", param: .string("input"))
            }
            for parts in [
                [["type": "audio", "text": "x"]],
                [["type": "image_url", "image_url": ["url": pixelURI]], ["type": "image_url", "image_url": ["url": pixelURI]]]
            ] as [[[String: Any]]] {
                try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody(parts)), status: 400,
                               code: "invalid_request", param: .string("input"))
            }
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody([["type": "text", "text": ""]])),
                           status: 400, code: "empty_input", param: .string("input"))
            try await f.load()
            #expect(try await f.request("POST", "/v1/embeddings", body: embeddingBody([["type": "text", "text": "a"]])).status == 200)
        }
    }

    @Test func localFilesAreBoundedAndRejectSymlinksDirectoriesAndFIFOs() async throws {
        let vision = MockMultimodalPredictor()
        try await withHTTPFixture(vision: vision) { f in
            try await f.load()
            // Foundation preserves the system /var alias even after resolving;
            // realpath gives the no-symlink /private/var path required by the reader.
            let canonical = try #require(realpath(f.store.home.path, nil))
            defer { free(canonical) }
            let directory = URL(fileURLWithPath: String(cString: canonical))
            let file = directory.appendingPathComponent("test image.png")
            try #require(Data(base64Encoded: pixelPNG)).write(to: file)
            let successfulFile = try await f.request("POST", "/v1/embeddings", body: imageBody(url: file.absoluteString))
            #expect(successfulFile.status == 200)
            let link = directory.appendingPathComponent("link.png")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
            let fifo = directory.appendingPathComponent("fifo")
            #expect(mkfifo(fifo.path, 0o600) == 0)
            let huge = directory.appendingPathComponent("huge.png")
            #expect(FileManager.default.createFile(atPath: huge.path, contents: nil))
            let handle = try FileHandle(forWritingTo: huge); try handle.truncate(atOffset: 64 * 1_024 * 1_024 + 1); try handle.close()
            for url in [directory, link, fifo, huge, directory.appendingPathComponent("missing.png")] {
                try checkError(await f.request("POST", "/v1/embeddings", body: imageBody(url: url.absoluteString)), status: 400,
                               code: "invalid_request", param: .string("input"))
            }
            #expect(await vision.imageRequests.count == 1)
        }
    }

    @Test func imageAndTextShareFIFOOverloadAndBusyUnload() async throws {
        let vision = MockMultimodalPredictor()
        try await withHTTPFixture(configuration: .init(maxQueueDepth: 1), vision: vision) { f in
            try await f.load(); await f.predictor.predictGate.close(); await vision.imageGate.close()
            let text = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("a")) }
            try await eventually { await f.server.lifecycle.snapshot().inFlight == 1 }
            let image = Task { try await f.request("POST", "/v1/embeddings", body: imageBody()) }
            try await eventually { await f.server.lifecycle.snapshot().queueDepth == 1 }
            #expect(await vision.imageRequests.isEmpty)
            try checkError(await f.request("POST", "/v1/embeddings", body: imageBody()), status: 503, code: "overloaded", type: "overloaded_error")
            await f.predictor.predictGate.open()
            #expect(try await text.value.status == 200)
            try await eventually { await vision.imageRequests.count == 1 }
            try checkError(await f.request("POST", "/control/unload", auth: true), status: 409, code: "busy")
            let second = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("b")) }
            try await eventually { await f.server.lifecycle.snapshot().queueDepth == 1 }
            #expect(await f.predictor.predictCount == 1)
            await vision.imageGate.open()
            #expect(try await image.value.status == 200); #expect(try await second.value.status == 200)
            #expect(await f.server.lifecycle.statistics().completedRequests == 3)
        }
    }

    @Test func badImageDoesNotPoisonReadyOrRejectQueuedText() async throws {
        let vision = MockMultimodalPredictor()
        try await withHTTPFixture(vision: vision) { f in
            try await f.load()
            await vision.setFailure(.invalidRequest("Cannot decode image pixels.", param: "image")); await vision.imageGate.close()
            let image = Task { try await f.request("POST", "/v1/embeddings", body: imageBody()) }
            try await eventually { await vision.imageRequests.count == 1 }
            let text = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("ok")) }
            try await eventually { await f.server.lifecycle.snapshot().queueDepth == 1 }
            await vision.imageGate.open()
            try checkError(await image.value, status: 400, code: "invalid_request", param: .string("image"))
            #expect(try await text.value.status == 200)
            #expect(await f.server.lifecycle.snapshot().state == .ready)
            #expect(await f.server.lifecycle.statistics().failedRequests == 1)
            await vision.setFailure(.inputTooLong(index: 0, contentTokens: 520))
            try checkError(await f.request("POST", "/v1/embeddings", body: imageBody()), status: 400, code: "input_too_long", param: .string("input[0]"))
            #expect(await f.server.lifecycle.snapshot().state == .ready)
            await vision.setFailure(nil); await vision.setInvalidOutput(true)
            try checkError(await f.request("POST", "/v1/embeddings", body: imageBody()), status: 500, code: "abi_mismatch")
            #expect(await f.server.lifecycle.snapshot().state == .failed)
        }
    }
    @Test func videoPartsRouteLocalFilesToTheJointPathAndRejectUnsafeSources() async throws {
        let vision = MockMultimodalPredictor()
        try await withHTTPFixture(vision: vision) { f in
            try await f.load()
            let canonical = try #require(realpath(f.store.home.path, nil))
            defer { free(canonical) }
            let directory = URL(fileURLWithPath: String(cString: canonical))
            let file = directory.appendingPathComponent("test clip.mp4")
            try Data("not decoded by the mock predictor".utf8).write(to: file)

            // Text parts are concatenated and travel with the video; the video comes
            // before the text regardless of part order.
            let parts: [[String: Any]] = [["type": "text", "text": "h"], ["type": "video_url", "video_url": ["url": file.absoluteString]],
                                          ["type": "text", "text": "i"]]
            let response = try await f.request("POST", "/v1/embeddings", body: embeddingBody(parts))
            #expect(response.status == 200)
            let json = try response.json()
            #expect(json["data"]?.array?.count == 1)
            #expect(json["usage"]?["prompt_tokens"] == .number(67))
            let captured = try #require(await vision.imageRequests.first)
            #expect(captured.media == .video(URL(fileURLWithPath: file.path)))
            #expect(captured.videoURL?.path == file.path && captured.imageData == nil)
            #expect(captured.text == "hi")
            #expect(await f.predictor.predictCount == 0)
            // A video alone is accepted.
            #expect(try await f.request("POST", "/v1/embeddings", body: videoBody(url: file.absoluteString, text: nil)).status == 200)

            // Remote video URLs are not fetched.
            for url in ["https://example.com/v.mp4", "http://127.0.0.1:1/v.mp4"] {
                let remote = try await f.request("POST", "/v1/embeddings", body: videoBody(url: url))
                try checkError(remote, status: 400, code: "invalid_request", param: .string("input"))
                #expect(try remote.json()["error"]?["message"]?.string?.contains("Remote video fetch not supported") == true)
            }
            // data: URLs are rejected for video, even with a video media type.
            let inline = try await f.request("POST", "/v1/embeddings", body: videoBody(url: "data:video/mp4;base64,AAAA"))
            try checkError(inline, status: 400, code: "invalid_request", param: .string("input"))
            #expect(try inline.json()["error"]?["message"]?.string?.contains("data: URLs are not supported") == true)

            // Symlinks, directories, FIFOs, missing files and malformed file URLs.
            let link = directory.appendingPathComponent("link.mp4")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: file)
            let fifo = directory.appendingPathComponent("fifo.mp4")
            #expect(mkfifo(fifo.path, 0o600) == 0)
            let empty = directory.appendingPathComponent("empty.mp4")
            #expect(FileManager.default.createFile(atPath: empty.path, contents: nil))
            for url in [link, directory, fifo, empty, directory.appendingPathComponent("missing.mp4")] {
                try checkError(await f.request("POST", "/v1/embeddings", body: videoBody(url: url.absoluteString)), status: 400,
                               code: "invalid_request", param: .string("input"))
            }
            for url in ["/local.mp4", "file://host/v.mp4", "file:relative", "file:///a/../b.mp4", "file:///a.mp4?q=1"] {
                try checkError(await f.request("POST", "/v1/embeddings", body: videoBody(url: url)), status: 400,
                               code: "invalid_request", param: .string("input"))
            }

            // At most one image OR video per request.
            for parts in [
                [["type": "image_url", "image_url": ["url": pixelURI]], ["type": "video_url", "video_url": ["url": file.absoluteString]]],
                [["type": "video_url", "video_url": ["url": file.absoluteString]], ["type": "video_url", "video_url": ["url": file.absoluteString]]]
            ] as [[[String: Any]]] {
                let both = try await f.request("POST", "/v1/embeddings", body: embeddingBody(parts))
                try checkError(both, status: 400, code: "invalid_request", param: .string("input"))
                #expect(try both.json()["error"]?["message"]?.string == "Only one image or video per request is supported.")
            }
            // Unknown part types and stray keys.
            let unknown = try await f.request("POST", "/v1/embeddings", body: embeddingBody([["type": "audio_url", "audio_url": ["url": file.absoluteString]]]))
            try checkError(unknown, status: 400, code: "invalid_request", param: .string("input"))
            #expect(try unknown.json()["error"]?["message"]?.string == "Content parts must have type text, image_url or video_url.")
            // Strict keys, as for image_url parts: an image_url key in a video part is unknown.
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody([["type": "video_url", "image_url": ["url": file.absoluteString]]])),
                           status: 400, code: "invalid_request")
            #expect(await vision.imageRequests.count == 2)
        }
    }

    @Test func videoWithoutVisionIsNotConfigured() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            let response = try await f.request("POST", "/v1/embeddings", body: videoBody(url: "file:///tmp/clip.mp4"))
            try checkError(response, status: 501, code: "vision_not_configured")
            let message = try #require(try response.json()["error"]?["message"]?.string)
            #expect(message.contains("vision_tower.mlmodelc"))
        }
    }
}
