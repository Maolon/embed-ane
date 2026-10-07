import EmbedANECore
import EmbedANEHTTP
import EmbedANETestSupport
import Foundation
import Testing

@Suite("HTTP real-socket contracts", .serialized)
struct HTTPContractTests {
    @Test func healthModelsLoadUnloadStatsAndSettingsRoutes() async throws {
        try await withHTTPFixture { f in
            let health = try await f.request("GET", "/health")
            #expect(health.status == 200)
            let h = try health.json()
            #expect(h.keys == ["status", "model", "uptime_s"])
            #expect(h["status"] == .string("ok"))
            #expect(h["model"]?.keys == ["id", "state", "queue_depth", "resident_bytes"])
            #expect(h["model"]?["state"] == .string("unloaded"))
            #expect(h["model"]?["resident_bytes"] == .number(654321))
            #expect(h["uptime_s"] == .number(0))
            #expect(EmbeddingHTTPServer.bindHost == "127.0.0.1")
            let models = try await f.request("GET", "/v1/models").json()
            #expect(models == .object(["object": .string("list"), "data": .array([
                .object(["id": .string(ModelABI.defaultModelID), "object": .string("model"), "owned_by": .string("embed-ane")])
            ])]))
            let settings = try await f.request("GET", "/control/settings", auth: true).json()
            #expect(settings["idle_timeout_s"] == .number(0))
            #expect(settings["restart_required"] == .array([]))
            let loaded = try await f.request("POST", "/control/load", auth: true)
            #expect(loaded.status == 200)
            #expect(try loaded.json()["state"] == .string("ready"))
            #expect(try loaded.json()["report"]?["per_chunk_ns"]?.array?.count == 6)
            #expect(try loaded.json()["report"]?["compute_plan_checked"] == .bool(false))
            let stats = try await f.request("GET", "/control/stats", auth: true).json()
            #expect(stats["window_count"] == .number(0))
            #expect(stats["p50_ns"] == .number(0) && stats["p95_ns"] == .number(0))
            #expect(stats["resident_bytes_scope"] == .string("process"))
            let unloaded = try await f.request("POST", "/control/unload", auth: true)
            #expect(unloaded.status == 200)
            #expect(try unloaded.json()["state"] == .string("unloaded"))
            #expect(await f.predictor.loadCount == 1)
            #expect(await f.predictor.unloadCount == 1)
        }
    }

    @Test func goldenOpenAIResponseOrderingEchoAndNoRenormalization() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            let response = try await f.request("POST", "/v1/embeddings", body: embeddingBody(["a", "bb"], model: "client-alias"))
            #expect(response.status == 200 && response.headers["content-type"] == "application/json")
            let url = try #require(Bundle.module.url(forResource: "embeddings-golden", withExtension: "json", subdirectory: "Fixtures"))
            let golden = try JSONDecoder().decode(WireJSON.self, from: Data(contentsOf: url))
            #expect(try response.json() == golden)
            #expect(await f.predictor.predictCount == 1)
        }
    }

    @Test func stringInputFloatEncodingAndDimensions() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            let body = try jsonData(["model": "echo", "input": " ", "encoding_format": "float", "dimensions": 2048])
            let response = try await f.request("POST", "/v1/embeddings", body: body)
            #expect(response.status == 200)
            let result = try response.json()
            #expect(result["model"] == .string("echo"))
            #expect(result["data"]?.array?.count == 1)
            #expect(result["usage"]?["prompt_tokens"] == .number(6)) // 1 byte + 4 chat-template ids + <embedding>
        }
    }

    @Test func maximumEightBy512UsageIsExactly4096() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            // 507 content tokens + 4 chat-template ids + <embedding> = 512 per input.
            let texts = Array(repeating: String(repeating: "a", count: 507), count: 8)
            let response = try await f.request("POST", "/v1/embeddings", body: embeddingBody(texts))
            #expect(response.status == 200)
            let result = try response.json()
            #expect(result["usage"] == .object(["prompt_tokens": .number(4096), "total_tokens": .number(4096)]))
            #expect(result["data"]?.array?.count == 8)
        }
    }

    @Test func schemaAlwaysPrecedesSemanticValidation() async throws {
        try await withHTTPFixture { f in
            for raw in ["{", "null", "[]", "{}", "{\"model\":1,\"input\":\"a\"}",
                        "{\"model\":\"m\",\"input\":[true],\"encoding_format\":\"base64\"}",
                        "{\"model\":\"m\",\"input\":\"a\",\"dimensions\":null}",
                        "{\"model\":\"m\",\"input\":\"a\",\"encoding_format\":null}",
                        "{\"model\":\"m\",\"input\":\"a\",\"unknown\":1}"] {
                let response = try await f.request("POST", "/v1/embeddings", body: Data(raw.utf8))
                #expect(response.status == 400)
                #expect(try response.json()["error"]?["code"] == .string("invalid_request"))
                #expect(try response.json()["error"]?.keys == ["message", "type", "param", "code"])
            }
            #expect(await f.preparer.calls == 0)
            #expect(await f.server.lifecycle.statistics().admittedRequests == 0)
        }
    }

    @Test func semanticErrorPrecedenceAndEveryInputValidationCode() async throws {
        try await withHTTPFixture { f in
            let encoding = try jsonData(["model": "m", "input": [], "encoding_format": "hex", "dimensions": 0])
            try checkError(await f.request("POST", "/v1/embeddings", body: encoding), status: 400,
                           code: "unsupported_encoding_format", param: .string("encoding_format"))
            let dimension = try jsonData(["model": "m", "input": [], "dimensions": 0])
            try checkError(await f.request("POST", "/v1/embeddings", body: dimension), status: 400,
                           code: "unsupported_dimension", param: .string("dimensions"))
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody([] as [String])),
                           status: 400, code: "empty_input", param: .string("input"))
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody("")),
                           status: 400, code: "empty_input", param: .string("input[0]"))
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody(["a", ""])),
                           status: 400, code: "empty_input", param: .string("input[1]"))
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody(Array(repeating: "", count: 9))),
                           status: 400, code: "batch_too_large", param: .string("input"))
            #expect(await f.preparer.calls == 0)
        }
    }

    @Test func configuredMaxBatchIsEnforcedBeforeTokenization() async throws {
        try await withHTTPFixture(configuration: .init(maxBatch: 2)) { f in
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody(["a", "b", "c"])),
                           status: 400, code: "batch_too_large", param: .string("input"))
            #expect(await f.preparer.calls == 0)
        }
    }

    @Test func overlengthBatchRejectsOffendingIndexWithoutEnqueuingPrefix() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            for count in [512, 513] {
                try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody(["a", String(repeating: "b", count: count)])),
                               status: 400, code: "input_too_long", param: .string("input[1]"))
            }
            #expect(await f.predictor.predictCount == 0)
            #expect(await f.server.lifecycle.statistics().admittedRequests == 0)
            let valid = try await f.request("POST", "/v1/embeddings", body: embeddingBody("a"))
            #expect(valid.status == 200)
        }
    }

    @Test func modelNotLoadedHasNullParamAndNoRetryAfter() async throws {
        try await withHTTPFixture(configuration: .init(autoLoad: false)) { f in
            let reply = try await f.request("POST", "/v1/embeddings", body: embeddingBody())
            try checkError(reply, status: 503, code: "model_not_loaded")
            #expect(reply.headers["retry-after"] == nil)
        }
    }

    @Test func loading503HasRetryAfterAndSharedLoadStillCompletes() async throws {
        try await withHTTPFixture(configuration: .init(autoLoad: false)) { f in
            await f.predictor.loadGate.close()
            let load = Task { try await f.request("POST", "/control/load", auth: true) }
            try await eventually { await f.server.lifecycle.snapshot().state == .loading }
            let response = try await f.request("POST", "/v1/embeddings", body: embeddingBody())
            try checkError(response, status: 503, code: "loading", type: "overloaded_error")
            #expect(response.headers["retry-after"] == "5")
            await f.predictor.loadGate.open()
            #expect(try await load.value.status == 200)
            #expect(await f.predictor.loadCount == 1)
        }
    }

    @Test func unloadedRequestAutoLoadsAndServesOverLoopback() async throws {
        try await withHTTPFixture { f in
            #expect(await f.server.lifecycle.snapshot().state == .unloaded)
            let reply = try await f.request("POST", "/v1/embeddings", body: embeddingBody("auto-load test"))
            #expect(reply.status == 200)
            #expect(await f.server.lifecycle.snapshot().state == .ready)
            #expect(await f.predictor.loadCount == 1)
        }
    }

    @Test func concurrentRequestsDuringLoadShareSingleLoadAndAllServed() async throws {
        try await withHTTPFixture(configuration: .init(maxQueueDepth: 4)) { f in
            await f.predictor.loadGate.close()
            let t1 = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("req 1")) }
            try await eventually { await f.server.lifecycle.snapshot().state == .loading }
            let t2 = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("req 2")) }
            let t3 = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("req 3")) }
            try await eventually { await f.server.lifecycle.snapshot().queueDepth == 3 }
            await f.predictor.loadGate.open()
            let r1 = try await t1.value
            let r2 = try await t2.value
            let r3 = try await t3.value
            #expect(r1.status == 200)
            #expect(r2.status == 200)
            #expect(r3.status == 200)
            #expect(await f.predictor.loadCount == 1)
            #expect(await f.server.lifecycle.snapshot().state == .ready)
        }
    }

    @Test func queueDepthEnforcedDuringAutoLoad() async throws {
        try await withHTTPFixture(configuration: .init(maxQueueDepth: 2)) { f in
            await f.predictor.loadGate.close()
            let t1 = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("req 1")) }
            try await eventually { await f.server.lifecycle.snapshot().state == .loading }
            let t2 = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("req 2")) }
            try await eventually { await f.server.lifecycle.snapshot().queueDepth == 2 }
            let over = try await f.request("POST", "/v1/embeddings", body: embeddingBody("req 3"))
            try checkError(over, status: 503, code: "overloaded", type: "overloaded_error")
            #expect(over.headers["retry-after"] == "5")
            await f.predictor.loadGate.open()
            let r1 = try await t1.value
            let r2 = try await t2.value
            #expect(r1.status == 200)
            #expect(r2.status == 200)
        }
    }

    @Test func autoLoadFailureReturns503RevertsToUnloadedAndRetrySucceeds() async throws {
        try await withHTTPFixture { f in
            await f.predictor.setLoadFailures(1)
            await f.predictor.loadGate.close()
            let t1 = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("req 1")) }
            try await eventually { await f.server.lifecycle.snapshot().state == .loading }
            let t2 = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("req 2")) }
            try await eventually { await f.server.lifecycle.snapshot().queueDepth == 2 }
            await f.predictor.loadGate.open()
            let r1 = try await t1.value
            let r2 = try await t2.value
            #expect(r1.status == 503)
            #expect(r2.status == 503)
            #expect(await f.server.lifecycle.snapshot().state == .unloaded)
            let retry = try await f.request("POST", "/v1/embeddings", body: embeddingBody("retry"))
            #expect(retry.status == 200)
            #expect(await f.server.lifecycle.snapshot().state == .ready)
        }
    }

    @Test func idleEvictionStillTriggersAfterHotLoadOverHTTP() async throws {
        try await withHTTPFixture(configuration: .init(idleTimeoutS: 4)) { f in
            #expect(await f.server.lifecycle.snapshot().state == .unloaded)
            let reply = try await f.request("POST", "/v1/embeddings", body: embeddingBody("eviction test"))
            #expect(reply.status == 200)
            #expect(await f.server.lifecycle.snapshot().state == .ready)
            try await eventually { f.clock.pendingSleeps == 1 }
            f.clock.advance(seconds: 4)
            try await eventually { await f.server.lifecycle.snapshot().state == .unloaded }
            #expect(await f.predictor.unloadCount == 1)
        }
    }

    @Test func liveSettingsApplyAutoLoadImmediately() async throws {
        try await withHTTPFixture { f in
            #expect(await f.server.lifecycle.snapshot().state == .unloaded)
            let update = try await f.request("PUT", "/control/settings", body: jsonData(["auto_load": false]), auth: true)
            #expect(update.status == 200)
            let check = try await f.request("POST", "/v1/embeddings", body: embeddingBody())
            try checkError(check, status: 503, code: "model_not_loaded")
            let restore = try await f.request("PUT", "/control/settings", body: jsonData(["auto_load": true]), auth: true)
            #expect(restore.status == 200)
            let loaded = try await f.request("POST", "/v1/embeddings", body: embeddingBody("restored"))
            #expect(loaded.status == 200)
        }
    }

    @Test func queuedOverloadRetryAfterAndBusyControlUnload() async throws {
        try await withHTTPFixture(configuration: .init(maxQueueDepth: 1)) { f in
            try await f.load(); await f.predictor.predictGate.close()
            let first = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("a")) }
            try await eventually { await f.server.lifecycle.snapshot().inFlight == 1 }
            let queued = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("b")) }
            try await eventually { await f.server.lifecycle.snapshot().queueDepth == 1 }
            let overflow = try await f.request("POST", "/v1/embeddings", body: embeddingBody("c"))
            try checkError(overflow, status: 503, code: "overloaded", type: "overloaded_error")
            #expect(overflow.headers["retry-after"] == "5")
            try checkError(await f.request("POST", "/control/unload", auth: true), status: 409, code: "busy")
            let health = try await f.request("GET", "/health").json()
            #expect(health["model"]?["queue_depth"] == .number(1))
            await f.predictor.predictGate.open()
            #expect(try await first.value.status == 200)
            #expect(try await queued.value.status == 200)
            #expect(await f.predictor.recordedFirstIDs == [97, 98])
        }
    }

    @Test func conflictAndFailureAndUnexpected500Envelopes() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            await f.predictor.setPredictionFailure(.conflict("Fixture conflict."))
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody()), status: 409, code: "conflict")
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody()), status: 503, code: "failed")
            let health = try await f.request("GET", "/health").json()
            #expect(health["model"]?["state"] == .string("failed"))
        }
        try await withHTTPFixture { f in
            try await f.load(); await f.predictor.setUnexpectedFailure(true)
            let response = try await f.request("POST", "/v1/embeddings", body: embeddingBody())
            try checkError(response, status: 500, code: "internal_error")
            #expect(try response.json()["error"]?["message"] == .string("Internal server error."))
        }
    }

    @Test func invalidPredictorShapeAndNaNAreNeverSerializedAsEmbeddings() async throws {
        for mode in [MockPredictor.OutputMode.wrongDimension, .wrongBatch, .nonFinite, .wrongTimings] {
            try await withHTTPFixture { f in
                try await f.load(); await f.predictor.setOutputMode(mode)
                try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody()), status: 500, code: "abi_mismatch")
            }
        }
    }

    @Test func contentLengthOversizeRejectsBeforeWaitingForBodyOnEveryRoute() async throws {
        try await withHTTPFixture { f in
            for (method, path, auth) in [("POST", "/v1/embeddings", false), ("GET", "/health", false),
                                        ("GET", "/v1/models", false), ("PUT", "/control/settings", true),
                                        ("GET", "/control/stats", true), ("POST", "/control/load", true),
                                        ("GET", "/control/settings", true), ("POST", "/control/unload", true)] {
                try checkError(await f.request(method, path, auth: auth,
                                               framing: .declaredOnly(1_048_577)), status: 413, code: "body_too_large")
            }
            #expect(await f.preparer.calls == 0)
        }
    }

    @Test func exactOneMiBIsAcceptedAndChunkedOneByteOverIs413() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            var exact = try embeddingBody()
            exact.append(Data(repeating: 32, count: 1_048_576 - exact.count))
            #expect(try await f.request("POST", "/v1/embeddings", body: exact).status == 200)
            #expect(try await f.request("POST", "/v1/embeddings", body: exact, framing: .chunked).status == 200)
            exact.append(32)
            try checkError(await f.request("POST", "/v1/embeddings", body: exact, framing: .chunked),
                           status: 413, code: "body_too_large")
            #expect(await f.predictor.predictCount == 2)
        }
    }

    @Test func bearerOriginMatrixProtectsEveryControlRoute() async throws {
        try await withHTTPFixture { f in
            let routes = [("GET", "/control/settings"), ("PUT", "/control/settings"),
                          ("POST", "/control/load"), ("POST", "/control/unload"), ("GET", "/control/stats")]
            for (method, path) in routes {
                let data = method == "PUT" ? Data("{}".utf8) : Data()
                try checkError(await f.request(method, path, body: data), status: 401, code: "unauthorized")
                for origin in ["null", "http://evil.example", "", "https://127.0.0.1:\(f.port)"] {
                    try checkError(await f.request(method, path, body: data, auth: true, headers: [("Origin", origin)]),
                                   status: 401, code: "unauthorized")
                }
                #expect(try await f.request(method, path, body: data, auth: true).status == 200)
            }
            #expect(await f.predictor.loadCount == 1)
            #expect(await f.predictor.unloadCount == 1)
        }
    }

    @Test func sameOriginIsExactAndNativeAbsentOriginIsAllowed() async throws {
        try await withHTTPFixture { f in
            for host in ["127.0.0.1", "localhost"] {
                let origin = "http://\(host):\(f.port)"
                for authority in [host, "\(host):\(f.port)"] {
                    #expect(try await f.request("GET", "/control/settings", auth: true,
                                               headers: [("Origin", origin)], host: authority).status == 200)
                    #expect(try await f.request("GET", "/control/settings", auth: true, host: authority).status == 200)
                }
            }
            let otherPort = f.port == 65535 ? 65534 : f.port + 1
            for origin in ["http://127.0.0.1:\(otherPort)", "http://localhost:\(f.port)",
                           "http://127.0.0.1:\(f.port)/", "http://user@127.0.0.1:\(f.port)"] {
                try checkError(await f.request("GET", "/control/settings", auth: true, headers: [("Origin", origin)]),
                               status: 401, code: "unauthorized")
            }
            try checkError(await f.request("GET", "/control/settings", auth: true,
                                           headers: [("Origin", "http://127.0.0.1:\(f.port)"), ("Origin", "null")]),
                           status: 401, code: "unauthorized")
        }
    }

    @Test func malformedBearerDuplicatesAndHostAllowlist() async throws {
        try await withHTTPFixture { f in
            for value in ["Bearer wrong", "Basic x", "Bearer", f.token.authorizationHeader + " extra"] {
                try checkError(await f.request("GET", "/control/settings", headers: [("Authorization", value)]),
                               status: 401, code: "unauthorized")
            }
            try checkError(await f.request("GET", "/control/settings", auth: true,
                                           headers: [("Authorization", f.token.authorizationHeader)]),
                           status: 401, code: "unauthorized")
            for host in ["evil.example", "127.0.0.1.evil", "localhost.evil", "localhost.", "127.0.0.2", "[::1]",
                         "localhost:0", "localhost:65536", "127.0.0.1:\(f.port == 65535 ? 65534 : f.port + 1)"] {
                try checkError(await f.request("GET", "/control/settings", auth: true, host: host),
                               status: 401, code: "unauthorized")
                try checkError(await f.request("GET", "/health", host: host), status: 401, code: "unauthorized")
            }
        }
    }

    @Test func unknownControlAndOptionsNeverBypassAuthorization() async throws {
        try await withHTTPFixture { f in
            try checkError(await f.request("GET", "/control/no-such-route"), status: 401, code: "unauthorized")
            try checkError(await f.request("OPTIONS", "/control/load", headers: [("Origin", "http://evil.example")]),
                           status: 401, code: "unauthorized")
            let invalid = try await f.request("GET", "/control/load", auth: true)
            try checkError(invalid, status: 400, code: "invalid_request")
            #expect(invalid.headers["access-control-allow-origin"] == nil)
            #expect(await f.predictor.loadCount == 0)
            try checkError(await f.request("GET", "/unknown"), status: 400, code: "invalid_request")
            try checkError(await f.request("POST", "/control/load", body: Data("[]".utf8), auth: true),
                           status: 400, code: "invalid_request")
        }
    }

    @Test func liveSettingsPersistButRestartOnlyFieldsRemainEffectiveOldValues() async throws {
        try await withHTTPFixture { f in
            let patch = try jsonData(["idle_timeout_s": 7, "max_queue_depth": 2, "model_id": "next-model", "compute_units": "cpu_only"])
            let response = try await f.request("PUT", "/control/settings", body: patch, auth: true)
            #expect(response.status == 200)
            let effective = try response.json()
            #expect(effective["idle_timeout_s"] == .number(7) && effective["max_queue_depth"] == .number(2))
            #expect(effective["model_id"] == .string(ModelABI.defaultModelID))
            #expect(effective["compute_units"] == .string("cpu_and_ne"))
            #expect(effective["restart_required"] == .array([.string("compute_units"), .string("model_id")]))
            let desired = try f.store.load()
            #expect(desired.modelID == "next-model" && desired.computeUnits == .cpuOnly)
            let second = try await f.request("PUT", "/control/settings", body: jsonData(["max_queue_depth": 3]), auth: true).json()
            #expect(second["restart_required"] == effective["restart_required"])
            #expect(try f.store.load().modelID == "next-model")
            let models = try await f.request("GET", "/v1/models").json()
            #expect(models["data"]?.array?.first?["id"] == .string(ModelABI.defaultModelID))
            let reset = try jsonData(["model_id": ModelABI.defaultModelID, "compute_units": "cpu_and_ne", "idle_timeout_s": 0])
            #expect(try await f.request("PUT", "/control/settings", body: reset, auth: true).json()["restart_required"] == .array([]))
        }
    }

    @Test func liveIdleTimeoutReallyEvictsAndQueueDepthReallyChanges() async throws {
        try await withHTTPFixture { f in
            try await f.load()
            _ = try await f.request("PUT", "/control/settings", body: jsonData(["idle_timeout_s": 3, "max_queue_depth": 1]), auth: true)
            try await eventually { f.clock.pendingSleeps == 1 }
            f.clock.advance(seconds: 3)
            try await eventually { await f.server.lifecycle.snapshot().state == .unloaded }
            #expect(await f.predictor.unloadCount == 1)
            _ = try await f.request("PUT", "/control/settings", body: jsonData(["idle_timeout_s": 0]), auth: true)
            try await f.load(); await f.predictor.predictGate.close()
            let first = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("a")) }
            try await eventually { await f.server.lifecycle.snapshot().inFlight == 1 }
            let second = Task { try await f.request("POST", "/v1/embeddings", body: embeddingBody("b")) }
            try await eventually { await f.server.lifecycle.snapshot().queueDepth == 1 }
            try checkError(await f.request("POST", "/v1/embeddings", body: embeddingBody()),
                           status: 503, code: "overloaded", type: "overloaded_error")
            await f.predictor.predictGate.open()
            _ = try await first.value; _ = try await second.value
        }
    }

    @Test func invalidSettingsLeaveDiskAndRuntimeUnchanged() async throws {
        try await withHTTPFixture { f in
            let before = try f.store.load()
            for body in ["{\"max_batch\":9}", "{\"idle_timeout_s\":-1}", "{\"idle_timeout_s\":null}",
                         "{\"unknown\":1}", "{\"model_id\":\"../bad\"}", "{\"port\":0}",
                         "{\"max_queue_depth\":0}", "{\"compute_units\":\"magic\"}"] {
                let response = try await f.request("PUT", "/control/settings", body: Data(body.utf8), auth: true)
                #expect(response.status == 400)
                #expect(try response.json()["error"]?["code"] == .string("invalid_request"))
                #expect(try f.store.load() == before)
            }
            #expect(try await f.request("GET", "/control/settings", auth: true).json()["idle_timeout_s"] == .number(0))
        }
    }

    @Test func persistenceFailureDoesNotApplyLiveSettings() async throws {
        try await withHTTPFixture { f in
            try FileManager.default.removeItem(at: f.store.file)
            try FileManager.default.createDirectory(at: f.store.file, withIntermediateDirectories: false)
            let response = try await f.request("PUT", "/control/settings", body: jsonData(["idle_timeout_s": 1]), auth: true)
            try checkError(response, status: 500, code: "io_error")
            #expect(try await f.request("GET", "/control/settings", auth: true).json()["idle_timeout_s"] == .number(0))
        }
    }

    @Test func bindCollisionIsExplicitAndNeverChangesPort() async throws {
        try await withHTTPFixture { f in
            let second = try EmbeddingHTTPServer(configuration: .init(), predictor: MockPredictor(),
                                                 preparer: CountingPreparer(), store: f.store, portOverride: f.port)
            do {
                try await second.run()
                Issue.record("Second bind unexpectedly succeeded.")
            } catch {
                let text = String(describing: error)
                #expect(text.contains("127.0.0.1:\(f.port)"))
                #expect(text.contains("Address already in use") || text.contains("EADDRINUSE") || text.contains("errno: 48"))
            }
            #expect(try await f.request("GET", "/health").status == 200)
        }
    }
}
