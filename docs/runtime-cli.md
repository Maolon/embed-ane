# CoreML runtime and CLI

Contract: [`SPEC.md`](../SPEC.md) §§3, 4, 7, 9 and 12.

This guide covers the CoreML runtimes in `EmbedANECore` and the `embed-ane` command-line tool. No third-party implementation is vendored or copied.

## Runtimes

`ServingRuntimeFactory.make(configuration:acquireLease:auditComputePlan:)` picks the runtime for a configuration:

| Installed bundle | Runtime | Serves |
|---|---|---|
| Multimodal (`vision_tower.mlmodelc` + `vision_abs_pos.fp32.npy`; position-input chunks) | `MultimodalPredictor` | text, token arrays, image, video, with one decoder cascade |
| Text-only (plain chunks) | `CoreMLRuntime` | text and token arrays; media returns `vision_not_configured` |
| Text-only plus explicitly configured vision paths | `ServingMultimodalRuntime` (plain `CoreMLRuntime` for text + `MultimodalPredictor` for media) | everything; both decoders are loaded |
| `engine_backend: coreai` (experimental) | `CoreAIRuntime` / `CoreAIMultimodalPredictor` | `.aimodel` assets through Apple's Core AI framework (macOS 27+) |

Each runtime implements `EmbeddingPredictor` and `EmbeddingPreparer` and is injected into a `LifecycleActor`, directly for benchmarks or through `EmbeddingHTTPServer` for serving. All of them use `SerialEmbeddingWorker`: one dedicated `DispatchQueue` constructs, calls and destroys the backend. Work items run inside autorelease pools, so request-scoped Objective-C buffers do not accumulate.

### Activation

1. Acquire the shared installation-use lease supplied by the composition (`ModelUseLease` from `EmbedANEDownload`; Core does not import Download).
2. Verify the whole installed bundle with `BundleVerifier`.
3. Construct the local tokenizer, map the embedding table read-only, and load each CoreML model (six chunks, plus the vision tower for multimodal bundles). For multimodal bundles the vision special tokens are checked against the tokenizer.
4. Validate every model's described inputs and outputs before publishing a usable engine.
5. Optionally inspect compute-plan assignments (`CoreMLRuntime` with `auditComputePlan: true`) and return the load report.

The backend owns the lease, tokenizer, mapping and models. A failure clears partial resources. Unload and final destruction release the models and the mapping first and the lease last, after the autorelease pool drains. A no-op lease is not appropriate for an installed production bundle.

`LifecycleActor` keeps authority over shared-load cancellation, FIFO admission, overflow, failed/backoff states, auto-load and idle eviction. Tokenization runs on the same worker, validates the whole batch, and is never repeated during prediction. A text request that reaches the preparer while the runtime is loading receives `loading` rather than waiting behind the load; with `auto_load`, the lifecycle queues unprepared text and prepares it after the load.

The lease protects against cooperating installers replacing the bundle. It is not a defense against another process running as the same user that modifies mapped files in place.

## Tensor path

`SerialCascade` executes a request with N inputs as exactly N ordered B=1 cascades. Each input first gets a CPU table lookup into fp16 `[1,512,2048]`. Chunks 0–4 receive `hidden_in` (plus `cos`/`sin` for the position-input decoder) and return `hidden_out`; the returned `MLMultiArray` is passed to the next chunk without casting or copying. Chunk 5 also receives the fp32 `[1,512]` mask and returns fp32 `[1,2048]`.

The mask is contiguous ones followed by zeros and is passed only to chunk 5. The final 2,048 floats are read using the array's strides. Swift performs neither pooling nor normalization. Shapes and types are validated again at prediction time, and non-finite or malformed outputs fail closed. Load-time validation rejects extra or missing names, wrong ranks, shapes or element types, and optional features.

For image and video requests, `VisionPreprocessor` resizes, normalizes and packs patches, interpolates absolute positions from the learned table, and computes the tower's rotary values. `VisionTowerEngine` produces up to 384 fp16 tokens per frame pair, and `VisionTokenSplicer` copies them over the placeholder rows of the looked-up hidden state before chunk 0. `Mrope` computes the decoder's multimodal positions. Budget and placeholder checks run before any graph executes.

`MappedEmbeddingTable` uses Core's strict NPY v1.0 parser and the descriptor's actual `fstat` size. It accepts only little-endian `<f2`, C order, rank 2, shape `[248078,2048]`, with the exact payload size and no trailing data. One read-only mapping lives for the loaded session. Lookup validates all 512 ids, then copies 4,096-byte rows. Padding ids are 0, so padded rows are **table row 0**, masked out at chunk 5. No fp32 copy of the table is made.

CoreML models always load with `.cpuAndNeuralEngine`. The configuration schema still accepts other `compute_units` values, but `CoreMLRuntime` rejects them at construction with an explicit `compute_units` error.

## Compute-plan audit

`ChunkComputePlanAudit` records per-chunk operation paths, operator names, preferred devices and supported devices, including nested blocks. A chunk is `checked` only when the program is nonempty and every operation has a recognized preferred device; otherwise it is `partial` or `unavailable` with a diagnostic. `compute_plan_checked` is true only when all six chunks are checked.

These are **assignments anticipated by `MLComputePlan`**, not measured utilization or proof of Neural Engine residency. Building a compute plan specializes each chunk a second time, so the app turns the audit off; the CLI keeps it for text-only bundles.

## Commands

The executable is `embed-ane`. Arguments are strict: unknown or duplicate flags, missing values, invalid ports or model ids, and unsupported combinations fail before anything runs. `--` ends option parsing. `--help` reads no configuration and needs no installed model.

```sh
swift run embed-ane fetch maolon/WeMM-Embedding-2B-CoreML-ANE     # spec.yaml from Hugging Face
swift run embed-ane fetch ./spec.yaml                               # local spec
swift run embed-ane fetch ./spec.yaml --offline
swift run embed-ane fetch ./spec.yaml --replace
swift run embed-ane verify wemm-embedding-2b-ane
swift run embed-ane verify-tokenizer ./golden.jsonl
swift run embed-ane verify-tokenizer ./golden.jsonl --assets /absolute/tokenizer-only-folder
swift run embed-ane verify-vision ./photo.jpg "a cat" --position-table ~/.embed-ane/models/wemm-embedding-2b-ane/vision_abs_pos.fp32.npy
swift run embed-ane serve --port 8080 --model-root /absolute/models
swift run embed-ane status
swift run embed-ane bench ./corpus.txt
swift run embed-ane bench ./corpus.txt --reference ./reference.jsonl --output-dir ./bench
```

- `--model-root` is accepted by every command. `--model-id` applies where it selects local assets (`serve`, `status`, `verify-tokenizer`, `verify-vision`, `bench`); `verify` takes the id as its operand and `fetch` uses the spec's id. `--port` applies to `serve` and `status`. `--offline` and `--replace` are fetch-only and mutually exclusive.
- `fetch <operand>`: an existing local file is used as the spec; otherwise an `org/name` operand reads `spec.yaml` from that Hugging Face repository (see [downloader.md](downloader.md)). `--offline` only verifies an existing install.
- `verify <id>` verifies an installed bundle without network access, under the installation lease.
- `serve` builds the runtime, verifies and loads the model through the lifecycle **before** listening, binds only `127.0.0.1`, handles SIGINT/SIGTERM, stops the listener, waits for admitted work and unloads. A bind failure does not try another port and still cleans up.
- `status` reads `/health` over loopback, refuses redirects, uses finite timeouts and a 1 MiB response limit, and prints the JSON. With no `--port` it uses the configured port. It needs no control token.

Configuration resolution: CLI flag > `EMBED_ANE_MODEL_ROOT` > `~/.embed-ane/config.yaml` > defaults. A `serve --port` override does not rewrite an existing configuration file; pass the same `--port` to `status`.

Exit codes: 0 success, 1 runtime or transport error, 2 verification failure or failed gate, 3 usage or configuration error. Reports go to stdout, errors to stderr. A completed run whose gate fails prints its report and exits 2.

## Tokenizer check

`LocalAssetTokenizer` reads only `tokenizer.json` and `tokenizer_config.json` through swift-transformers' local-folder API; Core does not import a networking module.

`verify-tokenizer` runs the golden harness (minimum 1,000 cases) and records the SHA-256 of both tokenizer files before and after the run. Besides the printed report it atomically writes a private receipt, `~/.embed-ane/tokenizer-gate.json`, with a version, date, golden-input digest, asset digests, case counts and result. A later failing run replaces an earlier pass. `--assets` points at a tokenizer-only directory without chunks, table or manifest; the default installed assets are read under the installation lease. The receipt records a run over the supplied corpus; it does not vouch for how that corpus was produced.

`bench --reference` requires a passing receipt whose tokenizer digests match the newly verified bundle before any prediction or cosine is computed. Missing, malformed, failed or stale receipts fail closed.

## Vision check

`verify-vision <image-or-video> <text>` loads a `MultimodalPredictor` directly from explicit paths (defaults: the installed bundle's `vision_tower.mlmodelc`, `chunks/`, tokenizer and table; `--position-table` is required), embeds one image or video (detected from the file type) with the given text, unloads, and prints a report: dimension, norm, the first four values, prompt and visual token counts, grid, per-stage timings and load report. `--resize` defaults to `smart`, like the server. With `--reference <vec.npy>` (fp32 `[2048]` or `[1,2048]`) it reports the cosine and exits 2 below 0.99. It is a smoke test of local artifacts, not a verified installation or a residency audit.

## Benchmark

`corpus.txt` has one nonempty text per line (LF or CRLF). Blank lines inside the corpus are rejected rather than skipped. Whitespace and Unicode inside a line are preserved. The reference JSONL has exactly one record per corpus line, in the same order:

```json
{"text":"the exact corpus line","embedding":[0.1,0.2]}
```

The vector above is abbreviated: real records must contain **2,048 finite numbers** and be nonzero. Count, text, order and vector validity are checked before any prediction. Cosine is accumulated in scaled doubles; raw values are not clipped, and negative or below-gate results are reported. The parity gate passes when the tokenizer gate has passed and the **mean cosine is ≥ 0.995**; the minimum is reported alongside. A latency-only run makes no parity claim.

Each corpus line is one B=1 request through the lifecycle. The report contains raw vectors, usage, split timings, nearest-rank p50/p95, per-case cosine and min/mean cosine when references are given. Initial verification and load, and a same-process unload and reload, are timed separately from the samples. One labeled prediction after the reload confirms the reloaded session works and is excluded from the percentiles. Provenance is checked again after the reload; a changed bundle aborts the run.

Reports are written atomically as `bench/<timestamp>-<uuid>.json` (or under `--output-dir`). They include the resolved configuration, the benchmark-only eviction-disabled setting, verified manifest, spec, commit and artifact digests, input and reference digests, the tokenizer receipt, per-chunk load timings, compute-plan records, and process RSS before load, loaded, after unload, after reload and after the final unload. RSS is process memory, not model memory. Device, OS, physical memory, installed Swift/Xcode versions and dependency pins are recorded. The first load is not claimed to be a cold-cache measurement.

## Tests

Tests live in lowercase `tests/`. Unit tests cover exact ABI metadata rejection, serial tensor identity and order, chunk-5-only masks, unchanged non-unit outputs, NPY row copying and invalid files or ids, worker serialization, the prompt template, video sampling, M-RoPE positions, vision preprocessing, benchmark gates and statistics, CLI parsing, precedence and exit codes, offline CLI wiring, tokenizer receipts, report publication, lease cleanup on failed activation, and `status` over a real `127.0.0.1:0` listener. No unit test loads a real CoreML model.

```sh
swift build --build-tests --force-resolved-versions
swift test --skip-build
swift run --skip-build embed-ane --help
```

CI compiles the CoreML sources against the macOS SDK and runs the unit tests without model files. Checks that need a real model (`bench`, `verify-vision`, `serve` against an installed bundle) are run by hand on Apple silicon with `EMBED_ANE_MODEL_ROOT` or `--model-root` pointing at an installed bundle.
