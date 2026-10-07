# embed-ane — Technical Specification

This document is the technical contract for embed-ane: the module layout, the CoreML model ABI, the model bundle format, the HTTP API, configuration, model lifecycle and the menu bar app. Code changes that alter any behavior described here must update this file in the same pull request. If you think a rule here is wrong, open an issue that describes the conflict instead of silently deviating from it.

Topic guides with more detail:

- [docs/downloader.md](docs/downloader.md): model acquisition and verification (`EmbedANEDownload`)
- [docs/http-api.md](docs/http-api.md): HTTP routes, lifecycle, control security and settings (`EmbedANEHTTP`)
- [docs/runtime-cli.md](docs/runtime-cli.md): the CoreML runtime and the `embed-ane` CLI (`EmbedANECore`, `embed-ane`)
- [docs/app.md](docs/app.md): the menu bar app (`App/`)

---

## 0. Scope

embed-ane is a macOS menu bar app and a headless CLI that serve [WeMM-Embedding-2B](https://huggingface.co/tencent/WeMM-Embedding-2B) from the Apple Neural Engine through an OpenAI-compatible `POST /v1/embeddings` endpoint on `127.0.0.1`. The model is a 2,048-dimensional multimodal embedding model; embed-ane runs its decoder as six precompiled CoreML chunks and its vision encoder as a separate CoreML model.

- Inputs: text, token arrays, one image with optional text, or one video with optional text.
- Lifecycle: keep the model loaded by default; idle eviction is opt-in; requests that arrive while the model is unloaded load it on demand (`auto_load`).
- Model acquisition: explicit, digest-pinned downloads from a Hugging Face repository described by a YAML spec.

**Out of scope:** Linux and Intel Macs, automatic update polling, App Store distribution, batched CoreML prediction (requests run as serial batch-size-1 cascades), and iOS.

## 1. Design choices

| Area | Choice |
|---|---|
| Language | All Swift. Tokenization uses swift-transformers; a golden differential check (≥1,000 cases) confirms it matches the Hugging Face tokenizer before any parity number is trusted. |
| Packaging | SPM targets `EmbedANECore`, `EmbedANEHTTP`, `EmbedANEDownload`, `EmbedANEAppSupport`, the `embed-ane` CLI, and the `App/` Xcode target. Core imports no UI or networking code. |
| Lifecycle | Keep-loaded by default, opt-in idle eviction, one shared load at a time (single-flight). |
| Concurrency | One serial worker owns all CoreML models. Admission is a bounded FIFO. Cancelling one waiter never cancels a shared load or prediction. |
| Downloads | Explicit trigger only; immutable revision (commit or tag, never a branch); stage, verify, then atomically promote; resumable. |
| Claims | Documentation states no latency, memory or residency numbers beyond what `embed-ane bench` records. |

**Parity gates.** `embed-ane bench --reference` always reports raw minimum and mean cosine. The release gate is **mean cosine ≥ 0.995** against reference embeddings produced by the Python coremltools ANE path; the minimum is reported for drift monitoring. Compared with the Hugging Face fp32 model, cosine around 0.957 is the expected band for the fp16 decoder; it is reported for information and never gated. `embed-ane verify-vision --reference` gates a single image or video embedding at cosine ≥ 0.99.

## 2. Toolchain and dependencies

| Component | Pin |
|---|---|
| Swift tools / language mode | 6.1 / Swift 6 strict concurrency |
| Platform | macOS 15+ on Apple silicon (`arm64`) only |
| Xcode (App target, CI) | 26 or later (needed for the Icon Composer app icon) |
| Hummingbird | **2.26.0** (HTTP module only) |
| swift-transformers (Tokenizers) | **1.3.4** |
| Yams | **6.2.2** |

Dependencies are pinned exactly and `Package.resolved` is committed. Open an issue before bumping a pin.

Targets: `EmbedANECore` (depends on Tokenizers, Yams), `EmbedANEHTTP` (Core + Hummingbird), `EmbedANEDownload` (Core + Yams), `EmbedANEAppSupport` (`App/Support`; Core + HTTP + Download), executable `embed-ane`, and one test target per module under `tests/`. **Dependency direction: HTTP → Core, Download → Core. Core depends on nothing else in this package.** Digest verification lives in Core because both the downloader and load-time activation use it.

## 3. CoreML ABI

The runtime refuses any model whose described inputs and outputs differ from this ABI. Validation runs at load time, before the first prediction, and a mismatch is a typed `abi_mismatch` load error.

1. **Batching is a serial B=1 loop.** Every graph takes exactly one `[1,512,2048]` fp16 hidden state. A request with N inputs runs N ordered single-item cascades. There is no batched CoreML prediction.
2. **Decoder chunks.** Chunks 0–4: input `hidden_in` (fp16 `[1,512,2048]`) → output `hidden_out` (same shape and type), handed to the next chunk unchanged. Chunk 5: inputs `hidden_in` and `attention_mask` (**fp32 `[1,512]`**, contiguous ones then zeros, fed to chunk 5 only) → output `embedding` (**fp32 `[1,2048]`**, already last-token pooled **and** L2-normalized inside the graph; Swift does not normalize again).
3. **Position-input decoder.** Multimodal bundles ship a decoder variant whose chunks also take `cos` and `sin` (fp16 `[1,512,64]`). The runtime computes these on the CPU from 3-D multimodal rotary positions (Qwen3.5 M-RoPE: 256-dim heads × 0.25 partial rotary = 64 dims, sections `[11,11,10]`, θ = 10,000,000). Text advances one position per token; each visual span is placed on a (t, h, w) grid. The same chunks serve text-only requests.
4. **Vision tower.** One two-frame CoreML model with fp16 inputs `patches [1536,1536]` (each patch packs C×T×16×16 = 3×2×16×16), `abs_pos [1536,1024]`, `key_mask [1536]`, `cos [1536,64]`, `sin [1536,64]`, and fp16 output `var_2722 [384,2048]` (2×2 patch merge, at most 384 visual tokens). A still image is fed as two identical frames. Absolute positions are bilinearly interpolated from the learned `[2304,1024]` fp32 table (48×48 grid). Single-frame towers are rejected.
5. **Embedding table.** A CPU lookup maps token ids to fp16 `[1,512,2048]` before chunk 0. `embed_table.fp16.npy` must be **NPY v1.0, little-endian `<f2`, C order, rank 2, `[248078, 2048]`** (payload 1,016,127,488 bytes, no trailing data). The parser rejects NPY v2/v3, Fortran order, other dtypes, truncated files and wrong shapes, and checks every id `< 248078`. The file is memory-mapped read-only once per loaded session; no fp32 copy is made. Visual tokens from the tower overwrite the placeholder rows after the lookup.
6. **Compute units.** CoreML models load with `.cpuAndNeuralEngine`. The CoreML runtime rejects any other `compute_units` setting rather than running a different policy.
7. **Memory.** The predict path allocates no persistent per-request buffers beyond request-scoped `MLMultiArray`s, inside autorelease pools.

## 4. Tokenizer and prompt

Assets: `tokenizer.json` and `tokenizer_config.json` from the model bundle, loaded from the local folder with swift-transformers (no Hub access). Loading fails unless `<embedding>` is id 248077 and the chat-template tokens `<|im_start|>` (248045), `<|im_end|>` (248046), `user` (846) and the newline token (198) match.

1. **Prompt.** Every input uses the upstream WeMM chat template for one user turn, followed by the special `<embedding>` token:

   ```
   text:   <|im_start|>user\n{text}<|im_end|><embedding>
   image:  <|im_start|>user<|vision_start|><|image_pad|>×N<|vision_end|>{text}<|im_end|><embedding>
   video:  <|im_start|>user\n<0.3 seconds><|vision_start|><|video_pad|>×N<|vision_end|><1.3 seconds>…{text}<|im_end|><embedding>
   ```

   As in the upstream processor, there is no role newline before a leading image. Text that starts with whitespace containing a newline is tokenized together with the role newline, matching how the upstream tokenizer merges them. Token-array inputs are treated as content ids and get the same wrapper as text.
2. **Length.** A sequence holds 512 tokens, specials included. Text and token-array inputs may use **507 content tokens** (512 minus `<embedding>` and the four template tokens). Longer input fails with `input_too_long`; nothing is silently truncated. For image and video inputs, visual tokens, text and template must fit in 512 together.
3. **Reserved tokens.** Text that tokenizes to a vision placeholder, `<|vision_start|>`, `<|vision_end|>`, a chat-template token or `<embedding>` is rejected on the multimodal path; pass media as content parts.
4. **Padding.** Right-pad to 512 with id 0 and an fp32 mask of ones then zeros. (The configured pad token is not used; padded rows are table row 0, masked out at chunk 5.)
5. **Golden differential** (`embed-ane verify-tokenizer`): JSONL with one `{"text", "ids", "mask", "n_tokens"}` object per line, produced by the Hugging Face tokenizer over raw content (no template). The check passes with **zero id, mask or count mismatches over ≥1,000 cases**; overlength records must match raw tokenization and be rejected. It needs only the two tokenizer files.

## 5. Model spec YAML and install manifest

```yaml
spec_version: 1
model:   { id: wemm-embedding-2b-ane, dim: 2048, max_seq: 512, normalize: l2 }
source:
  repo: <org>/<name>
  revision: <40-hex commit SHA | tag name>     # branches are rejected at resolve time
  endpoint: https://huggingface.co              # optional; this is the default
  auth: env:HF_TOKEN                            # optional; local specs only
files:                                          # flat, explicit list; no globs
  - { path: tokenizer.json,                       sha256: <hex>, size: N }
  - { path: tokenizer_config.json,                sha256: <hex>, size: N }
  - { path: embed_table.fp16.npy,                 sha256: <hex>, size: N }
  - { path: vision_abs_pos.fp32.npy,              sha256: <hex>, size: N }   # multimodal only
  - { path: vision_tower.mlmodelc/manifest.txt,   sha256: <hex>, size: N }   # multimodal only
  - { path: chunks/chunk0.mlmodelc/manifest.txt,  sha256: <hex>, size: N }
  # ... chunks/chunk1..5.mlmodelc/manifest.txt
runtime: { compute_units: cpu_and_ne, pad_side: right, mask_dtype: fp32 }
```

**Directories are described by inner manifests.** Hugging Face `/resolve` serves files, so every `.mlmodelc/` directory is represented by an inner manifest file `<dir>.mlmodelc/manifest.txt`, uploaded to the repository next to the files it lists: UTF-8, one `<relative-path> <size> <sha256>` line per contained file, sorted, LF endings. The spec digests the manifest; the downloader and verifier expand every `*/manifest.txt` entry, fetch and verify each listed file, and reject any unlisted file in the directory. Paths must be relative, without `..`, absolute components, symlinks or duplicates. Spec entries inside a `.mlmodelc/` other than its `manifest.txt` are rejected.

**Required entries:** both tokenizer files, the embedding table and all six `chunks/chunk{0..5}.mlmodelc/manifest.txt`. `vision_tower.mlmodelc/manifest.txt` and `vision_abs_pos.fp32.npy` must be present together or both absent. Their presence makes the bundle multimodal, and `chunks/` then holds the position-input decoder (§3.3). Without them the bundle is text-only and `chunks/` holds the plain decoder.

**Schema strictness:** unknown keys and duplicate YAML keys are rejected; `spec_version` must be 1; `model.id` must match `[a-z0-9-]{1,64}`; `dim`, `max_seq`, `normalize` and the `runtime` block must equal the values above; `source.repo` must be `org/name`; `source.endpoint` must be HTTPS (plain HTTP only to `127.0.0.1`, for tests); `auth` must be `env:<NAME>`, never a literal credential. `manifest.yaml`, `RESOLVED`, `SPEC_SHA256` and `.part`/`.etag` names are reserved. Revisions decode to `.commit(sha)` or `.tag(name)` and are resolved before any download.

**Install manifest.** On promotion the downloader writes `manifest.yaml`: manifest version, spec digest, resolved commit SHA, per-entry digests, generator, creation time, and the exact spec text. The installed bundle also contains `spec.yaml`.

`scripts/make-model-bundle.py` assembles a bundle in this layout (inner manifests, `spec.yaml`, and optionally a verified local install for testing).

## 6. Downloader (`EmbedANEDownload`)

`fetch(specPath:modelRoot:offline:replace:progress:)`:

1. **Resolve the revision first.** Query `<endpoint>/api/models/<repo>/revision/<rev>`. A branch is rejected with a typed error naming it. A tag or SHA resolves to a 40-hex commit, which is persisted in `.staging/<id>/RESOLVED`. A resume only continues against that commit; a spec change or commit drift discards the staging directory.
2. **Stage.** Files download into `<modelRoot>/.staging/<id>/` from `GET <endpoint>/<repo>/resolve/<sha>/<path>` (with a Bearer token when the spec declares `auth`). Transfers resume with HTTP Range. Partial and complete files survive recoverable interruptions; only files that fail their digest or belong to a stale commit are deleted. `206` appends; `200` after a Range request restarts the file; `416` restarts the file; a changed validator (ETag/Last-Modified) restarts the file.
3. **Verify, then promote.** All digests (including every inner manifest's members) are verified, `manifest.yaml` is written, and `.staging/<id>` is renamed atomically to `<modelRoot>/<id>`. Any failure leaves nothing promoted and exits non-zero, naming the offending path.
4. **Concurrency and idempotence.** A per-id file lock guards fetches; a concurrent fetch of the same id fails with `conflict`. An existing, verified, identical install is a no-op success. A different bundle under the same id is a `conflict`; replacing an install requires `--replace`, which swaps directories atomically and never touches an install that is in use.
5. **Offline.** `fetch --offline` and `verify <id>` only verify the installed bundle and make no network connections.
6. **Transport.** HTTPS only in production (tests may use loopback `http://127.0.0.1:<port>` through an injectable transport). Tokens are never logged; the Bearer header is dropped on cross-origin redirects.
7. **Remote specs.** `embed-ane fetch org/name` and the app's Hugging Face download first read `spec.yaml` from the repository. A remote spec must name the same repository and endpoint it was fetched from and must not declare `auth`; only trusted local spec files may name a credential variable. The token comes from `HF_TOKEN`, or from `~/.embed-ane/hf-token` (mode 0600) when the variable is unset, and is only sent to `huggingface.co` hosts.

## 7. Core runtime (`EmbedANECore`)

```swift
public protocol EmbeddingPredictor: Sendable {
  func load() async throws -> LoadReport        // per-chunk timings, process RSS, compute-plan audit flag
  func predict(_ request: PredictRequest) async throws -> PredictResult
  func unload() async throws -> UnloadReport    // post-unload process RSS
}
```

`MultimodalEmbeddingPredictor` adds `predictImage(_:)` for one image-or-video request with optional text. `ServingRuntimeFactory.make` builds the serving runtime from the configuration: for a multimodal bundle a single `MultimodalPredictor` (position-input decoder plus tower) serves text, image and video; for a text-only bundle `CoreMLRuntime` serves text and image requests return `vision_not_configured`.

**Execution has two layers:**

- `LifecycleActor` handles admission, the state machine and the queue. States: `unloaded → loading → ready → unloading`, plus `failed` with exponential backoff (1 s doubling to 60 s, reset on success). Admission is a bounded FIFO (`max_queue_depth`, default 16) with at most `max_batch` (1–8, default 8) inputs per request. A full queue returns `overloaded`. Image and video requests share the same queue and rules.
- A **dedicated serial worker** (its own `DispatchQueue`, not the cooperative thread pool) constructs, calls and releases all CoreML models, the table and the tokenizer. Cancelling one waiter never cancels a shared load or prediction; cancellation is checked only at admission.

**Load on demand.** With `auto_load: true` (default), a request that arrives while the model is unloaded starts the shared load and waits in the queue; requests arriving during the load queue up to `max_queue_depth`. If an automatic load fails, the state returns to `unloaded`, waiting requests receive 503 `failed`, the server keeps running, and a later request may try again. With `auto_load: false`, such requests receive 503 `model_not_loaded` or `loading` immediately. An explicit load failure enters `failed`/backoff.

**Idle eviction.** Disabled by default (`idle_timeout_s: 0`). When enabled, the timer arms only when the queue is empty and nothing is preparing or in flight. Unload is refused with `busy` (HTTP 409) while any work is queued, preparing or in flight; there is no draining mode.

**Activation.** Load takes the installation-use lease, verifies the whole bundle with `BundleVerifier`, constructs the tokenizer, maps the table and loads every model, then validates the ABI. Unload releases models and the mapping before the lease.

**Compute-plan audit.** `CoreMLRuntime` can inspect anticipated per-operation device assignments with `MLComputePlan` after loading. Because building the plan specializes each chunk a second time, doubling cold-load compile time, the app constructs its runtimes with `auditComputePlan: false` and reports `compute_plan_checked: false`. The CLI (`serve`, `bench`) keeps the audit for text-only bundles; the multimodal runtime does not run it. The audit reports what CoreML plans, not measured residency.

**Compile cache.** The first load of a model on a Mac compiles it for the Neural Engine and stores the result in a per-app cache under `~/Library/Caches/<bundle id>/com.apple.e5rt.e5bundlecache/<OS build>/`. Later loads with a populated cache take seconds. The cache is purgeable: under storage pressure macOS may empty it, and the next load then re-specializes every model, which can take tens of minutes. `E5CacheHealth` estimates warmth by comparing the allocated size of that directory with the size of the models to load (warm at ≥ 80%). The app warns when the cache is cold and defers idle eviction while it is cold (see §11).

**Worker process (app).** The app runs the model in a child process: the app's own executable started with `--worker [--internal-port N]` (or `EMBED_ANE_WORKER=1`). Because it is the same bundle, the worker shares the app's compile cache. The app process (supervisor) owns the configured port through a loopback reverse proxy (`SupervisorProxy`) and forwards requests to the worker's internal port (preferred 15536, otherwise an ephemeral port).

- The worker runs with `idle_timeout_s: 0`; idle eviction is done by the supervisor, which sends `SIGTERM` after the idle period when nothing is in flight. The worker drains and exits, so all of its memory returns to the OS. Before evicting, the supervisor checks `E5CacheHealth`; when the cache is cold it stays loaded for another idle period and logs `eviction_deferred_cold_cache`.
- While the worker is down or starting, `/v1/*` requests receive 503 `overloaded` with `Retry-After: 20` and start a worker in the background. `/health` reports `{status: "ok", worker: "starting"|"down"|"failed", model: {id, state, queue_depth, resident_bytes}, uptime_s}`.
- The supervisor waits for a loading worker without a timeout (an interrupted compile wastes all its work) and logs progress. A worker that hits `EADDRINUSE` at startup retries the bind with backoff for up to 60 s. A load longer than 120 s writes a warning to `worker.log`.
- If a worker exits unexpectedly while the model should stay loaded, the supervisor restarts it.
- In-process serving remains available for tests and debugging with `EMBED_ANE_USE_INPROCESS_SESSION=1` or the user default `use_inprocess_session`.

**Experimental Core AI backend.** `engine_backend: coreai` loads `.aimodel` assets through Apple's Core AI framework (macOS 27 or later; compiled only when the SDK provides it). It is not part of the release bundle and is not covered by the parity gates. The default is `coreml`.

**Timing and statistics.** Each request records monotonic `queue_wait_ns`, `tokenize_ns`, `table_lookup_ns`, `per_chunk_ns[6]` and `total_ns`. `/control/stats` reports nearest-rank p50/p95 over the last 256 requests. `resident_bytes` is process RSS from `mach_task_basic_info`, labeled as process memory, not model memory.

## 8. HTTP API (`EmbedANEHTTP`, Hummingbird)

Binds `127.0.0.1` only (default port 8080). 1 MiB request body limit, enforced cumulatively on streamed bodies. JSON on every route, including errors.

- `POST /v1/embeddings`: `{model, input, encoding_format?, dimensions?, user?}`. `model` is echoed back as given; `/v1/models` lists the configured model id. `user` is accepted and ignored (never logged). Unknown fields and explicit `null` values are schema errors.
- **Standard `input` forms:** string, array of strings, array of token ids, or array of token-id arrays. Items must be nonempty. At most `max_batch` (1–8) items per request; OpenAI's 2048-item limit is not a promise here. Token ids must be in `0..<248078`. The whole batch is validated and tokenized before admission, so an invalid item never lets a valid prefix run.
- `encoding_format`: `float` (default) or `base64` (a string of little-endian IEEE-754 float32 bytes).
- `dimensions`: integer in `1...2048`. Omitted or 2048 returns the model output unchanged. Smaller values return the prefix renormalized to unit length. This is an unvalidated convenience, **not** a trained Matryoshka representation. A zero-norm prefix returns 400.
- **Content parts (nonstandard extension).** OpenAI's embeddings API has no image or video input; embed-ane adds one. `input` may be an array of parts: `{"type":"text","text":"…"}`, `{"type":"image_url","image_url":{"url":"…"}}` or `{"type":"video_url","video_url":{"url":"…"}}`. The array produces **one joint embedding**, not a batch. It needs at least one nonempty text, image or video; at most one image **or** one video; at most 2048 parts. Text parts are concatenated in order without separators, and the image or video is placed before the text regardless of part order. Parts with only text use the text path. Unknown part types or keys return 400.
- **Image sources:** `data:image/<type>;base64,…` or `file:///absolute/path`. HTTP(S) URLs return 400 ("Remote image fetch not supported"); the server never fetches over the network. Local files must be readable regular files of at most 64 MiB with no symlink in any path component (use canonical paths such as `/private/tmp`, not `/tmp`). The 1 MiB body limit includes base64 data, so large images should use file URLs. Decoded images may be at most 16,384 px per side and 64 megapixels, with aspect ratio ≤ 200:1. EXIF orientation is applied; animated images use the first frame.
- **Image sizing:** `vision_resize_mode: smart` (default) keeps the native size, aligned to multiples of 32 px and scaled down only to fit the 1,536-patch budget (or up to the minimum pixel count). This is what the upstream processor does, so images within the budget are embedded exactly as upstream (fp32 cosine 1.0 on a 672×448 test image). `origin-bucket` is a legacy option that always scales to fill the budget; the upscale alone moved the same image to fp32 cosine 0.986. An image yields at most 384 visual tokens.
- **Video sources:** `file:///absolute/path` only (no `data:` URLs), a regular file of at most 2 GiB with no symlinks, decoded with AVFoundation. Sampling follows the upstream processor: 2 frames per second, at least 4 and at most 8 frames (an even count), indices from `linspace(0, frames-1).round()`. Frames are read upright (track transform applied) and grouped into consecutive pairs. Each pair is preceded by its mean timestamp as text, e.g. `<4.1 seconds>`. The resolution comes from `smart_resize` (factor 32) with a pixel budget chosen so that all pairs, timestamps, text and template fit in 512 tokens (at most 384 tokens per pair).
- **Vision unavailable:** with a text-only bundle, image and video input returns **501** `vision_not_configured`. Preprocessing and token-budget errors return 400 `invalid_request`. A bad image or video does not put a healthy model into `failed` or affect queued requests; model failures still do.
- **Success (200):** `{"object":"list","data":[{"object":"embedding","index":i,"embedding":<array or base64 string>}],"model":<echo>,"usage":{"prompt_tokens":N,"total_tokens":N}}`. Usage counts non-padding tokens including the template, `<embedding>` and visual placeholders. Indexes follow input order.
- **Errors (all routes):** `{"error":{"message":String,"type":"invalid_request_error"|"overloaded_error"|"internal_error","param":String|null,"code":String}}`.

  | Status | Codes |
  |---|---|
  | 400 | `invalid_request`, `input_too_long`, `empty_input`, `batch_too_large`, `unsupported_dimension`, `unsupported_encoding_format`, `invalid_spec`, `moving_branch` |
  | 401 | `unauthorized` (with `WWW-Authenticate: Bearer`) |
  | 409 | `busy`, `conflict` |
  | 413 | `body_too_large` |
  | 500 | `abi_mismatch`, `io_error`, `verification_failed`, `internal_error`, … |
  | 501 | `vision_not_configured` |
  | 503 | `model_not_loaded`, `failed`, `loading`, `overloaded` |

  `loading` and `overloaded` carry `Retry-After: 5` from the server (`Retry-After: 20` from the app supervisor while its worker starts) and use type `overloaded_error`. 5xx messages are sanitized.
- `GET /health`: `{status:"ok", model:{id, state, queue_depth, resident_bytes}, uptime_s}`.
- `GET /v1/models`: `{object:"list", data:[{id, object:"model", owned_by:"embed-ane"}]}`.
- **Control routes** (`GET`/`PUT /control/settings`, `POST /control/load`, `POST /control/unload`, `GET /control/stats`) require `Authorization: Bearer <token>`. The token is 256 bits of randomness stored in `~/.embed-ane/control-token` (mode 0600), created on first server start. Requests with a valid token and no `Origin` header are allowed (native clients); an `Origin` other than the exact server origin, including `null`, returns 401. `Host` must be `127.0.0.1` or `localhost`, optionally with the bound port. `PUT /control/settings` returns the effective values plus `restart_required: [fields]`. `idle_timeout_s`, `max_queue_depth` and `auto_load` apply immediately; all other fields need a restart. `/control/unload` returns 409 `busy` while work is queued, preparing or in flight.

## 9. CLI `embed-ane`

Exit codes: 0 success, 1 runtime or transport error, 2 verification failure or failed gate, 3 usage or configuration error.

```
fetch <spec.yaml | org/name> [--offline] [--replace] [--model-root <dir>]
verify <model-id> [--model-root <dir>]
verify-tokenizer <golden.jsonl> [--assets <dir>] [--model-root <dir>] [--model-id <id>]
verify-vision <image-or-video> <text> --position-table <fp32.npy> [--tower <mlmodelc>] [--chunks-dir <dir>]
      [--assets <tokenizer-dir>] [--embedding-table <fp16.npy>] [--resize smart|origin-bucket]
      [--reference <vec.npy>] [--model-root <dir>] [--model-id <id>]
serve [--port <1...65535>] [--model-root <dir>] [--model-id <id>]
status [--port <1...65535>]
bench <corpus.txt> [--reference <ref.jsonl>] [--output-dir <dir>] [--model-root <dir>] [--model-id <id>]
```

`fetch` with a path to an existing file installs from that spec; with `org/name` it reads `spec.yaml` from that Hugging Face repository first. `status` reads the port from `~/.embed-ane/config.yaml` unless `--port` is given. `bench` writes `bench/<timestamp>-<uuid>.json` with raw samples, p50/p95, device, OS, toolchain and artifact digests. See [docs/runtime-cli.md](docs/runtime-cli.md).

## 10. Configuration

`~/.embed-ane/config.yaml` (strict YAML, `config_version: 1`, written atomically via a temporary file and rename):

| Key | Default | Notes |
|---|---|---|
| `port` | `8080` | 1–65535. A port in use is an error; there is no fallback port. |
| `model_id` | `wemm-embedding-2b-ane` | Installed bundle directory name. |
| `model_root` | `~/.embed-ane/models` | Absolute or `~/`. Environment override `EMBED_ANE_MODEL_ROOT`. |
| `idle_timeout_s` | `0` (keep loaded) | 0 … 1,000,000,000. |
| `auto_load` | `true` | Load on the first request while unloaded. |
| `max_queue_depth` | `16` | Positive. |
| `max_batch` | `8` | 1–8. |
| `compute_units` | `cpu_and_ne` | Schema also accepts `cpu_only`, `cpu_and_gpu`, `all`; the CoreML runtime rejects them. |
| `engine_backend` | `coreml` | `coreai` is experimental (§7). |
| `vision_tower_path`, `vision_extrope_chunks_dir`, `vision_position_table_path` | detected | Overrides for vision assets; all three or none. |
| `vision_resize_mode` | `smart` | `smart` or `origin-bucket` (§8). |

Precedence: CLI flag > environment > file > default. Unknown keys, duplicate keys and explicit `null` values are rejected; a malformed file is an error, not a silent fallback to defaults.

Model root layout:

```
<root>/<id>/
  spec.yaml  manifest.yaml
  tokenizer.json  tokenizer_config.json  embed_table.fp16.npy
  vision_abs_pos.fp32.npy            # multimodal bundles
  vision_tower.mlmodelc/             # multimodal bundles
  chunks/chunk{0..5}.mlmodelc/       # position-input decoder (multimodal) or plain decoder (text-only)
```

**Vision assets.** When the three vision keys are absent, the configuration detects `vision_tower.mlmodelc` and `vision_abs_pos.fp32.npy` in the model directory and fills them in. Explicit absolute or `~/` paths override detection, for example to test locally converted assets; the chunks directory must contain `chunk{0..5}.mlmodelc`. Explicitly configured paths are checked for readability and absence of symlinks but are not digest-verified installations; leave them unchanged while serving. All vision keys and `vision_resize_mode` require a restart.

## 11. App (`App/`, Xcode target using the local package)

**Menu bar.** A `MenuBarExtra` whose label is a chip glyph drawn in code (`StatusGlyph`) with six states from `StatusPresentation`: Ready (solid die), Standby (unloaded, auto-load on; outlined die), Unloaded (auto-load off; no die), Working (starting, stopping, restarting, loading or unloading; the die pulses only after 0.5 s, static with Reduce Motion), Error (exclamation mark) and Stopped (dashed package).

**Menu.** Status title and detail, model, in-flight work, a cold compile cache warning when applicable, and the OpenAI base URL with **Copy Endpoint URL**; one primary action (Unload Model / Load Model / Retry Load / Restart Server / Start Server; Unload is disabled while work is active and asks for confirmation when the compile cache is cold); a **Model** submenu of verified models plus Manage Models…; **Keep Model Loaded** and **Launch at Login** toggles; Open Embed ANE… (⌘O), Settings… (⌘,) and Quit Embed ANE (⌘Q, confirmed).

**Main window.** One window with a sidebar: **Overview** (status, primary action, endpoint, queue / in-flight / preparing, p50/p95, process memory, lifecycle summary, compile cache state with Check, model folder), **Models** (verified models, switching via restart, installs that need attention, install from Hugging Face or a YAML file with progress, cancel and conflict replacement), **Playground** (embed a sentence; copy curl and Python examples) and **Logs** (recent events).

**Settings.** Tabs General (launch at login, port, model folder), Lifecycle (keep model loaded, unload-after preset, auto load) and Advanced (compute units, queue and batch limits, config.yaml and log file) with one Save / Reset footer. Compute units are shown read-only as Neural Engine; other values from a hand-edited file show a warning and **Reset to Neural Engine**. Fields that need a restart are tagged; after saving them a banner offers **Restart Now**, which restarts the server in place (`MenuBarModel.restartServer`, waits up to 30 s for in-flight work and never interrupts requests).

All windows share one `AppEnvironment` and `MenuBarModel`; closing windows does not stop serving. Login items use `SMAppService` and change only on an explicit toggle. Quitting asks for confirmation. The app icon is the Icon Composer document `App/AppIcon.icon` (Liquid Glass on macOS 26 and later; Xcode generates fallback images for macOS 15). CI compiles the app unsigned with `bash scripts/build-app.sh`. See [docs/app.md](docs/app.md).

## 12. Testing

**Unit tests** (`swift test`, run in CI on macOS without model files):

1. YAML schema strictness (unknown keys, moving branches, bad ids, vision pairing).
2. Digest verification on temporary fixtures (files, nested inner manifests, corrupted bytes, symlinks, `..`, duplicates, unexpected files).
3. NPY parsing (valid, truncated, v2, Fortran order, wrong dtype or shape, out-of-range ids).
4. The downloader against a real `127.0.0.1:0` listener: happy path, resume after a killed and relaunched process, Range ignored (200) restart, 416 restart, corrupted download (nothing promoted), atomic promotion, offline zero connections, concurrent-fetch lock, non-destructive conflict.
5. The lifecycle with a mock predictor and a controllable clock: single-flight load (10 concurrent → 1 load), overflow, cancelled waiter, eviction only at quiescence, failure and backoff, auto-load.
6. The HTTP contract with a mock predictor over a real socket: every route and error code, golden OpenAI response JSON, batch order, `Retry-After`, Bearer/Origin/Host matrix, 413 body limit, content parts, `dimensions` and `base64`.
7. Prompt template, video sampling and M-RoPE positions; tokenizer harness self-test; CLI parsing; app support (presentation, settings draft, supervisor and worker over loopback, compile cache probe).

**On-device checks** (run by hand with an installed model; set `EMBED_ANE_MODEL_ROOT` or pass `--model-root`): tokenizer differential on ≥1,000 real cases, `bench --reference` parity (mean cosine ≥ 0.995) and latency, `verify-vision --reference` for image and video (cosine ≥ 0.99), cold, warm and same-process reload timing, idle, loaded and post-unload RSS, and the compute-plan report.

## 13. Engineering rules

1. Third-party code is not copied into this repository; dependencies are used through their public SPM APIs.
2. macOS 15+ on arm64 only; Swift 6 strict concurrency; no `try!` or force unwraps in production paths.
3. Explicit-trigger downloads, fail-closed digests, loopback-only bind, control token file mode 0600.
4. No model binaries in the repository; everything resolves through the configurable model root.
5. Conventional commits, one concern per commit, CI green before merge.
6. No performance claims beyond `bench` recordings.
