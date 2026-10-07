# HTTP API, lifecycle, control security and configuration

Contract: [`SPEC.md`](../SPEC.md) §§7, 8 and 10.

`EmbedANEHTTP` is the local server used by both `embed-ane serve` and the menu bar app. It exposes an OpenAI-compatible embeddings endpoint plus authenticated control routes, and always listens on `127.0.0.1`.

## Quick examples

```sh
# Text (one or more strings)
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"wemm-embedding-2b-ane","input":["first text","second text"]}'

# One image plus optional text -> one joint embedding
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"wemm-embedding-2b-ane","input":[
        {"type":"image_url","image_url":{"url":"file:///private/tmp/cat.jpg"}},
        {"type":"text","text":"a cat on a sofa"}]}'

# One video (local file only) plus optional text
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"wemm-embedding-2b-ane","input":[
        {"type":"video_url","video_url":{"url":"file:///private/tmp/clip.mov"}}]}'
```

With the OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
vectors = client.embeddings.create(model="wemm-embedding-2b-ane", input=["hello"]).data
```

## Composition

`EmbeddingHTTPServer` takes an `EmbeddingPredictor`, an `EmbeddingPreparer`, a validated `ServiceConfiguration` and a `ConfigurationStore`. It never picks a mock or a real runtime by itself; the composition root injects them (normally from `ServingRuntimeFactory.make`).

```swift
import EmbedANECore
import EmbedANEHTTP

let store = ConfigurationStore()
let configuration = try store.resolve(
    cli: ConfigurationOverrides(port: requestedPort, modelRoot: requestedRoot)
)
let runtime = try ServingRuntimeFactory.make(configuration: configuration,
                                             acquireLease: makeLease, auditComputePlan: false)
let server = try EmbeddingHTTPServer(
    configuration: configuration,
    predictor: runtime.predictor,
    preparer: runtime.preparer,
    store: store
)
try await server.run()
```

The server starts with the model unloaded. The composition root may call `server.lifecycle.load()` (CLI `serve` does, before listening), clients may call `POST /control/load`, or, with `auto_load` on, the first embeddings request loads it. A request never triggers a model download. Callers own task cancellation, signal handling and the final unload.

The bind host is fixed at `127.0.0.1`; there is no hostname option. `portOverride: 0` asks the kernel for an ephemeral port (tests). A bind failure keeps the attempted address and underlying error; there is no retry on another port.

## `POST /v1/embeddings`

Validation runs in this order, and nothing is admitted until all of it passes:

1. JSON schema: required string `model`, an `input` in one of the accepted forms, and optional fields of the declared types. Unknown keys and explicit `null` values are rejected.
2. `encoding_format`, when present, is `float` or `base64`.
3. `dimensions`, when present, is in `1...2048`.
4. A nonempty batch of at most `min(max_batch, 8)` items, each nonempty. Whitespace-only strings are not treated as empty.
5. All texts are tokenized before admission, so an error in item 1 cannot let item 0 run. Token-array items are checked against the vocabulary (`0..<248078`) and the 507-content-token limit and wrapped in the same prompt template as text.

### Input forms

| `input` | Result |
|---|---|
| `"text"` or `["a", "b", …]` | One embedding per string. |
| `[1, 2, 3]` or `[[1, 2], [3, 4]]` | One embedding per token array (content ids, without the template). |
| `[{"type": …}, …]` content parts | **One joint embedding** for the whole array. |

Content parts are an embed-ane extension; OpenAI's embeddings API has no image or video input. Part types:

- `{"type":"text","text":"…"}`. Text parts are concatenated in order without separators.
- `{"type":"image_url","image_url":{"url":"…"}}` with `data:image/<type>;base64,…` or `file:///absolute/path`.
- `{"type":"video_url","video_url":{"url":"file:///absolute/path"}}`.

At most one image or one video per request, at most 2048 parts, and at least one nonempty text, image or video. The image or video is always placed before the text in the prompt (SPEC §4). Remote `http(s)` URLs are rejected with 400; the server never fetches over the network. Local files must be regular files with no symlink in any path component (use `/private/tmp`, not `/tmp`): images at most 64 MiB, videos at most 2 GiB. File checks run on a dedicated I/O queue. The 1 MiB body limit includes base64 data, so large images should be passed as file URLs.

Images are decoded with ImageIO (EXIF orientation applied, first frame of animated images), at most 16,384 px per side and 64 megapixels, aspect ratio ≤ 200:1. `vision_resize_mode` chooses the resize policy: `smart` (default) keeps the native size within the allowed pixel range, as the upstream processor does; `origin-bucket` (legacy) always scales to fill the 1,536-patch budget, which upscales small images and moves their embeddings away from upstream. An image uses at most 384 visual tokens.

Videos are decoded with AVFoundation: 2 frames per second, 4 to 8 frames (even), evenly spaced indices, upright orientation, grouped into frame pairs, each preceded by a timestamp such as `<4.1 seconds>`. The frame size is chosen with the upstream `smart_resize` rule so that every pair, the timestamps, the text and the template fit in 512 tokens.

When the loaded bundle is text-only, image and video parts return **501** `vision_not_configured`. Preprocessing and token-budget problems return 400 `invalid_request`. A bad image or video never moves a healthy model into `failed` and never rejects queued requests.

### Response

```json
{"object":"list",
 "data":[{"object":"embedding","index":0,"embedding":[0.0123, …]}],
 "model":"wemm-embedding-2b-ane",
 "usage":{"prompt_tokens":9,"total_tokens":9}}
```

- `embedding` is 2,048 floats, or a base64 string of little-endian float32 bytes with `encoding_format: "base64"`.
- Vectors are L2-normalized by the model graph and returned unchanged. With `dimensions < 2048` the prefix is renormalized; this is not a trained Matryoshka representation, and a zero-norm prefix returns 400.
- `usage` counts non-padding tokens including the template, `<embedding>` and visual placeholders, computed from the prepared inputs (never from the runtime's own report). The maximum is 8 × 512 = 4,096.
- `model` echoes the request; `/v1/models` reports the configured model id.

## Errors

Every error has exactly `error.message`, `error.type`, `error.param` (JSON `null` when absent) and `error.code`.

| Status | Codes | Type / headers |
| --- | --- | --- |
| 400 | `invalid_request`, `input_too_long`, `empty_input`, `batch_too_large`, `unsupported_dimension`, `unsupported_encoding_format` | `invalid_request_error` |
| 401 | `unauthorized` | `invalid_request_error`; `WWW-Authenticate: Bearer` |
| 409 | `busy`, `conflict` | `invalid_request_error` |
| 413 | `body_too_large` | `invalid_request_error` |
| 500 | `abi_mismatch`, `io_error`, `verification_failed`, other internal errors | `internal_error` |
| 501 | `vision_not_configured` | `internal_error` |
| 503 | `model_not_loaded`, `failed` | `internal_error` |
| 503 | `loading`, `overloaded` | `overloaded_error`; `Retry-After: 5` |

Messages of 5xx responses are sanitized: no paths or adapter details. The 1,048,576-byte body limit applies to all routes and to chunked bodies, not only to `Content-Length`; an oversized declared length is rejected without reading the body. Authentication is checked first on protected routes. 401 and 413 responses close the connection.

Invalid HTTP framing that the parser rejects before a request exists is outside this JSON boundary. No CORS `Access-Control-Allow-Origin` header is ever sent.

## Other routes

- `GET /health` → `{"status":"ok","model":{"id","state","queue_depth","resident_bytes"},"uptime_s"}`. `state` is one of `unloaded`, `loading`, `ready`, `unloading`, `failed`. Through the app's supervisor the body also has `"worker"`.
- `GET /v1/models` → `{"object":"list","data":[{"id","object":"model","owned_by":"embed-ane"}]}`.
- `GET /control/settings`, `PUT /control/settings` → effective configuration fields plus `restart_required`.
- `POST /control/load`, `POST /control/unload` (body absent or `{}`) → `{"state","report","resident_bytes_scope":"process"}`.
- `GET /control/stats` → the structure encoded by `RuntimeStatistics`: lifecycle snapshot, counters, p50/p95 total and per-phase percentiles over the last 256 requests, and the last error code.

## Lifecycle and the serial worker

`LifecycleActor` runs one shared load, one shared unload, a bounded FIFO of waiting requests and one prediction at a time. The default queue limit is 16 waiting requests, independent of the one in flight. Lowering the limit never drops accepted requests; new ones are rejected until there is room.

- Cancellation is checked before load, unload and admission. An accepted request is not cancelled just because its HTTP client disconnects, and cancelling one load waiter never cancels the shared load.
- A prediction failure completes every accepted waiter, enters `failed`, and leaves no continuation hanging. Explicit load retries back off 1, 2, 4, 8, 16, 32, then 60 s, reset after a successful load.
- **Auto-load.** With `auto_load: true`, a request arriving while unloaded starts the load and waits in the queue. If that load fails, the state returns to `unloaded` (not `failed`), queued requests get 503 `failed`, and a later request may try again. With `auto_load: false`, the request gets `model_not_loaded` (or `loading` during a load).
- Token preparation is counted separately from the queue. Unload and idle eviction are refused while anything is preparing, queued or in flight; new preparation is refused while an unload is running. Idle eviction is off by default; its timer arms only when the runtime is idle, is cancelled by activity or settings changes, and ignores stale wake-ups.

`SerialEmbeddingWorker` constructs, calls and releases its non-`Sendable` backend on one dedicated `DispatchQueue`, so blocking CoreML calls never run on the cooperative executor or an HTTP event loop. Tokenization runs on the same worker; prediction consumes the prepared token arrays without tokenizing again.

Statistics keep the last 256 completed admitted requests, failures included. Nearest-rank percentiles use rank `ceil(p·N)`; empty windows report zero. Total time starts before tokenization for text requests; `queue_wait_ns` starts at admission; `per_chunk_ns` sums each chunk across a request's serial B=1 cascades. Phases are not guaranteed to add up to the total.

`resident_bytes` is process RSS (`mach_task_basic_info`), not a model-size estimate. A failed read returns 0 as an "unavailable" value, which does not mean the model is unloaded.

## Control security

On first start, `ControlToken.loadOrCreate` writes 32 bytes from `SecRandomCopyBytes`, as 64 lowercase hex characters plus LF, to `~/.embed-ane/control-token`. The 0600 file is published atomically (create-only hard link from a synced temporary file), so concurrent starters read the same token and no process sees a partial file or a silently rotated token.

An existing token file must be a regular, non-symlink file owned by the current user with mode exactly 0600 and the expected encoding. Anything else fails closed; the file is never overwritten or chmod-repaired. The state directory is created with mode 0700.

Token descriptions are redacted. Bearer comparison runs over a fixed 64-byte candidate without content-dependent early exit. Duplicate `Authorization` or `Origin` headers are rejected; the `Bearer` scheme is case-insensitive.

`Host` is checked on every route: `127.0.0.1` or `localhost`, optionally with the bound port. Wrong ports, other names, look-alike suffixes, trailing dots, userinfo and IPv6 literals are rejected, and a missing `Host` is not accepted. For `/control/*`:

| Bearer | Origin | Outcome |
| --- | --- | --- |
| Missing, wrong or duplicated | Any | 401 |
| Valid | Absent | Allowed |
| Valid | Exactly `http://<request-host>:<bound-port>` | Allowed |
| Valid | Foreign, `null`, empty, duplicated, wrong port, scheme or host | 401 |

`localhost` and `127.0.0.1` are both accepted hosts but distinct browser origins. Origins with paths, queries, fragments or credentials are rejected. Unknown `/control/*` routes and `OPTIONS` cannot bypass these checks.

```sh
TOKEN=$(cat ~/.embed-ane/control-token)
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8080/control/stats
curl -s -X PUT -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"idle_timeout_s":900}' http://127.0.0.1:8080/control/settings
```

## Configuration and settings

`ConfigurationStore` reads strict version-1 YAML from `~/.embed-ane/config.yaml` (keys and defaults in SPEC §10). A missing file means defaults; a malformed file is an error. Writes use a private temporary file, fsync and an atomic rename in the same directory. Duplicate and unknown keys are rejected, and final-component symlinks are refused for reads and writes. Home-relative paths are expanded. Precedence is CLI flag > `EMBED_ANE_MODEL_ROOT` > file > default.

The server writes the configuration file at startup only if it does not exist.

`PUT /control/settings` merges a partial, strictly typed object into the saved configuration. The file is written first; only then are live changes applied. `idle_timeout_s`, `max_queue_depth` and `auto_load` apply immediately. `port`, `model_id`, `model_root`, `max_batch`, `compute_units`, `engine_backend`, the vision paths and `vision_resize_mode` take effect after a restart: responses keep reporting the old effective values and list those fields in a sorted `restart_required`. Later PUTs keep pending changes; reverting a pending field removes it from the list. Concurrent PUTs cannot interleave (a second one gets `conflict`).

`compute_units` accepts `cpu_and_ne`, `cpu_only`, `cpu_and_gpu` and `all` in the file, but the CoreML runtime only runs `cpu_and_ne` and refuses to start with any other value.

## Tests

HTTP tests run the real Hummingbird application on `127.0.0.1:0` with blocking Darwin TCP clients on separate queues; they do not use an in-memory router or URLProtocol interception. Startup waits for the bound-port callback, socket operations have timeouts, and cleanup removes only the test's temporary state directory. The committed golden response holds two full 2,048-dimensional vectors and detects unwanted normalization.

```sh
swift build
swift test --filter EmbedANEHTTPTests
```
