# Model downloader (`EmbedANEDownload`)

Contract: [`SPEC.md`](../SPEC.md) §§5–6.

`EmbedANEDownload` installs model bundles described by a spec YAML file. Downloads only start on an explicit request (`embed-ane fetch`, or Add Model in the app). Every file is pinned by SHA-256 and size, and a bundle becomes visible only after all of it has been verified.

## Public API

```swift
import Foundation
import EmbedANEDownload

let downloader = ModelDownloader() // HTTPS only; no network access at construction.
let report = try await downloader.fetch(
    specPath: URL(fileURLWithPath: "/path/to/spec.yaml"),
    modelRoot: URL(fileURLWithPath: "/path/to/models"),
    offline: false,
    replace: false,
    progress: { event in
        // Called on worker/callback queues; hop to the main actor for UI work.
        // event.phase / event.path / event.receivedBytes / event.totalBytes
    }
)
let verified = try await downloader.verify(
    modelID: report.modelID,
    modelRoot: URL(fileURLWithPath: "/path/to/models")
)
```

`FetchReport.disposition` distinguishes installed, replaced, already-installed and verified-offline outcomes. Errors are `EmbedANEError` values that name the artifact path for transport and digest failures. Credentials and server error bodies never appear in messages.

`fetchSpec(repo:endpoint:revision:)` reads `spec.yaml` (at most 1 MiB) from a Hugging Face repository given as `org/name`. A remote spec must name the same repository and endpoint it was fetched from and may not declare `auth`; only local spec files may name a credential variable. `embed-ane fetch org/name` and the app's Hugging Face download use it, write the spec to a private temporary file, and then call `fetch`.

`ModelUseLease(modelRoot:modelID:)` is a nonblocking shared lease on an installation. A serving composition takes it before verification and load and holds it until the runtime has unloaded; Core does not import Download, so the lease is passed in as a factory. Replacement takes the exclusive counterpart before transfer and holds it through the commit. These are advisory locks between cooperating processes; other programs that ignore them are outside this guarantee.

### Credentials

When a local spec declares `auth: env:HF_TOKEN`, the token is read from that environment variable. If `HF_TOKEN` is unset, `~/.embed-ane/hf-token` is used instead; it must be a private (0600) file owned by the current user. This matters for the app: apps launched from Finder do not inherit a terminal's environment.

```sh
printf '%s' "hf_..." > ~/.embed-ane/hf-token && chmod 600 ~/.embed-ane/hf-token
```

When fetching a remote spec, the token is only sent to `huggingface.co` hosts.

## Bundle format

See SPEC §5. Every `.mlmodelc/` directory is listed in the spec through its inner `manifest.txt` (`<path> <size> <sha256>` per line, sorted, LF). During fetch and verification every `*/manifest.txt` entry is expanded: the manifest is verified first, then each member, and any unlisted file in the directory blocks promotion. A multimodal bundle adds `vision_tower.mlmodelc/manifest.txt` and `vision_abs_pos.fp32.npy`, which must appear together.

## Resolution and staging

The revision endpoint is queried first. A 40-hex SHA must resolve to itself. A name additionally requires the refs endpoint: a branch is rejected, a tag must be unique, and its target commit must agree with the revision response. No artifact URL ever contains a moving tag or branch. An ambiguous name, or a tag that moves between the two responses, fails closed.

The writer lock is `<root>/.locks/<id>.fetch`. Neither it nor the `<id>.use` lease file is ever unlinked. A second fetch of the same id receives an explicit `conflict` instead of waiting in an unbounded queue.

Staging lives in `<root>/.staging/<id>`. `RESOLVED` and `SPEC_SHA256` bind the saved bytes to one commit and to the exact spec bytes. A missing or malformed binding, a changed spec, or commit drift discards stale staging before any further transfer. Valid complete files are verified again and kept; interrupted partial files are kept. A digest failure deletes only that artifact's partial file and resume metadata.

Partial bodies and their validators live under `.embed-ane-download/`, keyed by the SHA-256 of the relative artifact path, so they never appear inside an `.mlmodelc` directory's inventory. That directory is removed before promotion; `RESOLVED` and `SPEC_SHA256` remain as small crash-recovery receipts.

## Streaming and resume

`DownloadTransport` is the injectable streaming boundary. The production implementation uses URLSession data delegates. It validates the response before accepting data, serializes callbacks, closes sessions after completion and handles cancellation races. It never holds a whole artifact in memory, and disk writes run on dispatch queues, not on the cooperative executor.

| Response | Action |
|---|---|
| 206 | Check start, end, total, length and the saved validator before appending. |
| 200 after Range | Truncate and restart before accepting the full body. |
| 416 after Range | Reset and retry with an unconditional GET; a 416 on that GET fails. |
| Changed or missing validator | Reset and issue a new full request; never mix representations. |
| Malformed Content-Range | Fail without altering the existing valid prefix. |
| Truncated or failed transfer | Close and keep the prefix for a later fetch. |
| Complete partial file after a crash | Verify it locally and promote it without downloading again. |
| Digest or size mismatch | Delete only the affected partial file and refuse installation. |

Strong ETags are used for `If-Range`, with Last-Modified as a fallback. Final SHA-256 verification is mandatory even without a validator. Compression may not change the meaning of byte ranges, so a response with a non-identity `Content-Encoding` is rejected. Repeated restarts are bounded, and each partial response must advance the file.

Production URLs and redirect targets must be HTTPS. Only the explicit `.loopbackHTTPForTests` policy allows `http://127.0.0.1:<port>`. Cookies, implicit session credentials and caching are disabled. The Bearer header is removed on cross-origin redirects, including a port change.

## Verification and atomic installation

Core's `ModelSpec`, `ArtifactFile`, `InnerManifest`, `InstallManifest`, `FileDigest` and `BundleVerifier` are the verification authorities. Unexpected members, symlinks, path traversal and case- or normalization-colliding paths prevent promotion. Directory walks, locks, stale-stage cleanup and promotion operate relative to open directory descriptors; cleanup unlinks symlinks rather than following them.

`manifest.yaml` is written and the complete staged bundle verified before the commit point. A first install uses an exclusive atomic rename, so a destination that appears concurrently is never overwritten. `replace: true` swaps the whole verified bundle with an atomic directory exchange and removes the old one afterwards. A failed download leaves the previous install intact, and a crash during the exchange leaves either the old or the new complete bundle in place.

The successful rename is the commit point. A later failure to fsync the parent directory or remove the retired bundle is returned as a **warning on a committed install**, not as a failure.

Offline mode skips credentials, revision resolution, transport and staging. `fetch(offline: true)` verifies the installed bundle against the exact spec digest; `verify` verifies an installed bundle by id. Offline replacement is rejected.

## Tests

```sh
swift build --target EmbedANEDownload
swift test --filter DownloaderSocketTests
swift test
```

Fixtures use a real Darwin TCP listener on `127.0.0.1:0`, small generated files and real URLSession transfers, with finite timeouts. They do not use URLProtocol mocks, fixed ports, Python servers, CoreML models or the Hugging Face Hub. The kill-and-relaunch test starts the test binary as two real child processes (located with `xcrun --find xctest`) so the killed process is the downloader itself.

Coverage includes: happy path and nested manifests; idempotence; commit and slash-containing tag resolution; branch rejection; tag drift during resolution; interrupted 206 resume; 200, 416 and ETag restarts; invalid ranges; corrupted bytes; spec and commit drift in staging; atomic visibility; offline zero connections; concurrent writer conflict; cancellation and lease release; replacement of inactive and active installs and non-destructive failure; unexpected or unsafe members; a symlink attack on a partial file; rejection of production HTTP and missing auth; Bearer and redirect restrictions; safe error messages; process-death resume; and remote-spec origin checks.
