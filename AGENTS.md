# Contributing to embed-ane

A short guide for human contributors and coding agents. [`SPEC.md`](SPEC.md) is the technical contract: if your change alters behavior described there, update it in the same pull request. Topic guides live in [`docs/`](docs/).

## Build and test

Requirements: Apple silicon, macOS 15+, Xcode 26+ (Swift tools 6.1, Swift 6 language mode).

```sh
swift build                  # package: Core, HTTP, Download, AppSupport, CLI
swift test                   # unit tests; no model files needed
bash scripts/build-app.sh    # unsigned compile of App/App.xcodeproj (no launch, no model)
swift run embed-ane --help
```

CI runs exactly these three steps. Keep them green.

## Module boundaries

| Module | Path | Owns | May depend on |
|---|---|---|---|
| `EmbedANECore` | `Sources/EmbedANECore` | tokenizer and prompt template, embedding table, CoreML cascade and vision tower, lifecycle, configuration, digest verification | Tokenizers, Yams; no UI, no networking |
| `EmbedANEHTTP` | `Sources/EmbedANEHTTP` | Hummingbird server, OpenAI-compatible routes, control security, settings | Core, Hummingbird |
| `EmbedANEDownload` | `Sources/EmbedANEDownload` | spec-driven Hugging Face downloader, staging, atomic install, leases | Core, Yams |
| `EmbedANEAppSupport` | `App/Support` | testable app logic: `MenuBarModel`, sessions, supervisor proxy, worker runner, status presentation | Core, HTTP, Download |
| `embed-ane` | `Sources/embed-ane` | CLI composition | Core, HTTP, Download |
| App target | `App/Sources` | SwiftUI/AppKit views (Xcode only) | AppSupport |

Core depends on nothing else in this package. Blocking CoreML work runs only on the serial worker, never on the cooperative executor or an HTTP event loop.

## Rules

1. **No model binaries in the repository.** Chunks, the vision tower, tables and tokenizer files are downloaded into a configurable model root (`~/.embed-ane/models` by default, `EMBED_ANE_MODEL_ROOT` or `--model-root` to override). Tests use small generated fixtures.
2. **No copied third-party code.** Dependencies are used through their public SPM APIs.
3. **Pinned dependencies** (SPEC §2): Hummingbird 2.26.0, swift-transformers 1.3.4, Yams 6.2.2. Open an issue before bumping one; commit `Package.resolved`.
4. **macOS on Apple silicon only.** Swift 6 strict concurrency; no `try!` or force unwraps in production paths.
5. **Security defaults stay on:** loopback-only bind, explicit-trigger downloads, fail-closed digest checks, control token file mode 0600, no network fetch of request media.
6. **Model ABI is fixed** (SPEC §3–4). Changes to the prompt, tokenization, tensor path or preprocessing need on-device parity numbers in the pull request.
7. **No performance claims** beyond what `embed-ane bench` records.

## On-device checks

`swift test` never loads a real model. Changes to the predict path, the prompt or preprocessing should also be checked on a Mac with an installed bundle:

```sh
export EMBED_ANE_MODEL_ROOT=~/.embed-ane/models
swift run embed-ane fetch maolon/WeMM-Embedding-2B-CoreML-ANE
swift run embed-ane verify-tokenizer ./golden.jsonl            # >= 1,000 cases, zero mismatches
swift run embed-ane bench ./corpus.txt --reference ./ref.jsonl  # mean cosine >= 0.995
swift run embed-ane verify-vision ./photo.jpg "text" \
  --position-table "$EMBED_ANE_MODEL_ROOT/wemm-embedding-2b-ane/vision_abs_pos.fp32.npy" \
  --resize origin-bucket --reference ./ref.npy                  # cosine >= 0.99
```

Report the numbers in the pull request.

## Commits

Conventional commits (`feat:`, `fix:`, `perf:`, `docs:`, `test:`, …), one concern per commit; `perf:` and `bench:` commits include before/after numbers. CI must be green before merging to `main`.
