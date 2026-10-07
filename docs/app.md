# Menu bar app

Contract: [`SPEC.md`](../SPEC.md) §§7 and 11, with the Core, HTTP and Download contracts it builds on.

**Embed ANE.app** is a macOS 15+ arm64 accessory app (`LSUIElement`, no Dock icon) that serves the same HTTP API as `embed-ane serve`. Sources are split between `App/Sources` (SwiftUI and AppKit views, compiled only by the Xcode target) and `App/Support` (the `EmbedANEAppSupport` package target with the testable logic).

## Composition and startup

`App/Sources/EmbedANEApp.swift` declares a `MenuBarExtra` with a native menu, one main window (`Window("Embed ANE", id: "main")`) with a sidebar, and a Settings scene. `AppDelegate` creates one process-wide `AppEnvironment`, which owns the `@MainActor @Observable` `MenuBarModel`. Closing the menu, the main window or Settings does not stop serving or polling. The main window opens only on request, not at launch or through window restoration.

Startup resolves configuration through `ConfigurationStore`, builds the session without loading the model, starts the loopback listener, and verifies installed models off the main thread. The configured model loads once the listener, the first snapshot and a successful model catalogue are all available. A missing model leaves the server unloaded and the install UI usable. A port collision is reported as an error; the app never increments the port, attaches to another server or starts a download.

### Worker process

By default the model runs in a separate worker process (`WorkerSession`):

- **Supervisor.** The app process handles the UI and owns the configured port through a small Hummingbird reverse proxy (`SupervisorProxy`). It applies the same Host and control-token checks as the server. `/v1/*`, `/health`, `GET /control/settings` and `/control/stats` are forwarded to the worker when it is ready and answered by the supervisor otherwise; `POST /control/load` and `/control/unload` start or stop the worker; `PUT /control/settings` is applied by the supervisor.
- **Worker.** The same executable launched as `Embed ANE --worker --internal-port <N>` (with `EMBED_ANE_WORKER=1`) on a loopback port (15536 if free, otherwise ephemeral). Because it is the same app bundle, it shares the app's CoreML compile cache. It logs to `~/.embed-ane/logs/worker.log` and always runs with `idle_timeout_s: 0`.
- **Idle eviction.** When `idle_timeout_s` elapses with no request in flight, the supervisor sends `SIGTERM`; the worker drains and exits, returning all of its memory to the OS. If the compile cache is cold (below), eviction is deferred for another idle period and `eviction_deferred_cold_cache` is logged.
- **On-demand load.** While no worker is ready, `/v1/*` requests receive 503 `overloaded` with `Retry-After: 20` and a worker starts in the background. `/health` then returns `{"status":"ok","worker":"starting"|"down"|"failed","model":{…},"uptime_s":…}`.
- **Patience.** The supervisor waits for the worker's listener for up to 120 s, then waits without a limit while it reports `loading` (interrupting a compile wastes it), logging progress. A worker whose port is still held by a previous process retries the bind every 2 s for up to 60 s. A load over 120 s writes `slow load: possible E5 respecialization; do not kill mid-compile` to `worker.log`.
- **Crash recovery.** If the worker exits unexpectedly while the model should stay loaded, the supervisor shows a notice and starts a new one.
- **In-process mode.** For debugging and tests, `EMBED_ANE_USE_INPROCESS_SESSION=1` or the user default `use_inprocess_session` runs `InProcessAppSession` instead: one `EmbeddingHTTPServer` in the app process, whose `LifecycleActor` and `SettingsController` back both the UI and the HTTP routes.

`embed-ane status --port <port>` works against the supervisor port.

Both session types build their runtime with `ServingRuntimeFactory.make(…, auditComputePlan: false)`, acquiring `ModelUseLease` before load-time verification. With a multimodal bundle (or explicitly configured vision paths) image and video requests are served; otherwise they return `vision_not_configured`.

### Compile cache

The first load of a model compiles it for the Neural Engine; later loads reuse a per-app cache (`~/Library/Caches/<bundle id>/com.apple.e5rt.e5bundlecache/<OS build>/`). macOS may empty this cache under storage pressure, and the next load then takes tens of minutes. `E5CacheHealth` compares the cache's allocated size with the size of the models to load (warm at ≥ 80%). The app probes it after the listener starts, after each load, and when you click **Check** on Overview. When it is cold, the menu and Overview say so, and Unload and Restart ask for confirmation, warning that the next load may take 40 minutes or more. Freeing disk space helps keep the cache.

### Polling

Every 500 ms while the app runs (even with the menu and windows closed), the model reads lifecycle state, queue, in-flight and preparing counts, process RSS, and p50/p95 over the last 256 requests. Missing samples and unavailable RSS show a dash. Displayed numbers are live observations, not performance claims.

## Status and menu

`StatusPresentation` merges server phase and model lifecycle into six states, each with one title, an optional detail and one primary action:

| State | When | Glyph | Action |
|---|---|---|---|
| Ready | Model loaded | solid die | Unload Model |
| Standby | Unloaded, auto load on ("Loads on next request (~15–30 s)") | outlined die | Load Model |
| Unloaded | Unloaded, auto load off | no die | Load Model |
| Working | Starting, stopping, restarting, loading, unloading | die pulses after 0.5 s (static half-filled die with Reduce Motion) | — |
| Error | Server or model failure | exclamation mark | Restart Server / Retry Load |
| Stopped | Server stopped | dashed package | Start Server |

The glyph (`StatusGlyph`) is a template image drawn in code: a rounded chip package with a pin-1 mark around a die. Working animates only after it has lasted 0.5 s (`WorkingAnimation`), so quick transitions do not flicker.

The menu shows the status title (opens Overview), a detail line, the model, in-flight work, a cold-cache warning when applicable, and the OpenAI base URL with **Copy Endpoint URL**; then the primary action; a **Model** submenu listing verified models plus **Manage Models…**; **Keep Model Loaded** and **Launch at Login** toggles; **Open Embed ANE…** (⌘O), **Settings…** (⌘,) and **Quit Embed ANE** (⌘Q). Only standard menu items are used.

## Main window

The sidebar has four panes. A notice area and the restart-required banner appear above every pane. Reopening the window reuses it.

- **Overview**: status and primary action, base URL with copy, queue / in-flight / preparing / p50 / p95 / memory, a lifecycle summary linking to Settings, compile cache state with **Check**, and the model folder with **Show in Finder**.
- **Models**: verified models with *In use* and *Selected · restart to use* tags and **Use This Model**, installs that need attention (rejected with an error code), and **Add Model** from Hugging Face (`org/name`) or from a YAML file, with per-file progress, cancel, conflict replacement and outcome.
- **Playground**: type a sentence and embed it through the running `/v1/embeddings` listener with an ephemeral, no-proxy, no-redirect HTTP client. It shows the dimension, norm and first values; it does not create another runtime. **Copy curl** and **Copy Python** produce examples with the real port, model and text, correctly escaped; the Python example uses only the standard library. Closing the window cancels only the client's wait.
- **Logs**: up to fifty recent events from `~/.embed-ane/logs/app.log`, read when the pane opens or on **Refresh**, plus **Open Log File**.

Load is enabled only for a verified model in a loadable state. Unload is disabled while work is queued, preparing, in flight, loading or unloading; the click handler re-reads state, and `LifecycleActor.unload()` (or the supervisor) remains the race-safe guard.

`InstalledModelLibrary` ignores hidden entries and plain files in the model root, rejects symlinks and invalid ids, and verifies every candidate with `ModelDownloader.verify`. Rejected candidates are listed with their error codes and never offered. A verification failure invalidates the list rather than keeping stale results. Selecting another model re-verifies it, asks for confirmation, saves the selection and restarts the server; models are never hot-swapped. Unload and idle eviction never trigger an automatic reload loop, and a failed load stays visible until you act.

## Settings

The Settings window has three tabs sharing one **Save** / **Reset** footer:

- **General**: Launch at login, port, model folder. The server listens on `127.0.0.1` only; the model itself is chosen in the Models pane.
- **Lifecycle**: Keep model loaded; when off, an *Unload after* preset (5 minutes, 30 minutes, 1 hour, Custom… in seconds); Auto load. With auto load on, the model unloads when idle and reloads on the next request (about 15–30 s with a warm compile cache, much longer if macOS has cleared it). With auto load off, requests fail until you load the model.
- **Advanced**: compute units, queue and batch limits, **Open config.yaml** and **Open Log File**.

Polling never overwrites an unsaved draft, and Save submits only changed fields. Invalid numbers are rejected before anything is written. `idle_timeout_s`, `auto_load` and `max_queue_depth` apply immediately. Port, model folder, batch limit and compute units are tagged **Restart**: after saving them, the banner's **Restart Now** calls `MenuBarModel.restartServer()`, which waits up to 30 s for in-flight work (giving up with a notice rather than interrupting requests), stops the server and worker, and starts them with the saved configuration. Settings changed through the authenticated HTTP API appear in the same view.

Compute units are shown as a read-only **Neural Engine** row, because the CoreML runtime only accepts `cpu_and_ne`. If a hand-edited `config.yaml` contains another value, the editor shows a warning and **Reset to Neural Engine**. A malformed configuration can be opened for repair; **Open config.yaml** creates a default file only when none exists.

`vision_resize_mode` (`smart` by default, or the legacy `origin-bucket`) is edited in `config.yaml` and preserved by Settings; changing it requires a restart. `EMBED_ANE_MODEL_ROOT`, if the app process actually inherits it, takes precedence over the file. Apps launched from Finder do not inherit a terminal's environment.

## Installing models

On the Models pane, **Choose YAML File…** opens a file picker; **Download** fetches `spec.yaml` from the given Hugging Face repository and then the model files. A remote spec must name the requested repository and endpoint and may not declare credentials. Progress and **Cancel** stay visible. A conflict with an existing install offers **Replace**, which reuses the spec already fetched (no second fetch of a mutable branch), so the replaced files are exactly the ones you reviewed. Lock conflicts and revision drift show plain errors without a Replace option. Temporary spec files are removed on every outcome. Cancelling keeps recoverable staging for a later resume. Startup, opening the menu or a window, Refresh, settings and model selection never start or replace a download.

For private repositories, put a token in `~/.embed-ane/hf-token` (see [downloader.md](downloader.md#credentials)); an `HF_TOKEN` environment variable takes precedence.

Progress is coalesced through a bounded `AsyncStream`. Percentages refer to the **current file**, not the whole bundle; resolution and verification show indeterminate progress. Success is shown only after the downloader reports a verified promotion. Installing never unloads or replaces the active model.

## Logs

`~/.embed-ane/logs/app.log` stores timestamps and fixed event codes only, capped at the latest 512 entries in private, atomically written storage. It never records request text, embeddings, tokens or raw error strings. A failed log write cannot change whether a configuration or install operation succeeded; problems opening the log are shown as notices. The worker writes operational lines to `worker.log` in the same folder.

## Login and quit

`MacLoginItemService` uses `SMAppService.mainApp`. Reading the status has no side effects; registration changes only on an explicit toggle. The UI shows the service's actual status, distinguishes *requires approval* from enabled, and links to the system Login Items pane. Registration needs an installed, signed app.

Quitting returns `terminateLater`, reads fresh state and always asks for confirmation, because requests can arrive while the dialog is open. Busy includes installs, settings commands, and preparing, queued, in-flight, loading and unloading work; unreadable state counts as busy. Cancel keeps serving. Quitting interrupts work in progress; there is no draining mode.

## Build and CI

`App/App.xcodeproj` has a shared **App** scheme, a local package reference to the repository root, and an arm64, macOS 15, Swift 6 strict-concurrency application target. Only `App/Sources` is in the Xcode sources phase; `App/Support` is linked as a package product. No signing team is embedded. The app is not sandboxed (it reads arbitrary model folders and state files); hardened runtime is enabled for signed builds. The icon is the Icon Composer document `App/AppIcon.icon` (see [`App/ICON.md`](../App/ICON.md)), which needs Xcode 26 or later.

```sh
swift test
bash scripts/build-app.sh
```

The script runs:

```sh
xcodebuild -project App/App.xcodeproj -scheme App -configuration "${EMBED_ANE_CONFIGURATION:-Debug}" \
  -destination 'generic/platform=macOS' \
  -derivedDataPath "${EMBED_ANE_DERIVED_DATA:-.build/AppDerivedData}" \
  -clonedSourcePackagesDirPath .build/AppSourcePackages \
  -onlyUsePackageVersionsFromResolvedFile \
  build CODE_SIGNING_ALLOWED=NO ARCHS=arm64 ONLY_ACTIVE_ARCH=NO
```

Before building it seeds the Xcode workspace lock with the root `Package.resolved` pins (as a version-2 file, so no root-specific origin hash is reused) and prints them; afterwards it checks that the pins still match. The root lock is never rewritten. `EMBED_ANE_MARKETING_VERSION` optionally sets the version. CI runs `swift build`, `swift test` and this unsigned compile; it never launches the app or loads a model.

## Tests

Unit tests are in `tests/EmbedANEAppSupportTests`. Controller tests use `MockPredictor` and `LifecycleActor` with injected configuration, catalogue, listener, login service and clock. Loopback tests run the real `InProcessAppSession`, `WorkerSession` and `SupervisorProxy` against Hummingbird servers to check shared lifecycle state, settings visibility, busy-unload refusal, worker readiness and eviction. Other tests cover `StatusPresentation`, the settings draft, the model catalogue, the event log, login items, the compile cache probe and Hugging Face installs over loopback. SwiftUI views are compile-checked only.

The Playground helpers have a separate check: `python3 tests/AppWindowChecks/check_dashboard_tools.py` compiles the Swift helper source, tests request, preview and error handling and redirect refusal, and runs the copied curl and Python examples against a disposable loopback server, including quotes, Unicode, control characters and shell metacharacters. It never launches the app or loads a model.

## Manual checks

UI behavior, login-item registration and real-model performance are checked by hand on a Mac:

1. Run the app with a temporary model root and an unused port. Confirm there is no Dock icon or automatic window, a missing model stays unloaded, no download starts, and the menu reopens while serving continues. Open the main window from the menu and with ⌘O; confirm there is a single window. Try an occupied port and confirm the Error state, notice and Restart Server. Check each glyph state in light and dark menu bars, and that a load shorter than 0.5 s does not animate.
2. Install a model from a YAML file and from Hugging Face; check per-file progress and cancel, the model list and the Model submenu. Check that a corrupt directory is rejected and that installing neither replaces nor loads the current model.
3. Switch models from the Model submenu and confirm the restart. Check the load transition, endpoint response, queue and percentiles. Use the Playground and run both copied examples with a non-default port and quoted text. Submit work and confirm Unload is disabled. Try idle eviction, then load again.
4. Edit live settings, then restart-only settings; check effective versus saved values, external settings changes, and that an unsaved draft survives polling. Check the folder picker, the Neural Engine row with its warning and reset, and Restart Now after a port change. Toggle Keep Model Loaded and Launch at Login from the menu and confirm Settings follows. Check window resizing, light and dark appearance, keyboard navigation and VoiceOver.
5. Open the log file. Cancel a quit while work is active and confirm serving continues. On a signed, installed build, enable login registration, approve it if asked, verify the status, then disable it.

## API references

Apple documentation: [MenuBarExtra](https://developer.apple.com/documentation/swiftui/menubarextra), [SMAppService](https://developer.apple.com/documentation/servicemanagement/smappservice), [NSApplicationDelegate.applicationShouldTerminate](https://developer.apple.com/documentation/appkit/nsapplicationdelegate/applicationshouldterminate(_:)).
