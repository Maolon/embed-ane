# Embed ANE menu bar app

The shared **App** scheme in `App.xcodeproj` builds **Embed ANE.app**, a macOS 15+ arm64 accessory app, from the Swift package one directory above this folder. The app hosts the HTTP server itself (with the model in a worker child process); it does not spawn the `embed-ane` CLI or attach to another running server.

From the repository root:

```sh
swift test
bash scripts/build-app.sh
```

The second command compiles the app unsigned with the root dependency pins. It does not launch the app or load a model. To run the app, open `App/App.xcodeproj`, select the App scheme, choose a signing team if needed, and run. Login-item registration only works for an installed, signed app.

Layout:

- `Sources/`: SwiftUI and AppKit views, compiled only by the Xcode application target.
- `Support/`: app logic compiled once as the `EmbedANEAppSupport` package target, so it is covered by `swift test`.
- `AppIcon.icon`: the Icon Composer app icon (see [`ICON.md`](ICON.md)).

See [`../docs/app.md`](../docs/app.md) for the worker process, status states, settings, model installation, logs and manual checks.
