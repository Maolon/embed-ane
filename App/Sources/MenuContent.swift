import AppKit
import EmbedANEAppSupport
import SwiftUI

/// Native menu: status, model actions, quick toggles, windows. Low-frequency
/// work (installing, logs) lives in the main window.
@MainActor struct MenuContent: View {
    let environment: AppEnvironment
    @Environment(\.openSettings) private var openSettings
    @Environment(\.openWindow) private var openWindow
    private var model: MenuBarModel { environment.model }
    private var status: StatusPresentation { model.presentation }

    var body: some View {
        Group {
            Button { show(.overview) } label: {
                Label {
                    Text(status.title)
                } icon: {
                    Image(nsImage: StatusGlyph.image(for: status.kind, pulse: environment.animator.pulse))
                }
            }
            ForEach(statusLines, id: \.self) { Text($0).disabled(true) }
            if let url = environment.endpointURL {
                Text(verbatim: url).disabled(true)
                Button("Copy Endpoint URL") { environment.copyEndpoint() }
            }
            Divider()
            primaryAction
            modelMenu
            Divider()
            Toggle("Keep Model Loaded", isOn: Binding(
                get: { model.effective?.idleTimeoutS == 0 },
                set: { keep in Task { await model.setKeepLoaded(keep) } }))
                .disabled(model.saving || model.quitting || model.effective == nil)
            Toggle("Launch at Login", isOn: Binding(
                get: { environment.login.state.requested },
                set: { enabled in Task { await environment.login.setEnabled(enabled) } }))
                .disabled(environment.login.updating)
            Divider()
            Button("Open Embed ANE…") { show(environment.router.pane) }
                .keyboardShortcut("o")
            Button("Settings…") {
                NSApplication.shared.activate(ignoringOtherApps: true)
                openSettings()
            }
            .keyboardShortcut(",")
            Divider()
            Button("Quit Embed ANE") { NSApplication.shared.terminate(nil) }
                .keyboardShortcut("q")
        }
    }

    /// Model and context under the title; the endpoint gets its own line.
    private var statusLines: [String] {
        var lines: [String] = []
        if let detail = status.detail { lines.append(detail) }
        if let current = model.effective, status.kind == .ready || status.kind == .standby || status.kind == .unloaded {
            let vision = model.snapshot?.isVisionEnabled == true ? " · Vision on" : ""
            lines.append(current.modelID + vision)
        }
        if let snapshot = model.snapshot, snapshot.inFlight + snapshot.queueDepth > 0 {
            lines.append("\(snapshot.inFlight) in flight · queue \(snapshot.queueDepth)")
        }
        if model.cacheWarm == false { lines.append("Compile cache cold — reloads may take 40+ min") }
        return lines
    }

    @ViewBuilder private var primaryAction: some View {
        switch status.action {
        case .unload?:
            Button(PrimaryAction.unload.title) { Task { await environment.requestUnload() } }
                .disabled(!model.canUnload)
        case let action? where action == .load || action == .retryLoad:
            Button(action.title) { Task { await model.load() } }
                .disabled(!model.canLoad)
        case .startServer?:
            Button(PrimaryAction.startServer.title) { model.start() }
                .disabled(model.quitting)
        case .restartServer?:
            Button(PrimaryAction.restartServer.title) { Task { await environment.requestRestart() } }
                .disabled(!model.canRestart)
        default:
            Button(status.title) {}.disabled(true)
        }
    }

    private var modelMenu: some View {
        Menu("Model") {
            ForEach(model.models) { item in
                Toggle(item.id, isOn: Binding(
                    get: { item.id == model.effective?.modelID },
                    set: { chosen in
                        guard chosen, item.id != model.effective?.modelID else { return }
                        Task { await environment.requestRestart(switchingTo: item.id) }
                    }))
            }
            if model.models.isEmpty { Text("No verified models").disabled(true) }
            Divider()
            Button("Manage Models…") { show(.models) }
        }
        .disabled(model.quitting)
    }

    private func show(_ pane: MainPane) {
        environment.router.pane = pane
        NSApplication.shared.activate(ignoringOtherApps: true)
        openWindow(id: "main")
    }
}
