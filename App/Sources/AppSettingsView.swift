import AppKit
import EmbedANEAppSupport
import EmbedANECore
import SwiftUI

@MainActor struct AppSettingsView: View {
    let environment: AppEnvironment
    @State private var draft = SettingsDraft(ServiceConfiguration())
    @State private var previousIdleTimeout = "300"
    @State private var idlePreset: IdleTimeoutPreset = .fiveMinutes
    @Environment(\.colorScheme) private var scheme
    private var model: MenuBarModel { environment.model }
    private var editingDisabled: Bool { model.saving || model.quitting || model.phase == .starting || model.configuration == nil }

    var body: some View {
        VStack(spacing: 0) {
            TabView(selection: Binding(get: { environment.router.settingsTab },
                                       set: { environment.router.settingsTab = $0 })) {
                page { general }.tabItem { Label("General", systemImage: "gearshape") }.tag(SettingsTab.general)
                page { lifecycleSettings }.tabItem { Label("Lifecycle", systemImage: "arrow.triangle.2.circlepath") }.tag(SettingsTab.lifecycle)
                page { advanced }.tabItem { Label("Advanced", systemImage: "slider.horizontal.3") }.tag(SettingsTab.advanced)
            }
            Divider()
            footer
        }
        .frame(width: 560, height: 520)
        .background(AppStyle.canvas(scheme))
        .tint(AppStyle.accent)
        .onAppear { reload() }
        .onChange(of: model.desired) { _, value in
            if !draft.hasChanges, let value { setDraft(value) }
        }
    }

    private func page<Content: View>(@ViewBuilder _ content: () -> Content) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                if model.configuration == nil {
                    AppCard {
                        CardHeading(title: "Configuration needs attention", symbol: "exclamationmark.triangle")
                        Text("Open config.yaml in Advanced to repair its values, then start the server from the menu.")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                }
                if !model.restartRequired.isEmpty {
                    RestartNotice(fields: model.restartRequired,
                                  restart: { Task { await environment.requestRestart() } },
                                  disabled: !model.canRestart)
                }
                if let notice = model.notice { NoticeView(notice: notice, dismiss: model.dismissNotice) }
                content()
            }
            .padding(16)
        }
    }

    private var footer: some View {
        HStack(spacing: 12) {
            Text(draft.hasChanges ? "Unsaved changes" : "No unsaved changes")
                .font(.caption).foregroundStyle(.secondary)
            Spacer()
            Button("Reset") { reload() }.disabled(model.saving || !draft.hasChanges)
                .help("Discard edits and reload saved values")
            Button(model.saving ? "Saving…" : "Save") {
                let edited = draft
                Task {
                    await model.save(edited)
                    if model.notice == nil { reload() }
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(!draft.hasChanges || editingDisabled)
            .keyboardShortcut(.defaultAction)
        }
        .controlSize(.small).padding(12)
    }

    private var general: some View {
        Group {
            AppCard {
                CardHeading(title: "Startup", symbol: "power")
                Toggle("Launch at login", isOn: Binding(
                    get: { environment.login.state.requested },
                    set: { enabled in Task { await environment.login.setEnabled(enabled) } }))
                    .toggleStyle(.switch)
                    .disabled(environment.login.updating)
                Text("Start the local server when you sign in. Applies immediately. " + environment.login.state.description + ".")
                    .font(.caption).foregroundStyle(.secondary)
                if let error = environment.login.error {
                    Label(error, systemImage: "exclamationmark.circle").font(.caption).foregroundStyle(AppStyle.failure)
                }
                Button("Open Login Items Settings") { environment.login.openSystemSettings() }.controlSize(.small)
            }
            serverSettings
        }
    }

    private var serverSettings: some View {
        AppCard {
            CardHeading(title: "Server", symbol: "network")
            VStack(spacing: 8) {
                SettingsRow(title: "Port") {
                    TextField("Port", text: $draft.port).labelsHidden()
                        .font(.system(.body, design: .monospaced)).frame(width: 100)
                    RestartTag()
                }
                SettingsRow(title: "Model folder") {
                    TextField("Model folder", text: $draft.modelRoot).labelsHidden()
                        .font(.system(.body, design: .monospaced))
                    Button("…") { Task { await chooseModelRoot() } }
                        .help("Choose model folder").accessibilityLabel("Choose model folder")
                    RestartTag()
                }
            }
            .textFieldStyle(.roundedBorder)
            .disabled(editingDisabled)
            Text("The server listens on 127.0.0.1 only. Choose which model serves in the Models pane of the main window.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var lifecycleSettings: some View {
        AppCard {
            CardHeading(title: "Model memory", symbol: "memorychip")
            Text("Changes here apply when you save; no restart needed.").font(.caption).foregroundStyle(.secondary)
            Toggle("Keep model loaded", isOn: Binding(
                get: { Double(draft.idleTimeoutS) == 0 },
                set: { keepLoaded in
                    if keepLoaded {
                        if let value = Double(draft.idleTimeoutS), value.isFinite, value > 0 {
                            previousIdleTimeout = draft.idleTimeoutS
                        }
                        draft.idleTimeoutS = draft.baseline.idleTimeoutS == 0 ? String(draft.baseline.idleTimeoutS) : "0"
                    } else {
                        // A draft suggestion only. The runtime default remains 0
                        // (keep loaded), and only Save changes persists edits.
                        draft.idleTimeoutS = previousIdleTimeout
                        idlePreset = IdleTimeoutPreset.from(seconds: Double(previousIdleTimeout) ?? 300)
                    }
                }))
                .toggleStyle(.switch)
                .disabled(editingDisabled)
            if Double(draft.idleTimeoutS) != 0 {
                VStack(alignment: .leading, spacing: 8) {
                    Picker("Unload after", selection: Binding<IdleTimeoutPreset>(
                        get: {
                            if idlePreset == .custom { return IdleTimeoutPreset.custom }
                            return IdleTimeoutPreset.from(seconds: Double(draft.idleTimeoutS) ?? 0)
                        },
                        set: { (newPreset: IdleTimeoutPreset) in
                            idlePreset = newPreset
                            if let seconds = newPreset.seconds {
                                draft.idleTimeoutS = String(Int(seconds))
                            }
                        })) {
                        ForEach(IdleTimeoutPreset.allCases) { preset in
                            Text(preset.label).tag(preset)
                        }
                    }
                    .labelsHidden()
                    .disabled(editingDisabled)
                    if idlePreset == .custom {
                        HStack {
                            Text("Unload after")
                            TextField("Idle timeout in seconds", text: $draft.idleTimeoutS).labelsHidden()
                                .textFieldStyle(.roundedBorder).monospacedDigit().frame(width: 90)
                            Text("seconds idle").foregroundStyle(.secondary)
                        }
                        .disabled(editingDisabled)
                    }
                    if draft.autoLoad {
                        Text("Model auto-unloads after the idle period and auto-reloads on the next request (\(StatusPresentation.reloadEstimate); much longer if macOS has cleared the compile cache).")
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("Model auto-unloads after the idle period. The next request will fail until you manually load the model.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                .padding(.leading, 24)
            } else {
                Text("Keeps the model ready between requests. Uses more memory.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Toggle("Auto load", isOn: $draft.autoLoad)
                .toggleStyle(.switch)
                .disabled(editingDisabled)
            Text("Reloads the model automatically when a request arrives.")
                .font(.caption).foregroundStyle(.secondary)
            Text("These settings apply as soon as you save.").font(.caption).foregroundStyle(.secondary)
        }
    }

    private var advanced: some View {
        Group {
            AppCard {
                CardHeading(title: "Compute", symbol: "cpu")
                SettingsRow(title: "Compute units") {
                    if draft.isComputeUnitsSupported {
                        Text("Neural Engine")
                    } else {
                        Text(draft.computeUnits.appLabel).foregroundStyle(AppStyle.failure)
                        Button("Reset to Neural Engine") { draft.resetComputeUnits() }
                            .controlSize(.small).disabled(editingDisabled)
                    }
                    RestartTag()
                }
                if let warning = draft.computeUnitsWarning {
                    Label(warning, systemImage: "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(AppStyle.failure)
                }
            }
            AppCard {
                CardHeading(title: "Request limits", symbol: "line.3.horizontal.decrease")
                SettingsRow(title: "Queue limit") {
                    TextField("Maximum queue depth", text: $draft.maxQueueDepth).labelsHidden()
                        .frame(width: 100).monospacedDigit()
                }
                SettingsRow(title: "Batch limit") {
                    TextField("Maximum batch, 1 to 8", text: $draft.maxBatch).labelsHidden()
                        .frame(width: 100).monospacedDigit()
                    Text("1–8").font(.caption).foregroundStyle(.secondary)
                    RestartTag()
                }
                Text("Queue: requests allowed to wait. Batch: texts accepted in one request.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .textFieldStyle(.roundedBorder).disabled(editingDisabled)
            AppCard {
                CardHeading(title: "Files", symbol: "doc.text")
                HStack {
                    Button("Open config.yaml") { Task { await environment.openConfiguration() } }
                    Button("Open Log File") { Task { await environment.openLogs() } }
                }
                .controlSize(.small)
                Text("Precedence: CLI flag > environment > file > default. EMBED_ANE_MODEL_ROOT, when inherited by this app, overrides the file at launch.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func chooseModelRoot() async {
        let panel = NSOpenPanel()
        panel.title = "Choose model folder"
        panel.message = "Choose the folder that contains your installed models."
        panel.prompt = "Choose"
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: (draft.modelRoot as NSString).expandingTildeInPath, isDirectory: true)
        guard await panel.begin() == .OK, let url = panel.url, !editingDisabled else { return }
        draft.modelRoot = url.path
    }
    private func setDraft(_ configuration: ServiceConfiguration) {
        draft = SettingsDraft(configuration)
        previousIdleTimeout = configuration.idleTimeoutS > 0 ? String(configuration.idleTimeoutS) : "300"
        idlePreset = IdleTimeoutPreset.from(seconds: configuration.idleTimeoutS > 0 ? configuration.idleTimeoutS : 300)
    }
    private func reload() {
        if let config = model.desired { setDraft(config) }
        environment.login.refresh()
    }
}
