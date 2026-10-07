import AppKit
import EmbedANEAppSupport
import Foundation
import SwiftUI

/// The single regular window: Overview, Models, Playground and Logs in a sidebar.
/// Closing it never stops serving; it does not open at launch.
@MainActor struct MainWindow: View {
    let environment: AppEnvironment
    @Environment(\.colorScheme) private var scheme
    private var model: MenuBarModel { environment.model }
    private var router: WindowRouter { environment.router }

    var body: some View {
        NavigationSplitView {
            List(MainPane.allCases, selection: Binding<MainPane?>(
                get: { router.pane },
                set: { if let pane = $0 { router.pane = pane } })) { pane in
                Label(pane.title, systemImage: pane.symbol).tag(pane)
            }
            .navigationSplitViewColumnWidth(min: 170, ideal: 180, max: 220)
        } detail: {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    if let notice = model.notice { NoticeView(notice: notice, dismiss: model.dismissNotice) }
                    if !model.restartRequired.isEmpty {
                        RestartNotice(fields: model.restartRequired,
                                      restart: { Task { await environment.requestRestart() } },
                                      disabled: !model.canRestart)
                    }
                    switch router.pane {
                    case .overview: OverviewPane(environment: environment)
                    case .models: ModelsPane(environment: environment)
                    case .playground: playground
                    case .logs: LogsPane(environment: environment)
                    }
                }
                .padding(20)
                .frame(maxWidth: 760, alignment: .leading)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(AppStyle.canvas(scheme))
            .navigationTitle(router.pane.title)
        }
        .tint(AppStyle.accent)
        .frame(minWidth: 720, minHeight: 500)
    }

    private var playground: some View {
        let endpoint: DashboardEndpoint? = {
            guard model.phase == .running, let port = model.listeningPort, let current = model.effective else { return nil }
            return DashboardEndpoint(port: port, modelID: current.modelID)
        }()
        return EmbeddingPlayground(endpoint: endpoint,
            isReady: model.presentation.kind == .ready && !model.commanding && !model.quitting,
            guidance: guidance)
    }
    private var guidance: String {
        switch model.presentation.kind {
        case .ready: return ""
        case .working: return "\(model.presentation.title) You can try it when the status is Ready."
        case .standby: return "The model loads with your first request (\(StatusPresentation.reloadEstimate)). Load it on Overview to try it now."
        case .unloaded: return "Load the model on Overview, then click Embed."
        case .error, .stopped: return "Start the server on Overview to try an embedding."
        }
    }
}

// MARK: - Overview

@MainActor struct OverviewPane: View {
    let environment: AppEnvironment
    private var model: MenuBarModel { environment.model }
    private var status: StatusPresentation { model.presentation }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            AppCard {
                HStack(spacing: 14) {
                    Image(nsImage: StatusGlyph.image(for: status.kind, pulse: environment.animator.pulse, pointSize: 36))
                        .renderingMode(.template)
                        .foregroundStyle(glyphColor)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(status.title).font(.title2.bold()).accessibilityAddTraits(.isHeader)
                        Text(subtitle).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                    }
                    Spacer()
                    actionButton
                }
                HStack(spacing: 8) {
                    if let url = environment.endpointURL {
                        Text(verbatim: url)
                            .font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                        Text("OpenAI-compatible base URL").font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        CopyButton(title: "Copy endpoint URL", value: url)
                    } else {
                        Text("The local server is not listening.").font(.callout).foregroundStyle(.secondary)
                        Spacer()
                    }
                }
                .padding(10)
                .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                .controlSize(.small)
                metrics
            }
            AppCard {
                row("Lifecycle", detail: lifecycleSummary) {
                    Button("Change…") {
                        environment.router.settingsTab = .lifecycle
                        NSApplication.shared.activate(ignoringOtherApps: true)
                        openSettings()
                    }
                }
                Divider()
                row("Compile cache", detail: cacheSummary) {
                    Button("Check") { Task { await model.checkCacheHealth() } }
                }
                Divider()
                row("Model folder", detail: model.effective?.modelRoot ?? "—", monospaced: true) {
                    Button("Show in Finder") { environment.revealModelFolder() }.disabled(model.effective == nil)
                }
            }
            .controlSize(.small)
        }
    }
    @Environment(\.openSettings) private var openSettings

    private var subtitle: String {
        var parts: [String] = []
        if let detail = status.detail { parts.append(detail) }
        if let current = model.effective {
            parts.append(current.modelID)
            parts.append(model.snapshot?.isVisionEnabled == true ? "Vision on" : "Vision off")
        }
        parts.append("Local access only")
        return parts.joined(separator: " · ")
    }
    private var glyphColor: Color {
        switch status.kind {
        case .ready: AppStyle.accent
        case .error: AppStyle.failure
        default: .secondary
        }
    }
    @ViewBuilder private var actionButton: some View {
        switch status.action {
        case .unload?:
            Button(PrimaryAction.unload.title) { Task { await environment.requestUnload() } }
                .disabled(!model.canUnload)
                .help("Release the model's memory when no requests are active")
        case let action? where action == .load || action == .retryLoad:
            Button(action.title) { Task { await model.load() } }
                .buttonStyle(.borderedProminent).disabled(!model.canLoad)
        case .startServer?:
            Button(PrimaryAction.startServer.title) { model.start() }
                .buttonStyle(.borderedProminent).disabled(model.quitting)
        case .restartServer?:
            Button(PrimaryAction.restartServer.title) { Task { await environment.requestRestart() } }
                .buttonStyle(.borderedProminent).disabled(!model.canRestart)
        default:
            ProgressView().controlSize(.small)
        }
    }

    private var metrics: some View {
        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), alignment: .leading), count: 6), alignment: .leading, spacing: 12) {
            metric("Queue", value: model.snapshot.map { String($0.queueDepth) } ?? "—")
            metric("In flight", value: model.snapshot.map { String($0.inFlight) } ?? "—")
            metric("Preparing", value: model.snapshot.map { String($0.preparing) } ?? "—")
            metric("p50", value: model.snapshot.map { latency($0.p50NS, count: $0.windowCount) } ?? "—")
            metric("p95", value: model.snapshot.map { latency($0.p95NS, count: $0.windowCount) } ?? "—")
            metric("Memory", value: model.snapshot.map { memory($0.residentBytes) } ?? "—")
        }
    }
    private func metric(_ title: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(value).font(.system(.callout, design: .monospaced).weight(.medium)).monospacedDigit()
            Text(title).font(.caption).foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
    }
    private func row<Accessory: View>(_ title: String, detail: String, monospaced: Bool = false,
                                      @ViewBuilder accessory: () -> Accessory) -> some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.callout.weight(.medium))
                Text(detail).font(monospaced ? .system(.caption, design: .monospaced) : .caption)
                    .foregroundStyle(.secondary).textSelection(.enabled).lineLimit(2)
            }
            Spacer()
            accessory()
        }
    }
    private var lifecycleSummary: String {
        guard let current = model.effective else { return "—" }
        let idle = current.idleTimeoutS == 0 ? "Keeps the model loaded" : "Unloads after \(Self.duration(current.idleTimeoutS)) idle"
        return idle + " · " + (current.autoLoad ? "Auto-load on next request" : "Auto-load off")
    }
    private var cacheSummary: String {
        switch model.cacheWarm {
        case true?: "Warm — reloads take \(StatusPresentation.reloadEstimate)"
        case false?: "Cold — macOS cleared it; the next load may take 40+ minutes"
        case nil: "Not checked yet"
        }
    }
    static func duration(_ seconds: Double) -> String {
        if seconds.truncatingRemainder(dividingBy: 3600) == 0 { return "\(Int(seconds / 3600)) h" }
        if seconds.truncatingRemainder(dividingBy: 60) == 0 { return "\(Int(seconds / 60)) min" }
        return "\(Int(seconds)) s"
    }
    private func latency(_ nanoseconds: UInt64, count: Int) -> String {
        count == 0 ? "—" : String(format: "%.0f ms", Double(nanoseconds) / 1_000_000)
    }
    private func memory(_ bytes: UInt64) -> String {
        bytes == 0 ? "—" : ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
}

// MARK: - Logs

@MainActor struct LogsPane: View {
    let environment: AppEnvironment
    @State private var events: [RecentAppEvent] = []
    @State private var refreshing = false

    var body: some View {
        AppCard {
            HStack {
                Text("Recent events").font(.headline)
                Spacer()
                if refreshing { ProgressView().controlSize(.mini) }
                Button("Refresh") { Task { await refresh() } }.disabled(refreshing)
                Button("Open Log File") { Task { await environment.openLogs() } }
            }
            .controlSize(.small)
            if events.isEmpty {
                Text("No events yet.").font(.callout).foregroundStyle(.secondary)
            }
            ForEach(events) { entry in
                HStack(alignment: .firstTextBaseline, spacing: 12) {
                    Text(entry.date, format: .dateTime.month(.abbreviated).day().hour().minute().second())
                        .monospacedDigit().foregroundStyle(.secondary)
                    Text(entry.event.rawValue.replacingOccurrences(of: "_", with: " ").capitalized)
                }
                .font(.callout)
            }
            Text("~/.embed-ane/logs/app.log keeps the last 512 operational events. It never records request text, embeddings, tokens or raw errors.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .task { await refresh() }
    }
    private func refresh() async {
        guard !refreshing else { return }
        refreshing = true
        let entries = await environment.recentEvents()
        if !Task.isCancelled { events = entries }
        refreshing = false
    }
}

@MainActor struct NoticeView: View {
    let notice: AppNotice
    let dismiss: () -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Label("Something needs attention", systemImage: "exclamationmark.triangle").font(.callout.weight(.medium))
                Spacer()
                Button("Dismiss", action: dismiss).controlSize(.small)
            }
            Text(notice.message).font(.callout).textSelection(.enabled)
            DisclosureGroup("Details") { Text(notice.code).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
                .font(.caption)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(AppStyle.warning.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
    }
}
