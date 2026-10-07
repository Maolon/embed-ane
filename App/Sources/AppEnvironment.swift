import AppKit
import EmbedANEAppSupport
import EmbedANECore
import Foundation
import Observation
import ServiceManagement
import UniformTypeIdentifiers

@MainActor final class MacLoginItemService: LoginItemService {
    private let service = SMAppService.mainApp
    var state: LoginItemState {
        switch service.status {
        case .notRegistered: .notRegistered
        case .enabled: .enabled
        case .requiresApproval: .requiresApproval
        case .notFound: .notFound
        @unknown default: .notFound
        }
    }
    func register() throws { try service.register() }
    func unregister() async throws { try unregisterOnMainActor() }
    // Select the synchronous SDK overload without sending the retained
    // non-Sendable service object to a different executor under Swift 6.
    private func unregisterOnMainActor() throws { try service.unregister() }
    func openSystemSettings() { SMAppService.openSystemSettingsLoginItems() }
}

enum MainPane: String, CaseIterable, Identifiable {
    case overview, models, playground, logs
    var id: String { rawValue }
    var title: String {
        switch self {
        case .overview: "Overview"
        case .models: "Models"
        case .playground: "Playground"
        case .logs: "Logs"
        }
    }
    var symbol: String {
        switch self {
        case .overview: "gauge.with.needle"
        case .models: "shippingbox"
        case .playground: "text.bubble"
        case .logs: "list.bullet.rectangle"
        }
    }
}

enum SettingsTab: String, Hashable { case general, lifecycle, advanced }

/// Which pane the main window and Settings show, so the menu can deep-link.
@MainActor @Observable final class WindowRouter {
    var pane: MainPane = .overview
    var settingsTab: SettingsTab = .general
}

struct RecentAppEvent: Identifiable, Sendable {
    let id: Int
    let date: Date
    let event: AppEvent
}

@MainActor final class AppEnvironment {
    let model: MenuBarModel
    let login: LoginItemModel
    let router = WindowRouter()
    let animator = GlyphAnimator()
    private let store: ConfigurationStore
    private let log: AppEventLog
    private let io: AppIO

    init() {
        let store = ConfigurationStore()
        let log = AppEventLog(home: store.home)
        let io = AppIO()
        self.store = store; self.log = log; self.io = io
        let useInProcess = ProcessInfo.processInfo.environment["EMBED_ANE_USE_INPROCESS_SESSION"] == "1"
            || UserDefaults.standard.bool(forKey: "use_inprocess_session")
        model = MenuBarModel(services: AppServices(
            configuration: { try await io.run { try store.resolve() } },
            saveConfiguration: { configuration in try await io.run { try store.save(configuration) } },
            makeSession: { config in
                if useInProcess {
                    return try InProcessAppSession.production(configuration: config, store: store)
                } else {
                    return try WorkerSession.production(configuration: config, store: store)
                }
            },
            library: InstalledModelLibrary(), expandRoot: { store.expandedRoot($0) },
            record: { log.record($0) },
            cacheHealth: { config in
                var resolved = config
                resolved.modelRoot = store.expandedRoot(config.modelRoot)
                let probed = resolved
                return try? await io.run { E5CacheHealth.probe(configuration: probed).isWarm }
            }))
        login = LoginItemModel(service: MacLoginItemService())
        animator.start(observing: model)
    }

    /// Base URL for OpenAI-compatible clients, nil until the server listens.
    var endpointURL: String? {
        guard model.phase == .running, let port = model.listeningPort else { return nil }
        return "http://127.0.0.1:\(port)/v1"
    }
    func copyEndpoint() {
        guard let url = endpointURL else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(url, forType: .string)
    }

    private static let coldCacheWarning = "macOS has cleared the Neural Engine compile cache, so loading the model again may take 40 minutes or more. Freeing disk space keeps the cache."

    /// Unloading is cheap only while the compile cache is warm; ask first otherwise.
    func requestUnload() async {
        if await model.checkCacheHealth() == false {
            guard confirm(title: "Unload while the compile cache is cold?",
                          message: Self.coldCacheWarning, action: "Unload Anyway", style: .warning) else { return }
        }
        await model.unload()
    }
    /// Restarts the server, optionally switching to another verified model first.
    func requestRestart(switchingTo modelID: String? = nil) async {
        guard model.canRestart else { return }
        let cold = await model.checkCacheHealth() == false
        var message = "The server stops and loads the model again, usually in \(StatusPresentation.reloadEstimate). Requests in progress finish first."
        if cold { message += "\n\n" + Self.coldCacheWarning }
        guard confirm(title: modelID.map { "Restart with \($0)?" } ?? "Restart the server?",
                      message: message, action: "Restart", style: cold ? .warning : .informational) else { return }
        if let modelID {
            await model.selectModel(modelID)
            guard model.desired?.modelID == modelID else { return }
        }
        await model.restartServer()
    }
    private func confirm(title: String, message: String, action: String, style: NSAlert.Style) -> Bool {
        let alert = NSAlert()
        alert.alertStyle = style
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: action)
        alert.addButton(withTitle: "Cancel")
        NSApplication.shared.activate(ignoringOtherApps: true)
        return alert.runModal() == .alertFirstButtonReturn
    }
    func revealModelFolder() {
        guard let root = model.effective?.modelRoot else { return }
        let url = URL(fileURLWithPath: (root as NSString).expandingTildeInPath, isDirectory: true)
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
    func chooseModelSpec() async {
        let panel = NSOpenPanel()
        panel.title = "Install model from YAML"
        panel.message = "Select an immutable, digest-pinned model specification. Installation does not replace or activate an existing model."
        panel.prompt = "Install"
        panel.canChooseFiles = true; panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [UTType(filenameExtension: "yaml") ?? .text,
                                     UTType(filenameExtension: "yml") ?? .text]
        NSApplication.shared.activate(ignoringOtherApps: true)
        guard await panel.begin() == .OK, let url = panel.url else { return }
        model.install(from: url)
    }
    /// A read-only snapshot for the Logs pane, newest first, requested only on open or Refresh.
    /// Never creates the file, polls, or reports preview failures as app errors.
    func recentEvents(limit: Int = 50) async -> [RecentAppEvent] {
        let file = log.file
        do {
            return try await io.run {
                let storage = StateFileStorage(directory: file.deletingLastPathComponent())
                guard let data = try storage.read(file.lastPathComponent, maximumBytes: 131_072, requirePrivate: true) else { return [] }
                let formatter = ISO8601DateFormatter()
                return String(decoding: data, as: UTF8.self).split(separator: "\n").suffix(limit).reversed().enumerated().compactMap { index, line in
                    let parts = line.split(separator: " ")
                    guard parts.count == 2, let date = formatter.date(from: String(parts[0])),
                          let event = AppEvent(rawValue: String(parts[1])) else { return nil }
                    return RecentAppEvent(id: index, date: date, event: event)
                }
            }
        } catch { return [] }
    }
    func openLogs() async {
        do {
            let url = try await log.prepareFile()
            guard NSWorkspace.shared.open(url) else {
                throw EmbedANEError.io(path: url.path, reason: "macOS could not open the log file.")
            }
        } catch { model.report(error) }
    }
    func openConfiguration() async {
        do {
            try await io.run { [store] in try store.saveIfAbsent(ServiceConfiguration()) }
            guard NSWorkspace.shared.open(store.file) else {
                throw EmbedANEError.io(path: store.file.path, reason: "macOS could not open the configuration file.")
            }
        } catch { model.report(error) }
    }
}
