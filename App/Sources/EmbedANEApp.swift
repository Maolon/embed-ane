import EmbedANEAppSupport
import SwiftUI

@main
struct EmbedANEAppLauncher {
    static func main() {
        if AppWorkerRunner.isWorkerRequested() {
            AppWorkerRunner.run()
        } else {
            guard AppInstanceLock.acquire() else {
                fputs("Embed ANE is already running. Exiting duplicate instance.\n", stderr)
                return
            }
            EmbedANEApp.main()
        }
    }
}

struct EmbedANEApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    var body: some Scene {
        MenuBarExtra {
            MenuContent(environment: delegate.environment)
        } label: {
            StatusBarLabel(environment: delegate.environment)
        }
        .menuBarExtraStyle(.menu)

        Window("Embed ANE", id: "main") {
            MainWindow(environment: delegate.environment)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 860, height: 600)
        .defaultLaunchBehavior(.suppressed)
        .restorationBehavior(.disabled)

        Settings {
            AppSettingsView(environment: delegate.environment)
        }
    }
}

/// The menu bar glyph: one chip shape whose die shows the state.
@MainActor struct StatusBarLabel: View {
    let environment: AppEnvironment
    var body: some View {
        let status = environment.model.presentation
        Image(nsImage: StatusGlyph.image(for: status.kind, pulse: environment.animator.pulse))
            .accessibilityLabel("Embed ANE: \(status.title)")
            .help("Embed ANE: \(status.title)")
    }
}
