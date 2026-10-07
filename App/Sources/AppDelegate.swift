import AppKit
import EmbedANEAppSupport

@MainActor final class AppDelegate: NSObject, NSApplicationDelegate {
    let environment = AppEnvironment()
    private var decidingTermination = false

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApplication.shared.setActivationPolicy(.accessory)
        environment.login.refresh() // Status-only; never registers implicitly.
        environment.model.start()
    }
    func applicationDidBecomeActive(_ notification: Notification) { environment.login.refresh() }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard !decidingTermination else { return .terminateLater }
        decidingTermination = true
        Task { [self] in
            let busy = await environment.model.workAtQuit()
            let alert = NSAlert()
            alert.alertStyle = busy ? .warning : .informational
            alert.messageText = busy ? "Quit while work is active?" : "Quit Embed ANE?"
            alert.informativeText = busy
                ? "Loading, inference, queued work, or installation is active. Quitting stops this process and interrupts its requests. Recoverable download staging is retained."
                : "Quitting stops the local embeddings endpoint. Requests that arrive before the process exits may be interrupted."
            // Always confirm, including a quiet snapshot: clients can admit a new
            // request while a quit dialog is open. No misleading drain promise.
            alert.addButton(withTitle: "Cancel")
            alert.addButton(withTitle: busy ? "Quit Anyway" : "Quit")
            NSApplication.shared.activate(ignoringOtherApps: true)
            let approved = alert.runModal() == .alertSecondButtonReturn
            if approved { environment.model.cancelForTermination() }
            decidingTermination = false
            sender.reply(toApplicationShouldTerminate: approved)
        }
        return .terminateLater
    }
}
