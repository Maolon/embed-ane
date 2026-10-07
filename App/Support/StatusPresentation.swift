import EmbedANECore
import Foundation

/// The six states a person can see. Server phase and model lifecycle are merged
/// here so the menu bar glyph, menu header and main window always agree.
public enum StatusKind: String, Sendable, Equatable, CaseIterable {
    case ready, standby, unloaded, working, error, stopped
}

/// The one action offered for the current state, in the menu and on Overview.
public enum PrimaryAction: Sendable, Equatable {
    case load, retryLoad, unload, startServer, restartServer

    public var title: String {
        switch self {
        case .load: "Load Model"
        case .retryLoad: "Retry Load"
        case .unload: "Unload Model"
        case .startServer: "Start Server"
        case .restartServer: "Restart Server"
        }
    }
}

public struct StatusPresentation: Sendable, Equatable {
    public let kind: StatusKind
    public let title: String
    /// Short state-specific explanation; nil when the model and endpoint say enough.
    public let detail: String?
    public let action: PrimaryAction?

    public init(kind: StatusKind, title: String, detail: String? = nil, action: PrimaryAction? = nil) {
        self.kind = kind; self.title = title; self.detail = detail; self.action = action
    }

    /// Typical reload with a warm compile cache (SPEC.md, lifecycle).
    public static let reloadEstimate = "~15–30 s"

    public static func make(phase: ServerPhase, snapshot: AppSnapshot?,
                            autoLoad: Bool, restarting: Bool = false) -> StatusPresentation {
        if restarting { return .init(kind: .working, title: "Restarting server…") }
        switch phase {
        case .starting: return .init(kind: .working, title: "Starting server…")
        case .stopping: return .init(kind: .working, title: "Stopping server…")
        case .stopped: return .init(kind: .stopped, title: "Server stopped", action: .startServer)
        case .failed:
            return .init(kind: .error, title: "Server stopped unexpectedly",
                         detail: "Restart the server to try again.", action: .restartServer)
        case .running:
            guard let snapshot else { return .init(kind: .working, title: "Starting server…") }
            switch snapshot.state {
            case .ready: return .init(kind: .ready, title: "Ready", action: .unload)
            case .loading: return .init(kind: .working, title: "Loading model…")
            case .unloading: return .init(kind: .working, title: "Unloading model…")
            case .failed:
                return .init(kind: .error, title: "Model failed to load", action: .retryLoad)
            case .unloaded:
                return autoLoad
                    ? .init(kind: .standby, title: "Standby",
                            detail: "Loads on next request (\(reloadEstimate))", action: .load)
                    : .init(kind: .unloaded, title: "Unloaded",
                            detail: "Requests fail until you load the model", action: .load)
            }
        }
    }
}

/// The menu bar animates Working only once it has lasted long enough to matter,
/// so quick transitions do not flicker.
public enum WorkingAnimation {
    public static let delay: Duration = .milliseconds(500)

    public static func shouldAnimate(workingSince start: ContinuousClock.Instant?,
                                     now: ContinuousClock.Instant) -> Bool {
        guard let start else { return false }
        return now - start >= delay
    }
}
