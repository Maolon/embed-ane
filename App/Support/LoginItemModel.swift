import Foundation
import Observation

public enum LoginItemState: Sendable, Equatable {
    case notRegistered, enabled, requiresApproval, notFound
    public var requested: Bool { self == .enabled || self == .requiresApproval }
    public var description: String {
        switch self {
        case .notRegistered: "Off"
        case .enabled: "Enabled"
        case .requiresApproval: "Approval required in System Settings"
        case .notFound: "App service unavailable; install the built app first"
        }
    }
}
@MainActor public protocol LoginItemService: AnyObject {
    var state: LoginItemState { get }
    func register() throws
    func unregister() async throws
    func openSystemSettings()
}

@MainActor @Observable public final class LoginItemModel {
    public private(set) var state: LoginItemState = .notRegistered
    public private(set) var updating = false
    public private(set) var error: String?
    @ObservationIgnored private let service: any LoginItemService
    public init(service: any LoginItemService) { self.service = service; refresh() }
    /// Reading status must never implicitly opt the user into launch-at-login.
    public func refresh() { state = service.state }
    public func setEnabled(_ enabled: Bool) async {
        guard !updating else { return }
        updating = true; error = nil
        defer { updating = false; refresh() }
        do {
            if enabled {
                if !service.state.requested { try service.register() }
            } else if service.state.requested {
                try await service.unregister()
            }
        } catch {
            self.error = "macOS could not change the login item. Check System Settings and the installed app's signing status."
        }
    }
    public func openSystemSettings() { service.openSystemSettings() }
}
