import EmbedANEAppSupport
import Testing

@MainActor private final class FixtureLoginItem: LoginItemService {
    var state: LoginItemState = .notRegistered
    var registers = 0, unregisters = 0, settingsOpens = 0
    var shouldFail = false
    func register() throws {
        registers += 1
        if shouldFail { throw Failure.registration }
        state = .requiresApproval
    }
    func unregister() async throws {
        unregisters += 1
        if shouldFail { throw Failure.registration }
        state = .notRegistered
    }
    func openSystemSettings() { settingsOpens += 1 }
    enum Failure: Error { case registration }
}
@Suite("Launch-at-login is opt-in", .serialized) @MainActor struct LoginItemTests {
    @Test func initializationAndRefreshNeverRegister() {
        let service = FixtureLoginItem(); let model = LoginItemModel(service: service)
        model.refresh(); model.refresh()
        #expect(service.registers == 0 && service.unregisters == 0)
        #expect(model.state == .notRegistered)
    }
    @Test func explicitEnableReportsActualApprovalState() async {
        let service = FixtureLoginItem(); let model = LoginItemModel(service: service)
        await model.setEnabled(true)
        #expect(service.registers == 1)
        #expect(model.state == .requiresApproval && model.state.requested)
        #expect(model.error == nil)
    }
    @Test func pendingRegistrationIsNotRepeated() async {
        let service = FixtureLoginItem(); service.state = .requiresApproval
        let model = LoginItemModel(service: service)
        await model.setEnabled(true)
        #expect(service.registers == 0)
    }
    @Test func explicitDisableUnregisters() async {
        let service = FixtureLoginItem(); service.state = .enabled
        let model = LoginItemModel(service: service)
        await model.setEnabled(false)
        #expect(service.unregisters == 1 && model.state == .notRegistered)
    }
    @Test func failureDoesNotOptimisticallyClaimRegistration() async {
        let service = FixtureLoginItem(); service.shouldFail = true
        let model = LoginItemModel(service: service)
        await model.setEnabled(true)
        #expect(model.state == .notRegistered)
        #expect(model.error != nil && !model.updating)
    }
    @Test func externalSystemSettingsChangesAreObserved() {
        let service = FixtureLoginItem(); let model = LoginItemModel(service: service)
        service.state = .enabled; model.refresh()
        #expect(model.state == .enabled)
        model.openSystemSettings()
        #expect(service.settingsOpens == 1)
        #expect(service.registers == 0)
    }
}
