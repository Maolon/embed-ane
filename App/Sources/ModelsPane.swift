import EmbedANEAppSupport
import SwiftUI

/// Everything about which models exist and which one serves: the verified list,
/// switching (via restart), installations that need attention, and adding models.
@MainActor struct ModelsPane: View {
    let environment: AppEnvironment
    @State private var source: Source = .hub
    @State private var hfRepo = AppHubDefaults.defaultRepo
    private var model: MenuBarModel { environment.model }

    private enum Source: String, CaseIterable, Identifiable {
        case hub = "Hugging Face", yaml = "YAML File"
        var id: String { rawValue }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            installed
            add
        }
    }

    private var installed: some View {
        AppCard {
            HStack {
                Text("Installed").font(.headline)
                Spacer()
                if model.scanning { ProgressView().controlSize(.mini).accessibilityLabel("Checking models") }
                Button("Refresh") { Task { await model.refreshInstalls() } }
                    .disabled(model.scanning || model.quitting)
            }
            .controlSize(.small)
            if model.models.isEmpty && model.rejectedInstalls.isEmpty && !model.scanning {
                Text("No verified models in this folder. Add one below, or choose another model folder in Settings.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            ForEach(model.models) { item in
                modelRow(item)
                Divider()
            }
            ForEach(model.rejectedInstalls) { item in
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) { Text(item.id); Tag(title: "Needs attention", color: AppStyle.warning) }
                        Text(item.code).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    }
                    Spacer()
                }
                .textSelection(.enabled)
                Divider()
            }
            Text("Only verified installations are listed. Switching models restarts the server.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func modelRow(_ item: VerifiedModel) -> some View {
        let inUse = item.id == model.effective?.modelID
        let pending = !inUse && item.id == model.desired?.modelID
        return HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(item.id).font(.callout.weight(.medium))
                    if inUse { Tag(title: "In use") }
                    if pending { Tag(title: "Selected · restart to use", color: AppStyle.warning) }
                }
                Text("commit " + item.commit.prefix(12)).font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
            }
            Spacer()
            if !inUse {
                Button(pending ? "Restart Now" : "Use This Model") {
                    Task { await environment.requestRestart(switchingTo: pending ? nil : item.id) }
                }
                .controlSize(.small)
                .disabled(!model.canRestart || model.saving)
            }
        }
        .textSelection(.enabled)
    }

    private var add: some View {
        AppCard {
            HStack {
                Text("Add Model").font(.headline)
                Spacer()
                Picker("Source", selection: $source) {
                    ForEach(Source.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden().fixedSize()
            }
            switch source {
            case .hub:
                HStack(spacing: 8) {
                    TextField("Repository (org/name)", text: $hfRepo)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(.callout, design: .monospaced))
                    Button("Download") { model.downloadFromHub(repo: hfRepo) }
                        .buttonStyle(.borderedProminent)
                        .disabled(!model.canInstall || hfRepo.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
                .disabled(!model.canInstall)
                Text("Downloads spec.yaml and the model files from Hugging Face into your model folder, then verifies every digest.")
                    .font(.caption).foregroundStyle(.secondary)
            case .yaml:
                HStack {
                    Text("Install from a digest-pinned model spec on this Mac.").font(.callout)
                    Spacer()
                    Button("Choose YAML File…") { Task { await environment.chooseModelSpec() } }
                        .buttonStyle(.borderedProminent).disabled(!model.canInstall)
                }
            }
            Text("Installing never replaces or activates the model in use.").font(.caption).foregroundStyle(.secondary)
            if let conflict = model.conflictResolution {
                HStack(spacing: 12) {
                    Label("\(conflict.modelID) is already installed with different files. Replace it?", systemImage: "exclamationmark.triangle")
                        .font(.caption)
                    Spacer()
                    Button("Replace") { model.resolveConflict(replace: true) }
                        .buttonStyle(.borderedProminent).tint(AppStyle.warning)
                    Button("Cancel") { model.dismissConflict() }
                }
                .padding(8)
                .background(AppStyle.warning.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
            }
            if model.installing { progress }
            if let outcome = model.installOutcome { result(outcome) }
        }
        .controlSize(.small)
    }

    private var progress: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()
            HStack {
                Text(phaseTitle).font(.callout.weight(.medium))
                Spacer()
                Button("Cancel") { model.cancelInstall() }
            }
            if let fraction = model.progress?.fraction {
                ProgressView(value: fraction)
                HStack {
                    Text("Current file: \(Int((fraction * 100).rounded()))%")
                    Spacer()
                    if let received = model.progress?.received, let total = model.progress?.total, total > 0 {
                        Text("\(formatBytes(received)) / \(formatBytes(total))").foregroundStyle(.secondary)
                    }
                }
                .font(.caption).monospacedDigit()
            } else {
                ProgressView().controlSize(.small)
            }
            if let path = model.progress?.path {
                Text(path).font(.system(.caption, design: .monospaced)).lineLimit(2).textSelection(.enabled)
            }
        }
    }
    private func result(_ outcome: MenuBarModel.InstallOutcome) -> some View {
        Group {
            switch outcome.kind {
            case .success:
                Label(outcome.message, systemImage: "checkmark.circle.fill")
                    .foregroundStyle(AppStyle.success)
            case .cancelled:
                Label(outcome.message, systemImage: "xmark.circle").foregroundStyle(.secondary)
            }
        }
        .font(.callout.weight(.medium))
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(8)
        .background(outcome.kind == .cancelled ? Color.secondary.opacity(0.08) : AppStyle.success.opacity(0.08),
                    in: RoundedRectangle(cornerRadius: 6))
        .textSelection(.enabled)
    }
    private var phaseTitle: String {
        switch model.progress?.phase {
        case "fetching_spec": "Fetching spec…"
        case "resolving": "Resolving revision…"
        case "downloading": "Downloading files…"
        case "verifying": "Verifying digests…"
        case "promoted": "Installed"
        case let other?: other.capitalized
        case nil: "Installing…"
        }
    }
    private func formatBytes(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
