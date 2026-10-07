import AppKit
import EmbedANECore
import SwiftUI

/// App-only presentation tokens. Native controls still follow macOS appearance,
/// keyboard navigation and accessibility settings.
enum AppStyle {
    static let space: CGFloat = 8
    static let corner: CGFloat = 12
    /// Teal of the app icon's lit cores; reads on light and dark canvases.
    static let accent = Color(red: 14 / 255, green: 143 / 255, blue: 132 / 255)
    static let success = Color.green
    static let warning = Color.orange
    static let failure = Color.red

    static func canvas(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 18 / 255, green: 18 / 255, blue: 20 / 255) : Color(nsColor: .windowBackgroundColor)
    }
    static func panel(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color(red: 30 / 255, green: 30 / 255, blue: 36 / 255) : Color(nsColor: .controlBackgroundColor)
    }
    static func border(_ scheme: ColorScheme) -> Color {
        scheme == .dark ? Color.white.opacity(0.12) : Color.black.opacity(0.10)
    }
}

struct AppCard<Content: View>: View {
    @Environment(\.colorScheme) private var scheme
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: AppStyle.space * 1.5) { content }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(AppStyle.space * 2)
            .background(AppStyle.panel(scheme), in: RoundedRectangle(cornerRadius: AppStyle.corner))
            .overlay(RoundedRectangle(cornerRadius: AppStyle.corner).strokeBorder(AppStyle.border(scheme)))
    }
}

struct CardHeading: View {
    let title: String
    let symbol: String
    var body: some View {
        Label(title, systemImage: symbol).font(.headline)
            .accessibilityAddTraits(.isHeader)
    }
}

struct StatusPill: View {
    let title: String
    let color: Color
    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 7, height: 7).accessibilityHidden(true)
            Text(title).font(.caption.weight(.semibold))
        }
        .padding(.horizontal, 10).padding(.vertical, 5)
        .background(color.opacity(0.12), in: Capsule())
        .accessibilityElement(children: .combine)
    }
}

struct HTTPBadge: View {
    let method: String
    var body: some View {
        Text(method).font(.system(.caption2, design: .monospaced).bold())
            .foregroundStyle(method == "GET" ? AppStyle.success : AppStyle.accent)
            .padding(.horizontal, 6).padding(.vertical, 3)
            .background((method == "GET" ? AppStyle.success : AppStyle.accent).opacity(0.12), in: RoundedRectangle(cornerRadius: 4))
    }
}

@MainActor struct CopyButton: View {
    let title: String
    let value: String
    var iconOnly = false
    @State private var copied = false

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            copied = NSPasteboard.general.setString(value, forType: .string)
        } label: {
            if iconOnly {
                Image(systemName: copied ? "checkmark" : "doc.on.doc")
            } else {
                Label(copied ? "Copied" : title, systemImage: copied ? "checkmark" : "doc.on.doc")
            }
        }
        .accessibilityLabel(copied ? "Copied" : title)
        .help(title)
        .onChange(of: value) { copied = false }
        .task(id: copied) {
            guard copied else { return }
            do { try await Task.sleep(for: .seconds(2)) } catch { return }
            copied = false
        }
    }
}

struct SettingsRow<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content
    var body: some View {
        HStack(spacing: AppStyle.space * 2) {
            Text(title).frame(width: 140, alignment: .leading)
            content
            Spacer(minLength: 0)
        }
        .frame(minHeight: 32)
    }
}

struct RestartNotice: View {
    let fields: [String]
    let restart: () -> Void
    var disabled = false
    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Label("Saved changes apply after a restart", systemImage: "arrow.clockwise").font(.callout.weight(.medium))
                Text(fields.map(Self.label).joined(separator: " · ")).foregroundStyle(.secondary)
            }
            Spacer()
            Button("Restart Now", action: restart).disabled(disabled)
        }
        .font(.caption)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(AppStyle.warning.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
    }
    static func label(_ field: String) -> String {
        switch field {
        case "port": "Port"
        case "model_id": "Model"
        case "model_root": "Model folder"
        case "compute_units": "Compute units"
        case "max_batch": "Batch limit"
        default: field.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }
}

/// Marks a setting that takes effect only after Restart Now.
struct RestartTag: View {
    var body: some View {
        Text("Restart").font(.caption2.weight(.medium)).foregroundStyle(.secondary)
            .padding(.horizontal, 5).padding(.vertical, 1)
            .overlay(Capsule().strokeBorder(.secondary.opacity(0.5)))
            .help("Takes effect after the server restarts")
    }
}

/// A small capsule tag for list rows ("In use", "Needs attention").
struct Tag: View {
    let title: String
    var color: Color = AppStyle.accent
    var body: some View {
        Text(title).font(.caption2.weight(.semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 6).padding(.vertical, 2)
            .background(color.opacity(0.14), in: Capsule())
    }
}
