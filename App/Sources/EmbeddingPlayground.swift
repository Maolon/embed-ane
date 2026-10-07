import SwiftUI

@MainActor struct EmbeddingPlayground: View {
    let endpoint: DashboardEndpoint?
    let isReady: Bool
    let guidance: String
    @State private var text = "A small step, a new idea."
    @State private var preview: EmbeddingPreview?
    @State private var error: String?
    @State private var requestTask: Task<Void, Never>?
    @State private var requestID: UUID?

    private var working: Bool { requestID != nil }
    private var hasText: Bool { !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    private var canEmbed: Bool { endpoint != nil && isReady && !working && hasText }

    var body: some View {
        AppCard {
            HStack {
                CardHeading(title: "Try it", symbol: "text.bubble")
                Spacer()
                HTTPBadge(method: "POST")
                Text("/v1/embeddings").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
            }
            HStack(alignment: .top, spacing: 12) {
                TextField("Enter a sentence", text: $text, axis: .vertical)
                    .lineLimit(1...3)
                    .textFieldStyle(.roundedBorder)
                    .controlSize(.large)
                    .disabled(working)
                    .onSubmit { embed() }
                Button(action: embed) {
                    Label(working ? "Embedding…" : "Embed", systemImage: "arrow.up.right")
                }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(!canEmbed)
                .keyboardShortcut(.return, modifiers: .command)
                .help("Create an embedding (⌘Return)")
            }
            result
            Divider()
            HStack(spacing: 8) {
                Text("Use in your app").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if let endpoint, let examples = try? endpoint.examples(text: text) {
                    CopyButton(title: "Copy curl", value: examples.curl).disabled(!hasText)
                    CopyButton(title: "Copy Python", value: examples.python).disabled(!hasText)
                        .help("Copy a Python 3 example. No packages to install.")
                } else {
                    Text("Examples appear when the server starts.").font(.caption).foregroundStyle(.secondary)
                }
            }
            .controlSize(.small)
        }
        .onChange(of: text) { preview = nil; error = nil }
        .onChange(of: endpoint) { cancelRequest(); preview = nil; error = nil }
        .onDisappear { cancelRequest() }
    }

    @ViewBuilder private var result: some View {
        if working {
            HStack(spacing: 8) {
                ProgressView().controlSize(.mini)
                Text("Creating your embedding locally…").font(.caption).foregroundStyle(.secondary)
            }
        } else if let error {
            Label(error, systemImage: "exclamationmark.circle")
                .font(.callout).foregroundStyle(AppStyle.failure)
                .textSelection(.enabled)
        } else if let preview {
            VStack(alignment: .leading, spacing: 6) {
                Label("\(preview.dimension) dimensions · norm \(String(format: "%.6f", preview.norm))", systemImage: "checkmark.circle.fill")
                    .font(.callout.weight(.medium)).foregroundStyle(AppStyle.success)
                Text("First \(preview.firstValues.count): [" + preview.firstValues.map { String(format: "%.4f", $0) }.joined(separator: ", ") + "]")
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
            }
            .accessibilityElement(children: .combine)
        } else {
            Text(isReady ? "Click Embed to turn this sentence into a vector. Text stays on this Mac." : guidance)
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func embed() {
        guard canEmbed, let endpoint else { return }
        let id = UUID(), input = text
        requestID = id; preview = nil; error = nil
        requestTask = Task {
            do {
                let result = try await PlaygroundClient.embed(text: input, at: endpoint)
                guard !Task.isCancelled, requestID == id else { return }
                preview = result
            } catch {
                guard !Task.isCancelled, requestID == id else { return }
                self.error = PlaygroundError.message(for: error)
            }
            requestID = nil; requestTask = nil
        }
    }
    private func cancelRequest() {
        // Cancels this client's wait only, not a shared in-flight prediction.
        requestTask?.cancel(); requestTask = nil; requestID = nil
    }
}
