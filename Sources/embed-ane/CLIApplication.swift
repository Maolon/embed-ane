import Dispatch
import EmbedANECore
import Foundation

struct CLIResult: Sendable {
    let exitCode: Int32
    let stdout: Data
    let stderr: Data
    init(exitCode: Int32 = 0, stdout: Data = Data(), stderr: Data = Data()) {
        self.exitCode = exitCode; self.stdout = stdout; self.stderr = stderr
    }
}
struct CLIContext: Sendable {
    let configuration: ServiceConfiguration
    let store: ConfigurationStore
    let workingDirectory: URL
    let environment: [String: String]
    func path(_ string: String) -> URL {
        let expanded = store.expandedRoot(string)
        if expanded.hasPrefix("/") { return URL(fileURLWithPath: expanded).standardizedFileURL }
        return workingDirectory.appendingPathComponent(expanded).standardizedFileURL
    }
}
protocol CLICommandExecuting: Sendable {
    func execute(_ invocation: CLIInvocation, context: CLIContext) async throws -> CLIResult
}

enum CLIJSON {
    static func encode<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        var data = try encoder.encode(value); data.append(10); return data
    }
}

enum CLIApplication {
    static func run(arguments: [String], store: ConfigurationStore,
                    workingDirectory: URL, environment: [String: String],
                    executor: any CLICommandExecuting) async -> CLIResult {
        let invocation: CLIInvocation
        do { invocation = try CLIInvocation.parse(arguments) }
        catch { return failure(error, exitCode: 3) }
        if invocation.command == .help { return .init(stdout: Data((CLIInvocation.help + "\n").utf8)) }
        let configuration: ServiceConfiguration
        do { configuration = try store.resolve(cli: invocation.overrides, environment: environment) }
        catch { return failure(error, exitCode: 3) }
        do {
            return try await executor.execute(invocation, context: .init(configuration: configuration,
                store: store, workingDirectory: workingDirectory, environment: environment))
        } catch {
            let code: Int32
            if error is CLIUsageError { code = 3 }
            else if error is CancellationError { code = 1 }
            else if invocation.command == .verify || invocation.command == .verifyTokenizer { code = 2 }
            else if let typed = error as? EmbedANEError {
                switch typed {
                case .invalidRequest, .invalidSpec, .movingBranch, .emptyInput, .batchTooLarge, .inputTooLong,
                     .unsupportedDimension, .unsupportedEncodingFormat: code = 3
                default: code = typed.isVerificationFailure ? 2 : 1
                }
            } else { code = 1 }
            return failure(error, exitCode: code)
        }
    }
    private static func failure(_ error: any Error, exitCode: Int32) -> CLIResult {
        let text = (error as? EmbedANEError)?.message ?? String(describing: error)
        return .init(exitCode: exitCode, stderr: Data(("embed-ane: " + text + "\n").utf8))
    }
}

/// CPU-heavy tokenizer checks, digest work, and local metadata I/O do not run on
/// the cooperative executor. This queue never owns the CoreML model objects.
final class CLIFileWork: Sendable {
    private let queue = DispatchQueue(label: "embed-ane.cli.files", qos: .userInitiated)
    func run<T: Sendable>(_ operation: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do { continuation.resume(returning: try operation()) }
                catch { continuation.resume(throwing: error) }
            }
        }
    }
}
