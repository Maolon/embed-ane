import EmbedANECore
import Foundation
import Testing
@testable import embed_ane

struct CLIArgumentTests {
    @Test(arguments: [["fetch", "model.yaml", "--offline"], ["verify", "model"],
                      ["verify-tokenizer", "golden.jsonl", "--assets", "/tmp/assets"],
                      ["serve", "--port", "9999"], ["status"], ["bench", "corpus.txt", "--reference", "ref.jsonl"]])
    func acceptsEveryCommand(_ arguments: [String]) throws {
        #expect(try CLIInvocation.parse(arguments).command.rawValue == arguments[0])
    }
    @Test(arguments: [["unknown"], ["fetch"], ["verify", "../escape"], ["serve", "extra"],
                      ["serve", "--port", "0"], ["serve", "--port", "65536"], ["status", "--port"],
                      ["fetch", "x", "--offline", "--replace"], ["bench", "x", "--offline"],
                      ["status", "--port", "8080", "--port", "8081"], ["verify", "model", "--unknown"],
                      ["verify-tokenizer", "x", "--assets"], ["fetch", "x", "--model-id", "other"]])
    func rejectsInvalidUsage(_ arguments: [String]) {
        #expect(throws: CLIUsageError.self) { try CLIInvocation.parse(arguments) }
    }
    @Test func separatorAndHomeRelativeFlagsArePreserved() throws {
        let command = try CLIInvocation.parse(["bench", "--model-root", "~/models", "--", "-corpus.txt"])
        #expect(command.operand == "-corpus.txt"); #expect(command.overrides.modelRoot == "~/models")
    }
    @Test(arguments: [[], ["--help"], ["serve", "--help"]])
    func helpWorksWithoutModelArguments(_ arguments: [String]) throws {
        #expect(try CLIInvocation.parse(arguments).command == .help)
    }
}

private actor RecordingExecutor: CLICommandExecuting {
    var calls = 0
    var received: CLIContext?
    let failure: EmbedANEError?
    init(failure: EmbedANEError? = nil) { self.failure = failure }
    func execute(_ invocation: CLIInvocation, context: CLIContext) async throws -> CLIResult {
        calls += 1; received = context
        if let failure { throw failure }
        return .init(stdout: Data("{}\n".utf8))
    }
}

struct CLIApplicationTests {
    @Test func resolvesCLIOverEnvironmentOverFileAndExpandsHome() async throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: home) }
        let store = ConfigurationStore(home: home)
        try store.save(.init(port: 8123, modelRoot: "/file/models"))
        let executor = RecordingExecutor()
        let result = await CLIApplication.run(arguments: ["serve", "--model-root", "~/cli-models"],
            store: store, workingDirectory: home, environment: ["EMBED_ANE_MODEL_ROOT": "/env/models"], executor: executor)
        #expect(result.exitCode == 0)
        let context = try #require(await executor.received)
        #expect(context.configuration.modelRoot == home.appendingPathComponent("cli-models").path)
        #expect(context.configuration.port == 8123)
        let second = RecordingExecutor()
        _ = await CLIApplication.run(arguments: ["status"], store: store, workingDirectory: home,
            environment: ["EMBED_ANE_MODEL_ROOT": "/env/models"], executor: second)
        #expect(await second.received?.configuration.modelRoot == "/env/models")
        #expect(try store.load().modelRoot == "/file/models") // Transient flags do not overwrite config.
    }
    @Test func usageAndHelpDoNotInvokeServicesOrReadBadConfiguration() async throws {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: home) }
        let store = ConfigurationStore(home: home)
        try StateFileStorage(directory: store.file.deletingLastPathComponent()).write("config.yaml", data: Data("bad yaml: [".utf8))
        let executor = RecordingExecutor()
        for arguments in [["--help"], ["serve", "--port", "0"]] {
            let result = await CLIApplication.run(arguments: arguments, store: store, workingDirectory: home,
                environment: [:], executor: executor)
            #expect(result.exitCode == (arguments == ["--help"] ? 0 : 3))
        }
        #expect(await executor.calls == 0)
    }
    @Test(arguments: [("serve", EmbedANEError.failed("fixture"), Int32(1)),
                      ("verify", EmbedANEError.invalidSpec("manifest"), Int32(2)),
                      ("fetch", EmbedANEError.invalidSpec("source"), Int32(3)),
                      ("bench", EmbedANEError.verification(path: "reference", reason: "fixture"), Int32(2))])
    func exactExitClassification(_ command: String, _ failure: EmbedANEError, _ code: Int32) async {
        let home = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let arguments = command == "serve" ? [command] : [command, "fixture"]
        let result = await CLIApplication.run(arguments: arguments, store: .init(home: home), workingDirectory: home,
            environment: [:], executor: RecordingExecutor(failure: failure))
        #expect(result.exitCode == code); #expect(result.stdout.isEmpty); #expect(!result.stderr.isEmpty)
    }
}
