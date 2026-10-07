import EmbedANECore
import Foundation

struct CLIUsageError: Error, Sendable, CustomStringConvertible {
    let description: String
    init(_ description: String) { self.description = description }
}

enum CLICommand: String, Sendable, CaseIterable {
    case fetch, verify, verifyTokenizer = "verify-tokenizer", verifyVision = "verify-vision", serve, status, bench, help
}
struct CLIInvocation: Sendable {
    let command: CLICommand
    let operand: String?
    let overrides: ConfigurationOverrides
    let offline: Bool
    let replace: Bool
    let assets: String?
    let reference: String?
    let outputDirectory: String?
    var vision: VerifyVisionOptions? = nil

    static func parse(_ arguments: [String]) throws -> Self {
        if arguments.isEmpty || arguments == ["--help"] || arguments == ["-h"] || arguments == ["help"] {
            return .init(command: .help, operand: nil, overrides: .init(), offline: false, replace: false,
                         assets: nil, reference: nil, outputDirectory: nil)
        }
        guard let first = arguments.first, let command = CLICommand(rawValue: first), command != .help else {
            throw CLIUsageError("Unknown command. Use --help.")
        }
        if arguments.dropFirst().contains("--help") || arguments.dropFirst().contains("-h") {
            return try parse(["--help"])
        }
        if command == .verifyVision { return try VerifyVisionOptions.parse(Array(arguments.dropFirst())) }
        var overrides = ConfigurationOverrides()
        var positionals: [String] = []
        var flags = Set<String>()
        var offline = false; var replace = false
        var assets: String?; var reference: String?; var outputDirectory: String?
        var index = 1; var positionalOnly = false
        while index < arguments.count {
            let flag = arguments[index]; index += 1
            if flag == "--", !positionalOnly { positionalOnly = true; continue }
            if positionalOnly || !flag.hasPrefix("-") { positionals.append(flag); continue }
            guard flags.insert(flag).inserted else { throw CLIUsageError("Duplicate option \(flag).") }
            func value() throws -> String {
                guard index < arguments.count, !arguments[index].hasPrefix("--"), !arguments[index].isEmpty else {
                    throw CLIUsageError("Missing value for \(flag).")
                }
                let result = arguments[index]; index += 1; return result
            }
            switch flag {
            case "--model-root": overrides.modelRoot = try value()
            case "--model-id":
                guard [.serve, .status, .bench, .verifyTokenizer].contains(command) else {
                    throw CLIUsageError("--model-id is not applicable to \(command.rawValue).")
                }
                overrides.modelID = try value()
            case "--port":
                guard command == .serve || command == .status, let port = Int(try value()), (1...65535).contains(port) else {
                    throw CLIUsageError("--port requires 1...65535 and serve or status.")
                }
                overrides.port = port
            case "--offline":
                guard command == .fetch else { throw CLIUsageError("--offline is a fetch option.") }
                offline = true
            case "--replace":
                guard command == .fetch else { throw CLIUsageError("--replace is a fetch option.") }
                replace = true
            case "--assets":
                guard command == .verifyTokenizer else { throw CLIUsageError("--assets is a verify-tokenizer option.") }
                assets = try value()
            case "--reference":
                guard command == .bench else { throw CLIUsageError("--reference is a bench option.") }
                reference = try value()
            case "--output-dir":
                guard command == .bench else { throw CLIUsageError("--output-dir is a bench option.") }
                outputDirectory = try value()
            default: throw CLIUsageError("Unknown option \(flag).")
            }
        }
        let needsOperand = [.fetch, .verify, .verifyTokenizer, .bench].contains(command)
        guard positionals.count == (needsOperand ? 1 : 0) else {
            throw CLIUsageError("\(command.rawValue) requires \(needsOperand ? "exactly one operand" : "no operands").")
        }
        guard !(offline && replace) else { throw CLIUsageError("--offline and --replace cannot be combined.") }
        if command == .verify, let id = positionals.first {
            do { try ModelSpec.validateID(id) } catch { throw CLIUsageError("Invalid model id.") }
        }
        return .init(command: command, operand: positionals.first, overrides: overrides,
                     offline: offline, replace: replace, assets: assets, reference: reference,
                     outputDirectory: outputDirectory)
    }

    static let help = """
    Usage: embed-ane <command> [options]
      fetch <spec.yaml | org/name> [--offline] [--replace] [--model-root <dir>]
      verify <model-id> [--model-root <dir>]
      verify-tokenizer <golden.jsonl> [--assets <dir>] [--model-root <dir>] [--model-id <id>]
      verify-vision <image-or-video> <text> --position-table <fp32.npy>
            [--tower <mlmodelc>] [--chunks-dir <dir>] [--extrope]
            [--assets <tokenizer-dir>] [--embedding-table <fp16.npy>]
            [--resize smart|origin-bucket] [--reference <vec.npy>]
            [--model-root <dir>] [--model-id <id>]
      serve [--port <1...65535>] [--model-root <dir>] [--model-id <id>]
      status [--port <1...65535>]
      bench <corpus.txt> [--reference <ref.jsonl>] [--output-dir <dir>]
            [--model-root <dir>] [--model-id <id>]
    Configuration: ~/.embed-ane/config.yaml; CLI > EMBED_ANE_MODEL_ROOT > file > defaults.
    verify-tokenizer needs only tokenizer assets, requires >=1000 cases, and records an asset-bound gate receipt.
    bench --reference requires that passing receipt; reference JSONL records are {"text":...,"embedding":[...]}.
    verify-vision is an explicit local-artifact smoke test, not a verified installation or residency audit.
    It always uses extrope; --extrope is optional. --reference requires fp32 NPY [2048] or [1,2048], gate cosine >=0.99.
    Exit codes: 0 success; 1 runtime; 2 verification/parity gate; 3 usage/configuration.
    """
}
