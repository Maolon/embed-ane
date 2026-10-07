import Darwin
import EmbedANECore
import Foundation

struct HostProvenance: Codable, Sendable {
    let hardwareModel: String
    let processor: String
    let architecture: String
    let os: String
    let physicalMemoryBytes: UInt64
    let installedSwiftToolchain: String
    let installedXcode: String
    let compilerFamily: String
    let dependencyPins: [String: String]
    enum CodingKeys: String, CodingKey {
        case hardwareModel = "hardware_model", processor, architecture, os
        case physicalMemoryBytes = "physical_memory_bytes", installedSwiftToolchain = "installed_swift_toolchain"
        case installedXcode = "installed_xcode", compilerFamily = "compiler_family", dependencyPins = "dependency_pins"
    }
    static func capture() -> Self {
        #if arch(arm64)
        let architecture = "arm64"
        #else
        let architecture = "non-arm64 (unsupported)"
        #endif
        #if compiler(>=6.4)
        let compiler = "Swift >=6.4; exact build toolchain is in the build/CI receipt"
        #elseif compiler(>=6.2)
        let compiler = "Swift >=6.2 and <6.4; exact build toolchain is in the build/CI receipt"
        #else
        let compiler = "Swift >=6.1 and <6.2; exact build toolchain is in the build/CI receipt"
        #endif
        return .init(hardwareModel: stringSysctl("hw.model"), processor: stringSysctl("machdep.cpu.brand_string"),
            architecture: architecture, os: ProcessInfo.processInfo.operatingSystemVersionString,
            physicalMemoryBytes: ProcessInfo.processInfo.physicalMemory,
            installedSwiftToolchain: command("/usr/bin/xcrun", ["swift", "--version"]),
            installedXcode: command("/usr/bin/xcodebuild", ["-version"]), compilerFamily: compiler,
            dependencyPins: ["Hummingbird": "2.26.0", "swift-transformers": "1.3.4", "Yams": "6.2.2"])
    }
    private static func stringSysctl(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0, size <= 65_536 else { return "unavailable" }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return "unavailable" }
        return String(decoding: bytes.prefix(while: { $0 != 0 }).map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
    private static func command(_ executable: String, _ arguments: [String]) -> String {
        let process = Process(); let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: executable); process.arguments = arguments
        process.standardOutput = pipe; process.standardError = pipe
        do {
            try process.run()
            let bytes = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard process.terminationStatus == 0, let text = String(data: bytes, encoding: .utf8) else { return "unavailable" }
            return text.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch { return "unavailable" }
    }
}

struct BenchmarkArtifact: Encodable, Sendable {
    struct Load: Codable, Sendable {
        let totalNS: UInt64
        let report: LoadReport
        enum CodingKeys: String, CodingKey { case totalNS = "total_ns", report }
    }
    struct RSS: Encodable, Sendable {
        let scope = "process"
        let beforeLoad: UInt64
        let loaded: UInt64
        let postUnload: UInt64
        let reloaded: UInt64
        let finalUnload: UInt64
        enum CodingKeys: String, CodingKey {
            case scope, beforeLoad = "before_load", loaded, postUnload = "post_unload", reloaded, finalUnload = "final_unload"
        }
    }
    let reportVersion: Int
    let createdAt: String
    let configuration: ServiceConfiguration
    let benchmarkIdleTimeoutS: Double
    let provenance: HostProvenance
    let verification: VerificationReport
    let corpusSHA256: String
    let referenceSHA256: String?
    let tokenizerGate: TokenizerGateReceipt?
    let initialLoad: Load
    let sameProcessReload: Load
    let reloadPrediction: RequestTiming
    let measurements: BenchmarkMeasurements
    let processRSS: RSS
    let computePlans: [ChunkComputePlanAudit]
    let computePlanEvidence = "anticipated device assignments, not measured runtime residency"
    let loadEvidence = "initial process load and same-process reload; disk-cache coldness is not asserted"
    enum CodingKeys: String, CodingKey {
        case reportVersion = "report_version", createdAt = "created_at", configuration
        case benchmarkIdleTimeoutS = "benchmark_idle_timeout_s", provenance, verification
        case corpusSHA256 = "corpus_sha256", referenceSHA256 = "reference_sha256", tokenizerGate = "tokenizer_gate"
        case initialLoad = "initial_load", sameProcessReload = "same_process_reload"
        case reloadPrediction = "reload_prediction", measurements, processRSS = "process_rss", computePlans = "compute_plans"
        case computePlanEvidence = "compute_plan_evidence", loadEvidence = "load_evidence"
    }
}

struct BenchmarkSummary: Codable, Sendable {
    let report: String
    let samples: Int
    let p50NS: UInt64
    let p95NS: UInt64
    let minCos: Double?
    let meanCos: Double?
    let parityGatePassed: Bool?
    enum CodingKeys: String, CodingKey {
        case report, samples, p50NS = "p50_ns", p95NS = "p95_ns", minCos = "min_cos", meanCos = "mean_cos"
        case parityGatePassed = "parity_gate_passed"
    }
}
