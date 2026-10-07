import CoreML
import Foundation

/// Anticipated device assignments from Apple's public compute-plan API. These
/// are NOT measured runtime ANE residency, utilization, or performance evidence.
public struct ChunkComputePlanAudit: Codable, Sendable {
    public struct Operation: Codable, Sendable {
        public let path: String
        public let name: String
        public let preferred: String?
        public let supported: [String]
    }
    public let chunk: Int
    public let status: String
    public let operations: [Operation]
    public let diagnostic: String?
}

enum ComputePlanAuditor {
    static func inspect(bundle: URL) async -> [ChunkComputePlanAudit] {
        var reports: [ChunkComputePlanAudit] = []
        for chunk in 0..<6 {
            do {
                let configuration = MLModelConfiguration()
                configuration.computeUnits = .cpuAndNeuralEngine
                let plan = try await MLComputePlan.load(
                    contentsOf: bundle.appendingPathComponent("chunks/chunk\(chunk).mlmodelc"),
                    configuration: configuration)
                guard case let .program(program) = plan.modelStructure else {
                    reports.append(.init(chunk: chunk, status: "unavailable", operations: [],
                        diagnostic: "Compute plan has no ML Program structure."))
                    continue
                }
                var records: [ChunkComputePlanAudit.Operation] = []
                func visit(_ block: MLModelStructure.Program.Block, path: String) {
                    for (index, operation) in block.operations.enumerated() {
                        let location = "\(path)/\(index)"
                        let usage = plan.deviceUsage(for: operation)
                        records.append(.init(path: location, name: operation.operatorName,
                            preferred: usage.map { deviceName($0.preferred) },
                            supported: usage?.supported.map(deviceName) ?? []))
                        for (childIndex, child) in operation.blocks.enumerated() {
                            visit(child, path: "\(location)/block\(childIndex)")
                        }
                    }
                }
                for name in program.functions.keys.sorted() {
                    if let function = program.functions[name] { visit(function.block, path: name) }
                }
                let complete = !records.isEmpty && records.allSatisfy { $0.preferred != nil && $0.preferred != "unknown" }
                reports.append(.init(chunk: chunk, status: complete ? "checked" : "partial", operations: records,
                    diagnostic: complete ? nil : "Some anticipated device assignments are unavailable."))
            } catch {
                reports.append(.init(chunk: chunk, status: "unavailable", operations: [], diagnostic: String(describing: error)))
            }
        }
        return reports
    }
    private static func deviceName(_ device: MLComputeDevice) -> String {
        switch device {
        case .cpu: "cpu"
        case .gpu: "gpu"
        case .neuralEngine: "neural_engine"
        @unknown default: "unknown"
        }
    }
}
