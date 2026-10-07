import EmbedANECore
import Foundation

struct TokenizerGateReceipt: Codable, Sendable {
    let receiptVersion: Int
    let createdAt: String
    let goldenSHA256: String
    let report: TokenizerVerificationReport
    enum CodingKeys: String, CodingKey {
        case receiptVersion = "receipt_version", createdAt = "created_at", goldenSHA256 = "golden_sha256", report
    }
    func validate(artifactDigests: [String: String]) throws {
        let names = ["tokenizer.json", "tokenizer_config.json"]
        guard receiptVersion == 1, ArtifactFile.isDigest(goldenSHA256), report.gatePassed,
              report.minimumCases >= 1000, report.cases >= report.minimumCases,
              report.mismatches == 0, report.assetDigests.count == 2,
              names.allSatisfy({ name in
                  guard let digest = artifactDigests[name] else { return false }
                  return ArtifactFile.isDigest(digest) && report.assetDigests[name] == digest
              }) else {
            throw EmbedANEError.verification(path: "tokenizer-gate.json",
                reason: "Tokenizer gate is missing, failed, or does not match this bundle's tokenizer assets.")
        }
    }
    static func read(store: ConfigurationStore) throws -> Self {
        guard let data = try StateFileStorage(directory: store.home.appendingPathComponent(".embed-ane"))
            .read("tokenizer-gate.json", maximumBytes: 1_048_576, requirePrivate: true) else {
            throw EmbedANEError.verification(path: "tokenizer-gate.json", reason: "Run verify-tokenizer before bench --reference.")
        }
        do { return try JSONDecoder().decode(Self.self, from: data) }
        catch { throw EmbedANEError.verification(path: "tokenizer-gate.json", reason: "Malformed tokenizer gate receipt.") }
    }
}
