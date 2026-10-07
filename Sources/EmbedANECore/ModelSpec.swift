import Foundation
import Yams

public enum StrictCoding {
    private struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    public static func rejectUnknown(_ decoder: any Decoder, allowed: [String]) throws {
        let container = try decoder.container(keyedBy: Key.self)
        let unknown = Set(container.allKeys.map(\.stringValue)).subtracting(allowed)
        guard unknown.isEmpty else {
            throw EmbedANEError.invalidSpec("unknown keys at \(decoder.codingPath.map(\.stringValue).joined(separator: ".")): \(unknown.sorted().joined(separator: ", "))")
        }
    }
}

public enum StrictYAML {
    public static func decode<T: Decodable>(_ type: T.Type, from text: String) throws -> T {
        guard text.utf8.count <= 8 * 1_024 * 1_024 else { throw EmbedANEError.invalidSpec("YAML exceeds 8 MiB") }
        let composed: Node?
        do { composed = try Yams.compose(yaml: text) }
        catch let error as EmbedANEError { throw error }
        catch { throw EmbedANEError.invalidSpec(String(describing: error)) }
        guard let node = composed else { throw EmbedANEError.invalidSpec("empty YAML") }
        var count = 0
        try inspect(node, depth: 0, count: &count)
        do { return try YAMLDecoder().decode(type, from: text) }
        catch let error as EmbedANEError { throw error }
        catch { throw EmbedANEError.invalidSpec(String(describing: error)) }
    }
    private static func inspect(_ node: Node, depth: Int, count: inout Int) throws {
        count += 1
        guard depth <= 32, count <= 100_000 else { throw EmbedANEError.invalidSpec("YAML nesting/alias expansion limit") }
        if let mapping = node.mapping {
            var keys = Set<String>()
            for (key, value) in mapping {
                guard let string = key.string, keys.insert(string).inserted else {
                    throw EmbedANEError.invalidSpec("non-string or duplicate YAML key")
                }
                try inspect(value, depth: depth + 1, count: &count)
            }
        } else if let sequence = node.sequence {
            for value in sequence { try inspect(value, depth: depth + 1, count: &count) }
        }
    }
}

public enum Revision: Sendable, Equatable, Codable {
    case commit(String)
    case tag(String)
    public var value: String { switch self { case let .commit(value), let .tag(value): value } }
    public init(_ value: String) throws {
        guard !value.isEmpty, value.utf8.count <= 256, !value.contains(".."),
              !value.hasPrefix("/"), !value.hasSuffix("/"),
              value.unicodeScalars.allSatisfy({ !CharacterSet.whitespacesAndNewlines.contains($0) && !CharacterSet.controlCharacters.contains($0) }),
              !value.contains("\\"), !value.contains("?"), !value.contains("#"), !value.contains("%") else {
            throw EmbedANEError.invalidSpec("unsafe revision")
        }
        if Self.isCommit(value) { self = .commit(value.lowercased()) }
        else { self = .tag(value) }
    }
    public static func isCommit(_ value: String) -> Bool {
        value.utf8.count == 40 && value.utf8.allSatisfy { (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0) }
    }
    public init(from decoder: any Swift.Decoder) throws { try self.init(decoder.singleValueContainer().decode(String.self)) }
    public func encode(to encoder: any Swift.Encoder) throws { var c = encoder.singleValueContainer(); try c.encode(value) }
}

public struct ArtifactFile: Codable, Sendable, Equatable {
    public let path: String
    public let sha256: String
    public let size: Int64
    enum CodingKeys: String, CodingKey { case path, sha256, size }
    public init(path: String, sha256: String, size: Int64) throws {
        try SafePath.validate(path)
        guard Self.isDigest(sha256), size >= 0 else { throw EmbedANEError.invalidSpec("bad digest or size for \(path)") }
        self.path = path; self.sha256 = sha256.lowercased(); self.size = size
    }
    public static func isDigest(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0) }
    }
    public init(from decoder: any Swift.Decoder) throws {
        try StrictCoding.rejectUnknown(decoder, allowed: ["path", "sha256", "size"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(path: c.decode(String.self, forKey: .path), sha256: c.decode(String.self, forKey: .sha256), size: c.decode(Int64.self, forKey: .size))
    }
}

public struct ModelSpec: Codable, Sendable, Equatable {
    public struct Model: Codable, Sendable, Equatable {
        public let id: String
        public let dim: Int
        public let maxSeq: Int
        public let normalize: String
        enum CodingKeys: String, CodingKey { case id, dim, maxSeq = "max_seq", normalize }
        public init(from decoder: any Swift.Decoder) throws {
            try StrictCoding.rejectUnknown(decoder, allowed: ["id", "dim", "max_seq", "normalize"])
            let c = try decoder.container(keyedBy: CodingKeys.self)
            id = try c.decode(String.self, forKey: .id); dim = try c.decode(Int.self, forKey: .dim)
            maxSeq = try c.decode(Int.self, forKey: .maxSeq); normalize = try c.decode(String.self, forKey: .normalize)
            try ModelSpec.validateID(id)
            guard dim == ModelABI.dimension, maxSeq == ModelABI.sequenceLength, normalize == "l2" else {
                throw EmbedANEError.invalidSpec("model dimensions, sequence length or normalization do not match the frozen ABI")
            }
        }
    }
    public struct Source: Codable, Sendable, Equatable {
        public let repo: String
        public let revision: Revision
        public let endpoint: String
        public let auth: String?
        enum CodingKeys: String, CodingKey { case repo, revision, endpoint, auth }
        public init(from decoder: any Swift.Decoder) throws {
            try StrictCoding.rejectUnknown(decoder, allowed: ["repo", "revision", "endpoint", "auth"])
            let c = try decoder.container(keyedBy: CodingKeys.self)
            repo = try c.decode(String.self, forKey: .repo); revision = try c.decode(Revision.self, forKey: .revision)
            endpoint = try c.decodeIfPresent(String.self, forKey: .endpoint) ?? "https://huggingface.co"
            auth = try c.decodeIfPresent(String.self, forKey: .auth)
            let parts = repo.split(separator: "/", omittingEmptySubsequences: false)
            guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && $0.utf8.allSatisfy { (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45, 46, 95].contains($0) } }) else {
                throw EmbedANEError.invalidSpec("source.repo must be org/name")
            }
            guard let url = URLComponents(string: endpoint), let scheme = url.scheme, let host = url.host,
                  !host.isEmpty, url.user == nil, url.password == nil, url.query == nil, url.fragment == nil,
                  url.path.isEmpty || url.path == "/",
                  scheme == "https" || (scheme == "http" && host == "127.0.0.1") else {
                throw EmbedANEError.invalidSpec("endpoint must be HTTPS (loopback HTTP is test-only)")
            }
            if let auth {
                guard auth.hasPrefix("env:"), auth.count > 4,
                      auth.dropFirst(4).utf8.allSatisfy({ (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0) || $0 == 95 }) else {
                    throw EmbedANEError.invalidSpec("auth must name an environment variable; literal credentials are forbidden")
                }
            }
        }
    }
    public struct Runtime: Codable, Sendable, Equatable {
        public let computeUnits: String
        public let padSide: String
        public let maskDtype: String
        enum CodingKeys: String, CodingKey { case computeUnits = "compute_units", padSide = "pad_side", maskDtype = "mask_dtype" }
        public init(from decoder: any Swift.Decoder) throws {
            try StrictCoding.rejectUnknown(decoder, allowed: ["compute_units", "pad_side", "mask_dtype"])
            let c = try decoder.container(keyedBy: CodingKeys.self)
            computeUnits = try c.decode(String.self, forKey: .computeUnits)
            padSide = try c.decode(String.self, forKey: .padSide); maskDtype = try c.decode(String.self, forKey: .maskDtype)
            guard computeUnits == "cpu_and_ne", padSide == "right", maskDtype == "fp32" else {
                throw EmbedANEError.invalidSpec("runtime does not match the frozen CoreML ABI")
            }
        }
    }
    public let specVersion: Int
    public let model: Model
    public let source: Source
    public let files: [ArtifactFile]
    public let runtime: Runtime
    enum CodingKeys: String, CodingKey { case specVersion = "spec_version", model, source, files, runtime }
    public init(from decoder: any Swift.Decoder) throws {
        try StrictCoding.rejectUnknown(decoder, allowed: ["spec_version", "model", "source", "files", "runtime"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        specVersion = try c.decode(Int.self, forKey: .specVersion)
        guard specVersion == 1 else { throw EmbedANEError.invalidSpec("spec_version must be 1") }
        model = try c.decode(Model.self, forKey: .model); source = try c.decode(Source.self, forKey: .source)
        files = try c.decode([ArtifactFile].self, forKey: .files); runtime = try c.decode(Runtime.self, forKey: .runtime)
        guard !files.isEmpty else { throw EmbedANEError.invalidSpec("files must not be empty") }
        var canonical = Set<String>()
        let reserved: Set<String> = ["manifest.yaml", "RESOLVED", "SPEC_SHA256"]
        for file in files {
            guard !reserved.contains(file.path), !file.path.hasSuffix(".part"), !file.path.hasSuffix(".etag"),
                  canonical.insert(SafePath.canonical(file.path)).inserted else {
                throw EmbedANEError.invalidSpec("reserved or duplicate artifact path: \(file.path)")
            }
        }
        let paths = Set(files.map(\.path))
        let required = ["tokenizer.json", "tokenizer_config.json", "embed_table.fp16.npy"] + (0..<6).map { "chunks/chunk\($0).mlmodelc/manifest.txt" }
        guard Set(required).isSubset(of: paths) else { throw EmbedANEError.invalidSpec("bundle must describe both tokenizer files, the table and all six inner manifests") }
        for path in paths {
            let components = path.split(separator: "/")
            if components.count > 1 {
                for count in 1..<components.count {
                    guard !canonical.contains(SafePath.canonical(components.prefix(count).joined(separator: "/"))) else {
                        throw EmbedANEError.invalidSpec("artifact file/directory collision at \(path)")
                    }
                }
            }
            if path.contains(".mlmodelc/"), !path.hasSuffix("/manifest.txt") {
                throw EmbedANEError.invalidSpec("chunk and model package members must be listed by their inner manifest: \(path)")
            }
            if path.hasPrefix("chunks/"), !required.contains(path) {
                throw EmbedANEError.invalidSpec("chunk members must be listed by their inner manifest: \(path)")
            }
        }
        // A multimodal bundle adds the two-frame tower and its position table;
        // its chunks/ then hold the position-input (extrope) decoder.
        guard paths.contains(Self.visionTowerManifest) == paths.contains(Self.visionPositionTable) else {
            throw EmbedANEError.invalidSpec("a multimodal bundle must include both vision_tower.mlmodelc and vision_abs_pos.fp32.npy")
        }
    }
    public static let visionTowerManifest = "vision_tower.mlmodelc/manifest.txt"
    public static let visionPositionTable = "vision_abs_pos.fp32.npy"
    public var isMultimodal: Bool { files.contains { $0.path == Self.visionTowerManifest } }

    public static func validateID(_ id: String) throws {
        guard (1...64).contains(id.utf8.count), id.utf8.allSatisfy({ (97...122).contains($0) || (48...57).contains($0) || $0 == 45 }) else {
            throw EmbedANEError.invalidSpec("model.id must match [a-z0-9-]{1,64}")
        }
    }
    public static func parse(_ yaml: String) throws -> Self { try StrictYAML.decode(Self.self, from: yaml) }
}
