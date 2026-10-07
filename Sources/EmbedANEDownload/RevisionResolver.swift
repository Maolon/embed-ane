import Foundation
import EmbedANECore

struct HubSource: Sendable {
    let source: ModelSpec.Source
    let token: String?
    let policy: DownloadTransportPolicy

    init(_ source: ModelSpec.Source, environment: [String: String], policy: DownloadTransportPolicy) throws {
        self.source = source
        self.policy = policy
        guard let endpoint = URL(string: source.endpoint) else { throw EmbedANEError.invalidSpec("Invalid endpoint") }
        try policy.validate(endpoint)
        if let auth = source.auth {
            let name = String(auth.dropFirst(4))
            guard let value = environment[name], !value.isEmpty,
                  value.utf8.allSatisfy({ $0 >= 33 && $0 <= 126 }) else {
                throw EmbedANEError.invalidSpec("Missing or invalid authentication environment variable: \(name)")
            }
            token = value
        } else if let hfToken = environment["HF_TOKEN"], !hfToken.isEmpty {
            let endpointLower = source.endpoint.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            let isHF = endpointLower.hasPrefix("https://huggingface.co") || policy == .loopbackHTTPForTests
            if isHF {
                token = hfToken
            } else {
                token = nil
            }
        } else { token = nil }
    }

    private func request(segments: [String]) throws -> URLRequest {
        guard var components = URLComponents(string: source.endpoint) else {
            throw EmbedANEError.invalidSpec("Invalid endpoint")
        }
        // Encode each segment, including a slash inside a revision name. No
        // request uses string concatenation of unescaped remote paths.
        let unreserved = CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~")
        components.percentEncodedPath = "/" + (try segments.map { segment in
            guard let encoded = segment.addingPercentEncoding(withAllowedCharacters: unreserved) else {
                throw EmbedANEError.invalidSpec("Unencodable remote path")
            }
            return encoded
        }).joined(separator: "/")
        guard let url = components.url else { throw EmbedANEError.invalidSpec("Invalid download URL") }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        request.setValue("embed-ane/0.1.0", forHTTPHeaderField: "User-Agent")
        if let token { request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization") }
        return request
    }

    func artifactRequest(commit: String, path: String) throws -> URLRequest {
        try request(segments: source.repo.split(separator: "/").map(String.init)
                    + ["resolve", commit] + path.split(separator: "/").map(String.init))
    }

    func resolve(using transport: any DownloadTransport) async throws -> String {
        let repo = source.repo.split(separator: "/").map(String.init)
        let revision = source.revision.value
        let info: RevisionInfo = try await metadata(
            request(segments: ["api", "models"] + repo + ["revision", revision]),
            using: transport, label: "source.revision"
        )
        guard Revision.isCommit(info.sha) else {
            throw EmbedANEError.invalidSpec("Revision endpoint did not return a 40-hex commit SHA")
        }
        let sha = info.sha.lowercased()
        if case let .commit(expected) = source.revision {
            guard sha == expected else { throw EmbedANEError.conflict("Pinned commit differs from revision response") }
            return sha
        }

        // The revision endpoint's SHA alone cannot distinguish a moving branch
        // from a tag. Consult refs and fail closed on ambiguous/unlisted names.
        let refs: RepositoryRefs = try await metadata(
            request(segments: ["api", "models"] + repo + ["refs"]),
            using: transport, label: "source.revision"
        )
        if let branch = refs.branches.first(where: { $0.name == revision || $0.ref == revision }) {
            throw EmbedANEError.movingBranch(branch.name)
        }
        let tags = refs.tags.filter { $0.name == revision || $0.ref == revision }
        guard tags.count == 1, let tag = tags.first,
              Revision.isCommit(tag.targetCommit), tag.targetCommit.lowercased() == sha else {
            throw EmbedANEError.conflict("Revision is not an unambiguous tag, or changed while resolving")
        }
        return sha
    }

    private struct RevisionInfo: Decodable { let sha: String }
    private struct RepositoryRefs: Decodable {
        struct Ref: Decodable { let name: String; let ref: String; let targetCommit: String }
        let branches: [Ref]
        let tags: [Ref]
    }

    private func metadata<T: Decodable>(_ request: URLRequest, using transport: any DownloadTransport,
                                        label: String) async throws -> T {
        let buffer = MetadataBuffer()
        do {
            try await transport.execute(request, policy: policy, receiveResponse: { response in
                guard response.status == 200 else {
                    throw EmbedANEError.transport(path: label, reason: "HTTP \(response.status)")
                }
            }, receiveData: { try buffer.append($0) })
            return try JSONDecoder().decode(T.self, from: buffer.data)
        } catch let error as EmbedANEError { throw error }
        catch is CancellationError { throw CancellationError() }
        catch {
            // URLSession error descriptions can include signed URLs. Never
            // surface those descriptions or server bodies to logs/CLI/App.
            throw EmbedANEError.transport(path: label, reason: "Metadata transfer or JSON decoding failed")
        }
    }
}

private final class MetadataBuffer: @unchecked Sendable {
    private let lock = NSLock()
    private var storage = Data()
    var data: Data { lock.withLock { storage } }
    func append(_ data: Data) throws {
        try lock.withLock {
            guard data.count <= 8 * 1_024 * 1_024 - storage.count else {
                throw EmbedANEError.transport(path: "source.revision", reason: "Metadata exceeds 8 MiB")
            }
            storage.append(data)
        }
    }
}
