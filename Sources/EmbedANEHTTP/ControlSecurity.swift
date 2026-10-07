import EmbedANECore
import Foundation
import Security

public struct ControlToken: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    private let secret: String
    public var description: String { "ControlToken(<redacted>)" }
    public var debugDescription: String { description }
    public var authorizationHeader: String { "Bearer \(secret)" }

    private init(secret: String) throws {
        guard secret.utf8.count == 64,
              secret.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
            throw EmbedANEError.io(path: "control-token", reason: "Expected a 256-bit, lowercase hexadecimal control token.")
        }
        self.secret = secret
    }
    public static func loadOrCreate(in directory: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".embed-ane")) throws -> Self {
        let storage = StateFileStorage(directory: directory)
        if let existing = try storage.read("control-token", maximumBytes: 65, requirePrivate: true) {
            return try decode(existing)
        }
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = bytes.withUnsafeMutableBytes { buffer -> OSStatus in
            guard let base = buffer.baseAddress else { return errSecParam }
            return SecRandomCopyBytes(kSecRandomDefault, buffer.count, base)
        }
        guard status == errSecSuccess else {
            throw EmbedANEError.io(path: "control-token", reason: "System cryptographic random generation failed.")
        }
        let token = try Self(secret: bytes.map { String(format: "%02x", $0) }.joined())
        // Create-only publication prevents concurrent startups replacing each
        // other's token. Readers cannot see a partially written token.
        try storage.write("control-token", data: Data((token.secret + "\n").utf8), replace: false)
        guard let published = try storage.read("control-token", maximumBytes: 65, requirePrivate: true) else {
            throw EmbedANEError.io(path: "control-token", reason: "Control token publication failed.")
        }
        return try decode(published)
    }
    private static func decode(_ data: Data) throws -> Self {
        guard var text = String(data: data, encoding: .utf8) else {
            throw EmbedANEError.io(path: "control-token", reason: "Invalid control token encoding.")
        }
        if text.hasSuffix("\n") { text.removeLast() }
        return try Self(secret: text)
    }
    func accepts(_ header: String) -> Bool {
        let parts = header.split(separator: " ", omittingEmptySubsequences: false)
        guard parts.count == 2, parts[0].lowercased() == "bearer" else { return false }
        let supplied = Array(parts[1].utf8), expected = Array(secret.utf8)
        guard supplied.count == expected.count else { return false }
        var difference: UInt8 = 0
        for index in expected.indices { difference |= supplied[index] ^ expected[index] }
        return difference == 0
    }
}

public struct ControlAccessPolicy: Sendable {
    public struct Authority: Equatable {
        public let host: String
        public let port: Int?
        public init(host: String, port: Int?) {
            self.host = host
            self.port = port
        }
    }
    public let token: ControlToken

    public init(token: ControlToken) {
        self.token = token
    }

    public static func authority(_ value: String) -> Authority? {
        let pieces = value.split(separator: ":", omittingEmptySubsequences: false)
        guard (1...2).contains(pieces.count) else { return nil }
        let host = pieces[0].lowercased()
        guard host == "127.0.0.1" || host == "localhost" else { return nil }
        if pieces.count == 1 { return Authority(host: host, port: nil) }
        guard !pieces[1].isEmpty, pieces[1].utf8.allSatisfy({ (48...57).contains($0) }),
              let port = Int(pieces[1]), (1...65535).contains(port) else { return nil }
        return Authority(host: host, port: port)
    }
    public static func validateHost(_ host: String?, boundPort: Int) throws -> Authority {
        guard let host, let parsed = authority(host),
              parsed.port == nil || parsed.port == boundPort else { throw EmbedANEError.unauthorized }
        return parsed
    }
    public func authorize(host: Authority, boundPort: Int, origins: [String], authorization: [String]) throws {
        guard authorization.count == 1, token.accepts(authorization[0]), origins.count <= 1 else {
            throw EmbedANEError.unauthorized
        }
        // Native clients deliberately send no Origin. "null" is NOT absence.
        if let origin = origins.first {
            guard origin.hasPrefix("http://"),
                  let source = Self.authority(String(origin.dropFirst(7))),
                  source.host == host.host, (source.port ?? 80) == boundPort else {
                throw EmbedANEError.unauthorized
            }
        }
    }
}
