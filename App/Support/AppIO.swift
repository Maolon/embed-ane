import Dispatch
import Foundation

/// Small App filesystem operations and catalogue enumeration stay off MainActor.
/// This queue never owns or calls CoreML; the cascade worker remains unchanged.
public final class AppIO: Sendable {
    private let queue = DispatchQueue(label: "embed-ane.app-files", qos: .utility)
    public init() {}
    public func run<T: Sendable>(_ operation: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do { continuation.resume(returning: try operation()) }
                catch { continuation.resume(throwing: error) }
            }
        }
    }
}
