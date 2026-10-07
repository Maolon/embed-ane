import Dispatch
import Foundation

/// A synchronous monotonic reading avoids actor reentrancy between admission
/// checks and queue insertion. Test clocks can wake sleepers deterministically.
public protocol RuntimeClock: Sendable {
    func nowNS() -> UInt64
    func sleep(untilNS deadline: UInt64) async throws
}

public struct MonotonicRuntimeClock: RuntimeClock {
    public init() {}
    public func nowNS() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    public func sleep(untilNS deadline: UInt64) async throws {
        let now = nowNS()
        if deadline > now { try await Task.sleep(nanoseconds: deadline - now) }
        try Task.checkCancellation()
    }
}

func elapsedNS(_ end: UInt64, since start: UInt64) -> UInt64 { end >= start ? end - start : 0 }
func addingNS(_ start: UInt64, _ interval: UInt64) -> UInt64 {
    let (value, overflow) = start.addingReportingOverflow(interval)
    return overflow ? .max : value
}
