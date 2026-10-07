import Darwin
import Foundation

/// Advisory single-instance lock for the menu bar supervisor process.
/// Prevents multiple instances from colliding on HTTP port 15535 and worker port 15536.
public enum AppInstanceLock {
    nonisolated(unsafe) private static var lockFD: Int32 = -1

    @discardableResult
    public static func acquire(
        home: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".embed-ane")
    ) -> Bool {
        let lockDir = home.path
        try? FileManager.default.createDirectory(atPath: lockDir, withIntermediateDirectories: true)
        let lockPath = (lockDir as NSString).appendingPathComponent("supervisor.lock")
        let fd = open(lockPath, O_CREAT | O_RDWR, 0o600)
        guard fd >= 0 else { return true }
        if flock(fd, LOCK_EX | LOCK_NB) != 0 {
            close(fd)
            return false
        }
        lockFD = fd
        let pidString = "\(ProcessInfo.processInfo.processIdentifier)\n"
        ftruncate(fd, 0)
        _ = pidString.withCString { ptr in
            write(fd, ptr, strlen(ptr))
        }
        return true
    }

    public static func release() {
        if lockFD >= 0 {
            flock(lockFD, LOCK_UN)
            close(lockFD)
            lockFD = -1
        }
    }
}
