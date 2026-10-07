import Darwin
import EmbedANECore
import Foundation

/// Read-only estimate of whether CoreML's on-device specialization cache still
/// holds the models a fresh worker would load.
///
/// The E5 runtime writes one specialized bundle per model under
/// `~/Library/Caches/<bundle id>/com.apple.e5rt.e5bundlecache/<OS build>/`,
/// roughly the size of the compiled model, and marks it purgeable. Under
/// storage pressure macOS reclaims the contents but leaves the directories, so
/// the next load re-specializes every model (tens of minutes on the ANE
/// compiler). The layout is private; this only compares directory sizes.
public struct E5CacheHealth: Sendable, Equatable {
    public let cachedBytes: UInt64
    public let modelBytes: UInt64

    public init(cachedBytes: UInt64, modelBytes: UInt64) {
        self.cachedBytes = cachedBytes
        self.modelBytes = modelBytes
    }

    /// At least 80% of the loaded models' size is present in the cache.
    /// Unknown model size (nothing to load) counts as warm.
    public var isWarm: Bool { modelBytes == 0 || cachedBytes >= modelBytes / 10 * 8 }

    public static func probe(configuration: ServiceConfiguration,
                             bundleIdentifier: String? = Bundle.main.bundleIdentifier,
                             home: URL = FileManager.default.homeDirectoryForCurrentUser) -> E5CacheHealth {
        let cache = cacheDirectory(bundleIdentifier: bundleIdentifier, home: home)
        return E5CacheHealth(cachedBytes: cache.map(directoryBytes) ?? 0,
                             modelBytes: modelURLs(configuration).reduce(0) { $0 + directoryBytes($1) })
    }

    static func cacheDirectory(bundleIdentifier: String?, home: URL) -> URL? {
        guard let id = bundleIdentifier, let build = osBuild() else { return nil }
        return home.appendingPathComponent("Library/Caches", isDirectory: true)
            .appendingPathComponent(id, isDirectory: true)
            .appendingPathComponent("com.apple.e5rt.e5bundlecache", isDirectory: true)
            .appendingPathComponent(build, isDirectory: true)
    }

    /// The CoreML models a worker built from `configuration` loads.
    static func modelURLs(_ configuration: ServiceConfiguration) -> [URL] {
        guard configuration.engineBackend == .coreml else { return [] }
        let bundle = URL(fileURLWithPath: configuration.modelRoot, isDirectory: true)
            .appendingPathComponent(configuration.modelID, isDirectory: true)
        var urls = (0..<6).map { bundle.appendingPathComponent("chunks/chunk\($0).mlmodelc", isDirectory: true) }
        if let vision = try? VisionServingArtifacts.paths(configuration) {
            urls.append(vision.tower)
            urls += vision.chunks
        }
        return urls
    }

    static func osBuild() -> String? {
        var size = 0
        guard sysctlbyname("kern.osversion", nil, &size, nil, 0) == 0, size > 1 else { return nil }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname("kern.osversion", &buffer, &size, nil, 0) == 0 else { return nil }
        return String(cString: buffer)
    }

    /// Allocated bytes (not logical size): purged files may keep their length.
    static func directoryBytes(_ url: URL) -> UInt64 {
        let keys: [URLResourceKey] = [.totalFileAllocatedSizeKey, .isRegularFileKey]
        guard let walker = FileManager.default.enumerator(at: url, includingPropertiesForKeys: keys) else { return 0 }
        var total: UInt64 = 0
        for case let file as URL in walker {
            guard let values = try? file.resourceValues(forKeys: Set(keys)), values.isRegularFile == true else { continue }
            total += UInt64(values.totalFileAllocatedSize ?? 0)
        }
        return total
    }
}
