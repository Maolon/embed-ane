import EmbedANEAppSupport
import EmbedANECore
import Foundation
import Testing

private func writeFile(_ url: URL, bytes: Int) throws {
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Data(repeating: 0xA5, count: bytes).write(to: url)
}

@Suite("E5 specialization cache health")
struct E5CacheHealthTests {
    @Test func warmThreshold() {
        #expect(E5CacheHealth(cachedBytes: 800, modelBytes: 1000).isWarm)
        #expect(!E5CacheHealth(cachedBytes: 799, modelBytes: 1000).isWarm)
        #expect(!E5CacheHealth(cachedBytes: 0, modelBytes: 1000).isWarm)
        #expect(E5CacheHealth(cachedBytes: 0, modelBytes: 0).isWarm)
    }

    @Test func probeComparesCacheAgainstLoadedChunks() throws {
        let temp = try AppTemporaryDirectory()
        defer { temp.remove() }
        var config = ServiceConfiguration()
        config.modelRoot = temp.url.appendingPathComponent("models").path
        let bundle = temp.url.appendingPathComponent("models/\(config.modelID)/chunks")
        for chunk in 0..<6 {
            try writeFile(bundle.appendingPathComponent("chunk\(chunk).mlmodelc/weights/weight.bin"), bytes: 64 * 1024)
        }
        let home = temp.url.appendingPathComponent("home")

        let empty = E5CacheHealth.probe(configuration: config, bundleIdentifier: "test.embed-ane", home: home)
        #expect(empty.modelBytes >= 6 * 64 * 1024)
        #expect(empty.cachedBytes == 0)
        #expect(!empty.isWarm)

        // The probe reads only the running OS build's cache directory.
        var size = 0
        sysctlbyname("kern.osversion", nil, &size, nil, 0)
        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname("kern.osversion", &buffer, &size, nil, 0)
        let cache = home.appendingPathComponent("Library/Caches/test.embed-ane/com.apple.e5rt.e5bundlecache")
            .appendingPathComponent(String(cString: buffer))
        for chunk in 0..<6 {
            try writeFile(cache.appendingPathComponent("key\(chunk)/opts.bundle/H13S.bundle/main/main_ane/model.hwx"),
                          bytes: 64 * 1024)
        }
        let warm = E5CacheHealth.probe(configuration: config, bundleIdentifier: "test.embed-ane", home: home)
        #expect(warm.isWarm)

        // Purged bundles keep their directories but lose their contents.
        for chunk in 0..<6 {
            try FileManager.default.removeItem(at: cache.appendingPathComponent("key\(chunk)/opts.bundle/H13S.bundle/main/main_ane/model.hwx"))
        }
        let purged = E5CacheHealth.probe(configuration: config, bundleIdentifier: "test.embed-ane", home: home)
        #expect(!purged.isWarm)
    }

    @Test func idleEvictionWaitsForWarmCache() async throws {
        let temp = try AppTemporaryDirectory()
        defer { temp.remove() }
        let store = ConfigurationStore(home: temp.url)
        var config = ServiceConfiguration()
        config.idleTimeoutS = 0.05
        try store.save(config)
        let controller = MockWorkerProcessController()
        let session = WorkerSession(configuration: config, store: store, internalPort: 16000,
                                    workerController: controller, portOverride: 0, customHealthPoller: { _ in true })
        try await session.load()
        #expect(await session.getWorkerState() == .ready)

        await session.setCacheHealthProbe { _ in E5CacheHealth(cachedBytes: 0, modelBytes: 1000) }
        try await Task.sleep(for: .milliseconds(100))
        await session.checkIdleEviction()
        #expect(await session.getWorkerState() == .ready)
        #expect(controller.terminateCount == 0)

        await session.setCacheHealthProbe { _ in E5CacheHealth(cachedBytes: 1000, modelBytes: 1000) }
        try await Task.sleep(for: .milliseconds(100))
        await session.checkIdleEviction()
        #expect(await session.getWorkerState() == .down)
        #expect(controller.terminateCount == 1)
    }
}
