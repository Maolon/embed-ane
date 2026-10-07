import Foundation
import Testing
@testable import EmbedANECore

private struct CoreAITestTableFixture {
    let root: URL
    let file: URL
    let offset: Int

    init(dictionary: String = "{'descr': '<f2', 'fortran_order': False, 'shape': (248078, 2048), }") throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("coreai-table-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        file = root.appendingPathComponent(MappedEmbeddingTable.filename)
        var body = dictionary
        body += String(repeating: " ", count: (16 - ((10 + body.utf8.count + 1) % 16)) % 16) + "\n"
        var header = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0,
                           UInt8(body.utf8.count & 255), UInt8(body.utf8.count >> 8)])
        header.append(Data(body.utf8))
        offset = header.count
        try header.write(to: file)
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close() }
        try handle.truncate(atOffset: UInt64(offset) + UInt64(ModelABI.tablePayloadBytes))
        for (id, bits) in [(0, UInt16(0x3c00)), (7, UInt16(0x3555)), (248077, UInt16(0x4000))] {
            try handle.seek(toOffset: UInt64(offset + id * 4096))
            var row = Data()
            for _ in 0..<2048 { row.append(UInt8(bits & 255)); row.append(UInt8(bits >> 8)) }
            try handle.write(contentsOf: row)
        }
    }

    func remove() { try? FileManager.default.removeItem(at: root) }
}

private final class MockCoreAIFunction: CoreAIFunction, @unchecked Sendable {
    let name: String
    let chunkIndex: Int
    let inputDescriptions: [String: CoreAITensorDescription]
    let outputDescriptions: [String: CoreAITensorDescription]
    var recordedInputs: [[String: CoreAITensor]] = []

    init(chunkIndex: Int) {
        self.chunkIndex = chunkIndex
        self.name = "chunk\(chunkIndex)"
        var inputs = [
            "hidden_in": CoreAIABI.hidden,
            "cos": CoreAIABI.rotary,
            "sin": CoreAIABI.rotary
        ]
        if chunkIndex == 5 {
            inputs["attention_mask"] = CoreAIABI.mask
        }
        self.inputDescriptions = inputs
        self.outputDescriptions = [
            chunkIndex == 5 ? "embedding" : "hidden_out": chunkIndex == 5 ? CoreAIABI.embedding : CoreAIABI.hidden
        ]
    }

    func run(inputs: [String: CoreAITensor]) async throws -> [String: CoreAITensor] {
        recordedInputs.append(inputs)
        if chunkIndex == 5 {
            // Deliberately return non-unit norm values: [3.0, 4.0, ...]
            var emb = [Float](repeating: 0, count: 2048)
            emb[0] = 3.0
            emb[1] = 4.0
            return ["embedding": CoreAITensor(shape: [1, 2048], fp16: emb.map { Float16($0) })]
        } else {
            return ["hidden_out": CoreAITensor(shape: [1, 512, 2048], fp16: Array(repeating: Float16(chunkIndex + 1), count: 512 * 2048))]
        }
    }
}

private final class MockCoreAIModelHandle: CoreAIModelHandle, @unchecked Sendable {
    let functionNames: [String]
    let bookmarkData: Data?
    let function: any CoreAIFunction

    init(name: String, function: any CoreAIFunction, bookmarkData: Data? = nil) {
        self.functionNames = [name]
        self.function = function
        self.bookmarkData = bookmarkData
    }

    func loadFunction(named name: String) throws -> (any CoreAIFunction)? {
        guard functionNames.contains(name) else { return nil }
        return function
    }
}

private final class MockCoreAILoader: CoreAIAssetLoading, @unchecked Sendable {
    enum LoadSource { case bookmark, cache, specialize }
    private let lock = NSLock()
    private var _loadHistory: [(assetName: String, source: LoadSource)] = []
    var loadHistory: [(assetName: String, source: LoadSource)] {
        get { lock.withLock { _loadHistory } }
        set { lock.withLock { _loadHistory = newValue } }
    }
    var simulateBookmarkCorrupt = false
    var simulateCacheHit = false
    private var _functions: [Int: MockCoreAIFunction] = [:]
    var functions: [Int: MockCoreAIFunction] {
        get { lock.withLock { _functions } }
        set { lock.withLock { _functions = newValue } }
    }

    func loadModel(
        assetURL: URL,
        expectedFunction: String,
        modelID: String,
        assetName: String,
        options: CoreAISpecializationOptions,
        bookmarkStore: any CoreAIBookmarkStoring
    ) async throws -> CoreAILoadedAsset {
        let chunkIndex = Int(expectedFunction.replacingOccurrences(of: "chunk", with: "")) ?? 0
        let fn = MockCoreAIFunction(chunkIndex: chunkIndex)
        lock.withLock { _functions[chunkIndex] = fn }

        // 1. Try bookmark
        if let bookmarkData = try? bookmarkStore.read(modelID: modelID, asset: assetName), !simulateBookmarkCorrupt {
            lock.withLock { _loadHistory.append((assetName, .bookmark)) }
            let handle = MockCoreAIModelHandle(name: expectedFunction, function: fn, bookmarkData: bookmarkData)
            return CoreAILoadedAsset(model: handle, function: fn)
        }

        // 2. Cache probe
        if simulateCacheHit {
            lock.withLock { _loadHistory.append((assetName, .cache)) }
            let bData = Data("cache-bookmark-\(assetName)".utf8)
            try? bookmarkStore.save(bData, modelID: modelID, asset: assetName)
            let handle = MockCoreAIModelHandle(name: expectedFunction, function: fn, bookmarkData: bData)
            return CoreAILoadedAsset(model: handle, function: fn)
        }

        // 3. Specialize
        lock.withLock { _loadHistory.append((assetName, .specialize)) }
        let bData = Data("specialized-bookmark-\(assetName)".utf8)
        try? bookmarkStore.save(bData, modelID: modelID, asset: assetName)
        let handle = MockCoreAIModelHandle(name: expectedFunction, function: fn, bookmarkData: bData)
        return CoreAILoadedAsset(model: handle, function: fn)
    }
}

/// `CoreAIAsyncBridge.runSync` blocks its caller on a semaphore while a Task
/// runs the async CoreAI call. Production calls it only on the serial worker's
/// own thread; tests do the same, or a small cooperative pool (CI runners)
/// deadlocks.
private func onDedicatedThread(_ body: @escaping @Sendable () throws -> Void) async throws {
    try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
        Thread { continuation.resume(with: Result { try body() }) }.start()
    }
}

@Suite("CoreAI Cascade Backend Contracts")
struct CoreAICascadeBackendTests {
    @Test func prepareThrowsWhenUnloaded() async throws {
        try await onDedicatedThread {
            let fixture = try CoreAITestTableFixture(); defer { fixture.remove() }
            let table = try MappedEmbeddingTable(url: fixture.file)
            let tokenizer = VisionFixtureTokenizer(ids: [7])
            let loader = MockCoreAILoader()
            let bookmarkDir = FileManager.default.temporaryDirectory.appendingPathComponent("bm-\(UUID())")
            defer { try? FileManager.default.removeItem(at: bookmarkDir) }
            let bookmarkStore = CoreAIBookmarkStore(directory: bookmarkDir)

            let backend = CoreAICascadeBackend(
                bundle: fixture.root,
                modelID: "test-model",
                loader: loader,
                bookmarkStore: bookmarkStore,
                injectedTokenizer: tokenizer,
                injectedTable: table
            )

            // Must throw modelNotLoaded before load()
            #expect(throws: EmbedANEError.self) {
                try backend.prepare(["hello"])
            }

            let report = try backend.load()
            #expect(report.perChunkNS.count == 6)

            // Now prepare succeeds
            let request = try backend.prepare(["hello"])
            #expect(request.inputs.count == 1)

            // Unload
            _ = try backend.unload()

            // Must throw modelNotLoaded again after unload
            #expect(throws: EmbedANEError.self) {
                try backend.prepare(["hello"])
            }
            #expect(throws: EmbedANEError.self) {
                try backend.predict(request)
            }
        }
    }

    @Test func loadSequenceBookmarkCacheSpecialize() async throws {
        try await onDedicatedThread {
            let fixture = try CoreAITestTableFixture(); defer { fixture.remove() }
            let table = try MappedEmbeddingTable(url: fixture.file)
            let tokenizer = VisionFixtureTokenizer(ids: [7])
            let bookmarkDir = FileManager.default.temporaryDirectory.appendingPathComponent("bm-\(UUID())")
            defer { try? FileManager.default.removeItem(at: bookmarkDir) }
            let bookmarkStore = CoreAIBookmarkStore(directory: bookmarkDir)

            // First pass: clean store -> must specialize all 6 chunks
            let loader1 = MockCoreAILoader()
            let backend1 = CoreAICascadeBackend(
                bundle: fixture.root,
                modelID: "test-model",
                loader: loader1,
                bookmarkStore: bookmarkStore,
                injectedTokenizer: tokenizer,
                injectedTable: table
            )
            _ = try backend1.load()
            #expect(loader1.loadHistory.count == 6)
            #expect(loader1.loadHistory.allSatisfy { $0.source == .specialize })

            // Second pass: same store has bookmarks -> must hit bookmark for all 6 chunks
            let loader2 = MockCoreAILoader()
            let backend2 = CoreAICascadeBackend(
                bundle: fixture.root,
                modelID: "test-model",
                loader: loader2,
                bookmarkStore: bookmarkStore,
                injectedTokenizer: tokenizer,
                injectedTable: table
            )
            _ = try backend2.load()
            #expect(loader2.loadHistory.count == 6)
            #expect(loader2.loadHistory.allSatisfy { $0.source == .bookmark })

            // Third pass: corrupt bookmark -> falls back to cache hit
            let loader3 = MockCoreAILoader()
            loader3.simulateBookmarkCorrupt = true
            loader3.simulateCacheHit = true
            let backend3 = CoreAICascadeBackend(
                bundle: fixture.root,
                modelID: "test-model",
                loader: loader3,
                bookmarkStore: bookmarkStore,
                injectedTokenizer: tokenizer,
                injectedTable: table
            )
            _ = try backend3.load()
            #expect(loader3.loadHistory.count == 6)
            #expect(loader3.loadHistory.allSatisfy { $0.source == .cache })
        }
    }

    @Test func predictFeedsExactTensorNamesAndOutputsUnnormalized() async throws {
        try await onDedicatedThread {
            let fixture = try CoreAITestTableFixture(); defer { fixture.remove() }
            let table = try MappedEmbeddingTable(url: fixture.file)
            let tokenizer = VisionFixtureTokenizer(ids: [7, 8])
            let loader = MockCoreAILoader()
            let bookmarkDir = FileManager.default.temporaryDirectory.appendingPathComponent("bm-\(UUID())")
            defer { try? FileManager.default.removeItem(at: bookmarkDir) }
            let bookmarkStore = CoreAIBookmarkStore(directory: bookmarkDir)

            let backend = CoreAICascadeBackend(
                bundle: fixture.root,
                modelID: "test-model",
                loader: loader,
                bookmarkStore: bookmarkStore,
                injectedTokenizer: tokenizer,
                injectedTable: table
            )
            _ = try backend.load()

            let request = try backend.prepare(["two words"])
            let result = try backend.predict(request)

            #expect(result.embeddings.count == 1)
            let emb = result.embeddings[0]
            #expect(emb.count == 2048)
            // Verify output was NOT re-normalized: raw values [3.0, 4.0, ...] preserved!
            #expect(emb[0] == 3.0)
            #expect(emb[1] == 4.0)

            // Check recorded inputs for chunk 0...4
            for c in 0..<5 {
                let fn = loader.functions[c]!
                #expect(fn.recordedInputs.count == 1)
                let inputs = fn.recordedInputs[0]
                #expect(inputs["hidden_in"] != nil)
                #expect(inputs["cos"] != nil)
                #expect(inputs["sin"] != nil)
                #expect(inputs["attention_mask"] == nil)
            }

            // Check recorded inputs for chunk 5
            let fn5 = loader.functions[5]!
            #expect(fn5.recordedInputs.count == 1)
            let inputs5 = fn5.recordedInputs[0]
            #expect(inputs5["hidden_in"] != nil)
            #expect(inputs5["cos"] != nil)
            #expect(inputs5["sin"] != nil)
            #expect(inputs5["attention_mask"] != nil)
            #expect(inputs5["attention_mask"]?.shape == [1, 512])
        }
    }
}
