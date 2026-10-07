import Foundation

public protocol CoreAIFunction: AnyObject, Sendable {
    var name: String { get }
    var inputDescriptions: [String: CoreAITensorDescription] { get }
    var outputDescriptions: [String: CoreAITensorDescription] { get }
    func run(inputs: [String: CoreAITensor]) async throws -> [String: CoreAITensor]
}

public protocol CoreAIModelHandle: AnyObject, Sendable {
    var functionNames: [String] { get }
    var bookmarkData: Data? { get }
    func loadFunction(named name: String) throws -> (any CoreAIFunction)?
}

public struct CoreAILoadedAsset: Sendable {
    public let model: any CoreAIModelHandle
    public let function: any CoreAIFunction
    public init(model: any CoreAIModelHandle, function: any CoreAIFunction) {
        self.model = model
        self.function = function
    }
}

public protocol CoreAIAssetLoading: Sendable {
    func loadModel(
        assetURL: URL,
        expectedFunction: String,
        modelID: String,
        assetName: String,
        options: CoreAISpecializationOptions,
        bookmarkStore: any CoreAIBookmarkStoring
    ) async throws -> CoreAILoadedAsset
}

/// Chunk assets are independent graphs, and each loadFunction materializes
/// several hundred MB of specialized weights. Loading the six chunks
/// sequentially spends the SUM of those mapping times; concurrently it spends
/// roughly the slowest single graph. Order and per-chunk timing are preserved,
/// and every chunk is ABI-validated before the caller installs anything.
public enum CoreAIChunkLoad {
    public static func parallel(
        loader: any CoreAIAssetLoading,
        urlAt: @escaping @Sendable (Int) -> URL,
        modelID: String,
        options: CoreAISpecializationOptions,
        bookmarkStore: any CoreAIBookmarkStoring,
        validate: @escaping @Sendable (Int, CoreAILoadedAsset) throws -> Void
    ) throws -> [(function: any CoreAIFunction, loadNS: UInt64)] {
        try CoreAIAsyncBridge.runSync {
            try await withThrowingTaskGroup(of: (Int, CoreAILoadedAsset, UInt64).self) { group in
                for index in 0..<6 {
                    group.addTask {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let loaded = try await loader.loadModel(
                            assetURL: urlAt(index),
                            expectedFunction: "chunk\(index)",
                            modelID: modelID,
                            assetName: "chunk\(index)",
                            options: options,
                            bookmarkStore: bookmarkStore
                        )
                        return (index, loaded, start)
                    }
                }
                var byIndex: [Int: (CoreAILoadedAsset, UInt64)] = [:]
                for try await (index, loaded, start) in group {
                    try validate(index, loaded)
                    byIndex[index] = (loaded, DispatchTime.now().uptimeNanoseconds - start)
                }
                var ordered: [(function: any CoreAIFunction, loadNS: UInt64)] = []
                ordered.reserveCapacity(6)
                for index in 0..<6 {
                    guard let entry = byIndex[index] else {
                        throw EmbedANEError.abiMismatch(chunk: index, reason: "Chunk load did not complete.")
                    }
                    ordered.append((entry.0.function, entry.1))
                }
                return ordered
            }
        }
    }
}

final class CoreAISyncBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Result<T, any Error>?
    func put(_ res: Result<T, any Error>) {
        lock.withLock { value = res }
    }
    func get() -> Result<T, any Error>? {
        lock.withLock { value }
    }
}

enum CoreAIAsyncBridge {
    static func runSync<T: Sendable>(_ operation: @escaping @Sendable () async throws -> T) throws -> T {
        let box = CoreAISyncBox<T>()
        let sem = DispatchSemaphore(value: 0)
        Task {
            do {
                let val = try await operation()
                box.put(.success(val))
            } catch {
                box.put(.failure(error))
            }
            sem.signal()
        }
        sem.wait()
        guard let res = box.get() else {
            throw EmbedANEError.failed("CoreAI async operation interrupted.")
        }
        return try res.get()
    }
}

public struct UnsupportedCoreAILoader: CoreAIAssetLoading, Sendable {
    public init() {}
    public func loadModel(
        assetURL: URL,
        expectedFunction: String,
        modelID: String,
        assetName: String,
        options: CoreAISpecializationOptions,
        bookmarkStore: any CoreAIBookmarkStoring
    ) async throws -> CoreAILoadedAsset {
        throw EmbedANEError.failed("CoreAI engine requires macOS 27.0 or later.")
    }
}

public func defaultCoreAILoader() -> any CoreAIAssetLoading {
    #if canImport(CoreAI)
    if #available(macOS 27.0, *) {
        return ProductionCoreAIAssetLoader()
    }
    #endif
    return UnsupportedCoreAILoader()
}

#if canImport(CoreAI)
import CoreAI

@available(macOS 27.0, *)
extension CoreAITensorDescription {
    public static func describe(_ descriptor: InferenceFunctionDescriptor) -> (inputs: [String: CoreAITensorDescription], outputs: [String: CoreAITensorDescription]) {
        var ins: [String: CoreAITensorDescription] = [:]
        for name in descriptor.inputNames {
            if let desc = descriptor.inputDescriptor(of: name) {
                switch desc {
                case .ndArray(let nd):
                    let elem: CoreAITensorDescription.Element = nd.scalarType == .float16 ? .float16 : (nd.scalarType == .float32 ? .float32 : .unsupported)
                    ins[name] = CoreAITensorDescription(shape: nd.shape, element: elem)
                case .image:
                    ins[name] = CoreAITensorDescription(shape: [], element: .unsupported)
                @unknown default:
                    ins[name] = CoreAITensorDescription(shape: [], element: .unsupported)
                }
            }
        }
        var outs: [String: CoreAITensorDescription] = [:]
        for name in descriptor.outputNames {
            if let desc = descriptor.outputDescriptor(of: name) {
                switch desc {
                case .ndArray(let nd):
                    let elem: CoreAITensorDescription.Element = nd.scalarType == .float16 ? .float16 : (nd.scalarType == .float32 ? .float32 : .unsupported)
                    outs[name] = CoreAITensorDescription(shape: nd.shape, element: elem)
                case .image:
                    outs[name] = CoreAITensorDescription(shape: [], element: .unsupported)
                @unknown default:
                    outs[name] = CoreAITensorDescription(shape: [], element: .unsupported)
                }
            }
        }
        return (ins, outs)
    }
}

@available(macOS 27.0, *)
public final class ProductionCoreAIFunction: CoreAIFunction, Sendable {
    public let name: String
    private let function: InferenceFunction
    public let inputDescriptions: [String: CoreAITensorDescription]
    public let outputDescriptions: [String: CoreAITensorDescription]

    public init(name: String, function: InferenceFunction, model: AIModel) {
        self.name = name
        self.function = function
        if let desc = model.functionDescriptor(for: name) {
            let described = CoreAITensorDescription.describe(desc)
            self.inputDescriptions = described.inputs
            self.outputDescriptions = described.outputs
        } else {
            self.inputDescriptions = [:]
            self.outputDescriptions = [:]
        }
    }

    private static func extractFP16(from nd: NDArray) -> [Float16]? {
        let view = nd.view(as: Float16.self)
        guard let span = view.contiguousElements else { return nil }
        var vals = [Float16](repeating: 0, count: span.count)
        vals.withUnsafeMutableBufferPointer { dest in
            for i in 0..<span.count { dest[i] = span[i] }
        }
        return vals
    }

    private static func extractFP32(from nd: NDArray) -> [Float]? {
        let view = nd.view(as: Float.self)
        guard let span = view.contiguousElements else { return nil }
        var vals = [Float](repeating: 0, count: span.count)
        vals.withUnsafeMutableBufferPointer { dest in
            for i in 0..<span.count { dest[i] = span[i] }
        }
        return vals
    }

    public func run(inputs: [String: CoreAITensor]) async throws -> [String: CoreAITensor] {
        var ndInputs: [String: NDArray] = [:]
        for (name, tensor) in inputs {
            switch tensor.element {
            case .float16(let fp16):
                ndInputs[name] = NDArray(scalars: fp16, shape: tensor.shape)
            case .float32(let fp32):
                ndInputs[name] = NDArray(scalars: fp32, shape: tensor.shape)
            }
        }
        var outputs = try await function.run(inputs: ndInputs)
        var result: [String: CoreAITensor] = [:]
        let names = Array(outputs.names)
        for name in names {
            if let iv = outputs.remove(name), let outND = iv.ndArray {
                if outND.scalarType == .float16, let vals = Self.extractFP16(from: outND) {
                    result[name] = CoreAITensor(shape: outND.shape, fp16: vals)
                } else if outND.scalarType == .float32, let vals = Self.extractFP32(from: outND) {
                    result[name] = CoreAITensor(shape: outND.shape, fp32: vals)
                }
            }
        }
        return result
    }
}

@available(macOS 27.0, *)
public final class ProductionCoreAIModelHandle: CoreAIModelHandle, Sendable {
    private let model: AIModel

    public init(model: AIModel) {
        self.model = model
    }

    public var functionNames: [String] {
        model.functionNames
    }

    public var bookmarkData: Data? {
        model.bookmarkData
    }

    public func loadFunction(named name: String) throws -> (any CoreAIFunction)? {
        guard let fn = try model.loadFunction(named: name) else { return nil }
        return ProductionCoreAIFunction(name: name, function: fn, model: model)
    }
}

@available(macOS 27.0, *)
public struct ProductionCoreAIAssetLoader: CoreAIAssetLoading, Sendable {
    public init() {}

    public func loadModel(
        assetURL: URL,
        expectedFunction: String,
        modelID: String,
        assetName: String,
        options: CoreAISpecializationOptions,
        bookmarkStore: any CoreAIBookmarkStoring
    ) async throws -> CoreAILoadedAsset {
        let opts: SpecializationOptions = options.preferredUnit == .cpu
            ? .cpuOnly
            : SpecializationOptions(preferredComputeUnitKind: .neuralEngine)

        func trace(_ stage: String, _ asset: String) {
            FileHandle.standardError.write(Data("coreai-load \(asset) \(stage)\n".utf8))
        }

        // 1. Fast respawn: try resolving persisted bookmark first
        if let bookmarkData = try? bookmarkStore.read(modelID: modelID, asset: assetName) {
            do {
                trace("bookmark-begin", assetName)
                if let model = try AIModel(resolvingBookmark: bookmarkData) {
                    if model.functionNames.contains(expectedFunction),
                       let fn = try model.loadFunction(named: expectedFunction) {
                        trace("bookmark-ok", assetName)
                        let handle = ProductionCoreAIModelHandle(model: model)
                        let function = ProductionCoreAIFunction(name: expectedFunction, function: fn, model: model)
                        return CoreAILoadedAsset(model: handle, function: function)
                    }
                    trace("bookmark-unusable", assetName)
                } else {
                    trace("bookmark-nil", assetName)
                }
            } catch {
                trace("bookmark-error", assetName)
                // Ignore corrupt or invalidated bookmark and fall through to cache probe
            }
        } else {
            trace("bookmark-absent", assetName)
        }

        // 2. Cache probe
        do {
            trace("probe-begin", assetName)
            if let probe = try AIModelCache.default.model(for: assetURL, options: opts) {
                if probe.functionNames.contains(expectedFunction),
                   let fn = try probe.loadFunction(named: expectedFunction) {
                    trace("probe-ok", assetName)
                    try? bookmarkStore.save(probe.bookmarkData, modelID: modelID, asset: assetName)
                    let handle = ProductionCoreAIModelHandle(model: probe)
                    let function = ProductionCoreAIFunction(name: expectedFunction, function: fn, model: probe)
                    return CoreAILoadedAsset(model: handle, function: function)
                }
                trace("probe-unusable", assetName)
            } else {
                trace("probe-miss", assetName)
            }
        } catch {
            trace("probe-error", assetName)
            // Fall through to specialize
        }

        // 3. Specialize (heavy AOT/JIT compile).
        // Observed on device: a killed or failed specialization
        // can leave cache records that make both the probe and specialize fail
        // with a misleading "malformed asset / missing hash file" error. Purging
        // the entries for this asset URL and retrying once recovers reliably.
        trace("specialize-begin", assetName)
        let model: AIModel
        do {
            model = try await AIModel.specialize(contentsOf: assetURL, options: opts, cache: .default, cachePolicy: .default)
        } catch {
            trace("specialize-purge-retry", assetName)
            try? AIModelCache.default.deleteEntries(for: assetURL)
            model = try await AIModel.specialize(contentsOf: assetURL, options: opts, cache: .default, cachePolicy: .default)
        }
        trace("specialize-ok", assetName)
        guard let fn = try model.loadFunction(named: expectedFunction) else {
            throw EmbedANEError.failed("CoreAI loadFunction returned nil after specialize for \(expectedFunction).")
        }
        try? bookmarkStore.save(model.bookmarkData, modelID: modelID, asset: assetName)
        let handle = ProductionCoreAIModelHandle(model: model)
        let function = ProductionCoreAIFunction(name: expectedFunction, function: fn, model: model)
        return CoreAILoadedAsset(model: handle, function: function)
    }
}
#endif
