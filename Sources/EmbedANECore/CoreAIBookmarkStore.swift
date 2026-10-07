import Foundation

public protocol CoreAIBookmarkStoring: Sendable {
    func read(modelID: String, asset: String) throws -> Data?
    func save(_ data: Data, modelID: String, asset: String) throws
}

public struct CoreAIBookmarkStore: CoreAIBookmarkStoring, Sendable {
    public let directory: URL

    public init(directory: URL = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".embed-ane/coreai-bookmarks")) {
        self.directory = directory
    }

    public func read(modelID: String, asset: String) throws -> Data? {
        let storage = StateFileStorage(directory: directory)
        return try storage.read(fileName(modelID: modelID, asset: asset), maximumBytes: 16 * 1024 * 1024)
    }

    public func save(_ data: Data, modelID: String, asset: String) throws {
        let storage = StateFileStorage(directory: directory)
        _ = try storage.write(fileName(modelID: modelID, asset: asset), data: data, replace: true)
    }

    private func fileName(modelID: String, asset: String) -> String {
        "\(modelID)-\(asset).bookmark"
    }
}
