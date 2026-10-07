import XCTest
@testable import EmbedANECore

final class CoreTests: XCTestCase {
    func testBootstrap() { XCTAssertEqual(EmbedANECoreInfo.specVersion, "1.1") }
}
