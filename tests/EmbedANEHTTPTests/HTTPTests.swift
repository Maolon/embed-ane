import XCTest
@testable import EmbedANEHTTP

final class HTTPTests: XCTestCase {
    func testBootstrap() { XCTAssertEqual(EmbedANEHTTPInfo.version, "0.1.0") }
}
