// swift-tools-version:6.1
import PackageDescription

let package = Package(
    name: "embed-ane",
    platforms: [.macOS(.v15)],
    products: [
        .library(name: "EmbedANECore", targets: ["EmbedANECore"]),
        .library(name: "EmbedANEHTTP", targets: ["EmbedANEHTTP"]),
        .library(name: "EmbedANEDownload", targets: ["EmbedANEDownload"]),
        .library(name: "EmbedANEAppSupport", targets: ["EmbedANEAppSupport"]),
        .executable(name: "embed-ane", targets: ["embed-ane"]),
    ],
    dependencies: [
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", exact: "2.26.0"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", exact: "1.3.4"),
        .package(url: "https://github.com/jpsim/Yams.git", exact: "6.2.2"),
    ],
    targets: [
        .target(name: "EmbedANECore", dependencies: [
            .product(name: "Tokenizers", package: "swift-transformers"),
            .product(name: "Yams", package: "Yams"),
        ]),
        .target(name: "EmbedANEHTTP", dependencies: [
            "EmbedANECore", .product(name: "Hummingbird", package: "hummingbird"),
        ]),
        .target(name: "EmbedANEDownload", dependencies: [
            "EmbedANECore", .product(name: "Yams", package: "Yams"),
        ]),
        .target(name: "EmbedANEAppSupport", dependencies: ["EmbedANECore", "EmbedANEHTTP", "EmbedANEDownload"], path: "App/Support"),
        .executableTarget(name: "embed-ane", dependencies: [
            "EmbedANECore", "EmbedANEHTTP", "EmbedANEDownload",
        ]),
        .target(name: "EmbedANETestSupport", dependencies: ["EmbedANECore"], path: "tests/Support"),
        .testTarget(name: "EmbedANECoreTests", dependencies: ["EmbedANECore", "EmbedANETestSupport"], path: "tests/EmbedANECoreTests", resources: [.copy("Fixtures")]),
        .testTarget(name: "EmbedANEHTTPTests", dependencies: ["EmbedANEHTTP", "EmbedANETestSupport"], path: "tests/EmbedANEHTTPTests", resources: [.copy("Fixtures")]),
        .testTarget(name: "EmbedANEDownloadTests", dependencies: ["EmbedANEDownload"], path: "tests/EmbedANEDownloadTests"),
        .testTarget(name: "EmbedANEAppSupportTests", dependencies: ["EmbedANEAppSupport", "EmbedANECore", "EmbedANEDownload", "EmbedANETestSupport"], path: "tests/EmbedANEAppSupportTests"),
        .testTarget(name: "EmbedANECLITests", dependencies: ["embed-ane", "EmbedANECore", "EmbedANEHTTP", "EmbedANEDownload", "EmbedANETestSupport"], path: "tests/EmbedANECLITests"),
    ],
    swiftLanguageModes: [.v6]
)
