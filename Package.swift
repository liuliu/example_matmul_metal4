// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "matmul",
    platforms: [
        .macOS(.v26)
    ],
    targets: [
        .target(
            name: "ConvolutionShared",
            path: "Sources/ConvolutionShared"
        ),
        .executableTarget(
            name: "matmul",
            path: "Sources/matmul",
            exclude: ["default.metallib", "package.sh"],
            resources: [.process("shader.metal")]
        ),
        .executableTarget(
            name: "attention",
            path: "Sources/attention"
        ),
        .executableTarget(
            name: "convolution",
            dependencies: ["ConvolutionShared"],
            path: "Sources/convolution"
        ),
        .executableTarget(
            name: "convolution3d",
            path: "Sources/convolution3d"
        ),
        .executableTarget(
            name: "convolution3d_padding",
            path: "Sources/convolution3d_padding"
        ),
        .executableTarget(
            name: "convolution2d_padding",
            path: "Sources/convolution2d_padding"
        ),
    ]
)
