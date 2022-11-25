// swift-tools-version: 5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

// To declare other packages that this package depends on.
let packageDependencies: [Package.Dependency] = [
  .package(
    url: "https://github.com/grpc/grpc-swift.git",
    from: "1.10.0"
  ),
  .package(
    url: "https://github.com/apple/swift-nio.git",
    from: "2.41.1"
  ),
  .package(
    url: "https://github.com/apple/swift-protobuf.git",
    from: "1.20.1"
  ),
]

// Defines dependencies for our targets.
extension Target.Dependency {
  static let vtsServiceModel: Self = .target(name: "VtsServiceModel")

  static let grpc: Self = .product(name: "GRPC", package: "grpc-swift")
  static let nio: Self = .product(name: "NIO", package: "swift-nio")
  static let nioCore: Self = .product(name: "NIOCore", package: "swift-nio")
  static let nioPosix: Self = .product(name: "NIOPosix", package: "swift-nio")
  static let protobuf: Self = .product(name: "SwiftProtobuf", package: "swift-protobuf")
}

// Targets are the basic building blocks of a package. A target can define a module or a test suite.
// Targets can depend on other targets in this package, and on products in packages this package depends on.
extension Target {
  static let vtsServiceModel: Target = .target(
    name: "VtsServiceModel",
    dependencies: [
      .grpc,
      .nio,
      .protobuf,
    ],
    path: "Sources/vtsserving/grpc/v1alpha1"
  )

  static let vtsServiceClient: Target = .executableTarget(
    name: "VtsServiceClient",
    dependencies: [
      .grpc,
      .vtsServiceModel,
      .nioCore,
      .nioPosix,
    ],
    path: "Sources/VtsServiceClient"
  )
}

let package = Package(
  name: "iris-swift-client",
  dependencies: packageDependencies,
  targets: [.vtsServiceModel, .vtsServiceClient]
)
