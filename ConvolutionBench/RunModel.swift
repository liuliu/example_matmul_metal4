import Foundation
import Combine
import SwiftUI
import UIKit

struct DeviceRunReport: Codable {
  var generatedAt: String
  var deviceName: String
  var deviceModel: String
  var systemVersion: String
  var gpuCaptureSupport: GPUCaptureSupportResult
  var tensorOpLargeSweep: [TensorOpTileProfileResult]?
  var tensorOpConv3DValidation: Convolution3DValidationSuiteResult?
  var tensorOpConv3DLeftRightPaddingValidation: Convolution3DValidationSuiteResult?
  var tensorOpConv3DProfile: TensorOpConv3DProfileResult?
  var tensorOpConv3DBiasValidation: Convolution3DValidationSuiteResult?
  var tensorOpConv3DBiasProfile: TensorOpConv3DProfileResult?
  var tensorOpConv3DPermutationProfile: TensorOpConv3DPermutationProfileResult?
  var tensorOpVariantDebug: TensorOpVariantSuiteResult?
  var validation: ConvolutionValidationSuiteResult?
  var profile: ConvolutionProfileResult?
  var mpsGraphWeightLayouts: MPSGraphWeightLayoutProfileResult?
  var mpsGraphLargeWeightLayouts: MPSGraphWeightLayoutProfileResult?
  var mpsGraphConv3DWeightLayouts: MPSGraphConv3DWeightLayoutProfileResult?
  var mpsGraphLargeConv3DWeightLayouts: MPSGraphConv3DWeightLayoutProfileResult?
  var gpuTraceCapture: GPUTraceCaptureResult?
  var gpuTraceCaptureError: String?
}

struct BackgroundRunResult {
  var logText: String
  var reportPath: String?
}

@MainActor
final class RunModel: ObservableObject {
  @Published var logText = "Waiting to start."
  @Published var isRunning = false
  @Published var reportPath: String?

  private var hasStarted = false

  func start(force: Bool) {
    if isRunning {
      return
    }
    if hasStarted && !force {
      return
    }
    hasStarted = true
    isRunning = true
    logText = "Starting run..."
    reportPath = nil

    let deviceName = UIDevice.current.name
    let deviceModel = UIDevice.current.model
    let systemVersion = UIDevice.current.systemVersion

    Task { [weak self] in
      let result = await Task.detached(priority: .userInitiated) {
        var lines: [String] = []
        var savedReportPath: String?

        func append(_ line: String) {
          lines.append(line)
          print(line)
        }

        do {
          append("Starting Conv3D tensor-op left/right padding validation on \(deviceName)")
          let gpuCaptureSupport = ConvolutionHarness.gpuCaptureSupport()
          append(
            "GPU capture support: gpuTraceDocument=\(gpuCaptureSupport.gpuTraceDocumentSupported), developerTools=\(gpuCaptureSupport.developerToolsSupported)"
          )
          let tensorOpConv3DLeftRightPaddingValidation = Self.runTensorOpConv3DLeftRightPaddingValidation(log: append)

          let report = DeviceRunReport(
            generatedAt: ISO8601DateFormatter().string(from: Date()),
            deviceName: deviceName,
            deviceModel: deviceModel,
            systemVersion: systemVersion,
            gpuCaptureSupport: gpuCaptureSupport,
            tensorOpLargeSweep: nil,
            tensorOpConv3DValidation: nil,
            tensorOpConv3DLeftRightPaddingValidation: tensorOpConv3DLeftRightPaddingValidation,
            tensorOpConv3DProfile: nil,
            tensorOpConv3DBiasValidation: nil,
            tensorOpConv3DBiasProfile: nil,
            tensorOpConv3DPermutationProfile: nil,
            tensorOpVariantDebug: nil,
            validation: nil,
            profile: nil,
            mpsGraphWeightLayouts: nil,
            mpsGraphLargeWeightLayouts: nil,
            mpsGraphConv3DWeightLayouts: nil,
            mpsGraphLargeConv3DWeightLayouts: nil,
            gpuTraceCapture: nil,
            gpuTraceCaptureError: nil
          )
          let reportURL = try Self.writeReport(report)
          savedReportPath = reportURL.path
          append("Saved report to \(reportURL.path)")
          append("Conv3D tensor-op left/right padding validation finished.")
        } catch {
          append("Run failed: \(error)")
        }

        return BackgroundRunResult(
          logText: lines.joined(separator: "\n"),
          reportPath: savedReportPath
        )
      }.value

      self?.logText = result.logText
      self?.reportPath = result.reportPath
      self?.isRunning = false
    }
  }

  nonisolated private static func reportURL() throws -> URL {
    guard let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else {
      throw NSError(domain: "ConvolutionBench", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not locate documents directory"])
    }
    return documentsURL.appendingPathComponent("convolution-device-results.json")
  }

  nonisolated private static func writeReport(_ report: DeviceRunReport) throws -> URL {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(report)
    let reportURL = try reportURL()
    try data.write(to: reportURL, options: .atomic)
    return reportURL
  }

  nonisolated private static func runTensorOpConv3DValidation(
    log: (String) -> Void
  ) -> Convolution3DValidationSuiteResult {
    log("")
    log("Conv3D tensor-op validation")
    let buildOptions = ConvolutionHarness.defaultTensorOpConv3DBuildOptions()
    if let outputTileWidth = buildOptions.outputTileWidth, let outputTileHeight = buildOptions.outputTileHeight {
      log("Build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(outputTileWidth)x\(outputTileHeight)")
    }
    let validation = ConvolutionHarness.validateTensorOpConv3D(log: log)
    for (index, result) in validation.cases.enumerated() {
      log("Case \(index + 1)/\(validation.cases.count): \(result.dimensions)")
      log("Max absolute error: \(result.tensorOpValidation.maxAbsoluteError)")
      log("Mismatches above tolerance: \(result.tensorOpValidation.mismatches)")
    }
    log("All Conv3D tensor-op validation cases passed: \(validation.allPassed)")
    return validation
  }

  nonisolated private static func runTensorOpConv3DBiasValidation(
    log: (String) -> Void
  ) -> Convolution3DValidationSuiteResult {
    log("")
    log("Conv3D bias tensor-op validation")
    let buildOptions = ConvolutionHarness.defaultTensorOpConv3DBuildOptions()
    if let outputTileWidth = buildOptions.outputTileWidth, let outputTileHeight = buildOptions.outputTileHeight {
      log("Build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(outputTileWidth)x\(outputTileHeight)")
    }
    let validation = ConvolutionHarness.validateTensorOpConv3DBias(log: log)
    for (index, result) in validation.cases.enumerated() {
      log("Case \(index + 1)/\(validation.cases.count): \(result.dimensions)")
      log("Max absolute error: \(result.tensorOpValidation.maxAbsoluteError)")
      log("Mismatches above tolerance: \(result.tensorOpValidation.mismatches)")
    }
    log("All Conv3D bias tensor-op validation cases passed: \(validation.allPassed)")
    return validation
  }

  nonisolated private static func runTensorOpConv3DLeftRightPaddingValidation(
    log: (String) -> Void
  ) -> Convolution3DValidationSuiteResult {
    log("")
    log("Conv3D tensor-op padded validation")
    log("Padding assumption: left=1, right=1, top=1, bottom=1")
    let buildOptions = ConvolutionHarness.defaultTensorOpConv3DBuildOptions()
    if let outputTileWidth = buildOptions.outputTileWidth, let outputTileHeight = buildOptions.outputTileHeight {
      log("Build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(outputTileWidth)x\(outputTileHeight)")
    }
    let validation = ConvolutionHarness.validateTensorOpConv3DLeftRightPadding(
      buildOptions: buildOptions,
      log: log
    )
    for (index, result) in validation.cases.enumerated() {
      log("Case \(index + 1)/\(validation.cases.count): \(result.dimensions)")
      log("Max absolute error: \(result.tensorOpValidation.maxAbsoluteError)")
      log("Mismatches above tolerance: \(result.tensorOpValidation.mismatches)")
    }
    log("All Conv3D padded validation cases passed: \(validation.allPassed)")
    return validation
  }

  nonisolated private static func runTensorOpConv3DProfile(
    log: (String) -> Void
  ) -> TensorOpConv3DProfileResult {
    let buildOptions = ConvolutionHarness.defaultTensorOpConv3DBuildOptions()
    let options = ConvolutionHarness.largeTensorOpConv3DProfileOptions()
    let dimensions = ConvolutionHarness.largeTensorOpConv3DProfileDimensions()

    log("")
    log("Large Conv3D tensor-op profile dimensions: \(dimensions)")
    let profile = ConvolutionHarness.profileTensorOpConv3D(
      dimensions: dimensions,
      buildOptions: buildOptions,
      options: options,
      log: log
    )
    if let gpuThroughput = profile.gpuThroughputGFLOPS {
      log(String(format: "Large Conv3D tensor-op GPU throughput: %.3f GFLOP/s", gpuThroughput))
    }
    return profile
  }

  nonisolated private static func runTensorOpConv3DBiasProfile(
    log: (String) -> Void
  ) -> TensorOpConv3DProfileResult {
    let buildOptions = ConvolutionHarness.defaultTensorOpConv3DBuildOptions()
    let options = ConvolutionHarness.largeTensorOpConv3DProfileOptions()
    let dimensions = ConvolutionHarness.largeTensorOpConv3DProfileDimensions()

    log("")
    log("Large Conv3D bias tensor-op profile dimensions: \(dimensions)")
    let profile = ConvolutionHarness.profileTensorOpConv3DBias(
      dimensions: dimensions,
      buildOptions: buildOptions,
      options: options,
      log: log
    )
    if let gpuThroughput = profile.gpuThroughputGFLOPS {
      log(String(format: "Large Conv3D bias tensor-op GPU throughput: %.3f GFLOP/s", gpuThroughput))
    }
    return profile
  }

  nonisolated private static func runTensorOpConv3DPermutationProfile(
    log: (String) -> Void
  ) -> TensorOpConv3DPermutationProfileResult {
    let buildOptions = ConvolutionHarness.defaultTensorOpConv3DBuildOptions()
    let options = ConvolutionHarness.largeTensorOpConv3DProfileOptions()
    let dimensions = ConvolutionHarness.largeTensorOpConv3DProfileDimensions()

    log("")
    log("Large Conv3D permutation + tensor-op profile dimensions: \(dimensions)")
    let profile = ConvolutionHarness.profileTensorOpConv3DWithPermutation(
      profileDimensions: dimensions,
      buildOptions: buildOptions,
      options: options,
      log: log
    )
    if let combinedGPUThroughput = profile.combinedGPUThroughputGFLOPS {
      log(String(format: "Large permute+Conv3D GPU throughput: %.3f GFLOP/s", combinedGPUThroughput))
    }
    return profile
  }
}
