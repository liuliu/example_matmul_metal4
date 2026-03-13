import Foundation
import Metal
import MetalPerformanceShadersGraph

public struct Conv2DDimensions: Codable, CustomStringConvertible {
  public var batchSize: Int
  public var inputHeight: Int
  public var inputWidth: Int
  public var inputChannels: Int
  public var outputChannels: Int
  public var kernelHeight: Int
  public var kernelWidth: Int
  public var strideY: Int
  public var strideX: Int
  public var dilationY: Int
  public var dilationX: Int
  public var paddingTop: Int
  public var paddingBottom: Int
  public var paddingLeft: Int
  public var paddingRight: Int

  public init(
    batchSize: Int,
    inputHeight: Int,
    inputWidth: Int,
    inputChannels: Int,
    outputChannels: Int,
    kernelHeight: Int,
    kernelWidth: Int,
    strideY: Int,
    strideX: Int,
    dilationY: Int,
    dilationX: Int,
    paddingTop: Int = 0,
    paddingBottom: Int = 0,
    paddingLeft: Int = 0,
    paddingRight: Int = 0
  ) {
    self.batchSize = batchSize
    self.inputHeight = inputHeight
    self.inputWidth = inputWidth
    self.inputChannels = inputChannels
    self.outputChannels = outputChannels
    self.kernelHeight = kernelHeight
    self.kernelWidth = kernelWidth
    self.strideY = strideY
    self.strideX = strideX
    self.dilationY = dilationY
    self.dilationX = dilationX
    self.paddingTop = paddingTop
    self.paddingBottom = paddingBottom
    self.paddingLeft = paddingLeft
    self.paddingRight = paddingRight
  }

  public var outputHeight: Int {
    ((inputHeight + paddingTop + paddingBottom - ((kernelHeight - 1) * dilationY + 1)) / strideY) + 1
  }

  public var outputWidth: Int {
    ((inputWidth + paddingLeft + paddingRight - ((kernelWidth - 1) * dilationX + 1)) / strideX) + 1
  }

  public var description: String {
    let paddingDescription =
      (paddingTop != 0 || paddingBottom != 0 || paddingLeft != 0 || paddingRight != 0)
      ? ", PT=\(paddingTop), PB=\(paddingBottom), PL=\(paddingLeft), PR=\(paddingRight)" : ""
    return "N=\(batchSize), H=\(inputHeight), W=\(inputWidth), C=\(inputChannels), O=\(outputChannels), KH=\(kernelHeight), KW=\(kernelWidth), SH=\(strideY), SW=\(strideX), DH=\(dilationY), DW=\(dilationX)\(paddingDescription)"
  }
}

public struct Conv3DDimensions: Codable, CustomStringConvertible {
  public var batchSize: Int
  public var inputDepth: Int
  public var inputHeight: Int
  public var inputWidth: Int
  public var inputChannels: Int
  public var outputChannels: Int
  public var kernelDepth: Int
  public var kernelHeight: Int
  public var kernelWidth: Int
  public var strideZ: Int
  public var strideY: Int
  public var strideX: Int
  public var dilationZ: Int
  public var dilationY: Int
  public var dilationX: Int
  public var paddingTop: Int
  public var paddingBottom: Int
  public var paddingLeft: Int
  public var paddingRight: Int

  public init(
    batchSize: Int,
    inputDepth: Int,
    inputHeight: Int,
    inputWidth: Int,
    inputChannels: Int,
    outputChannels: Int,
    kernelDepth: Int,
    kernelHeight: Int,
    kernelWidth: Int,
    strideZ: Int,
    strideY: Int,
    strideX: Int,
    dilationZ: Int,
    dilationY: Int,
    dilationX: Int,
    paddingTop: Int = 0,
    paddingBottom: Int = 0,
    paddingLeft: Int = 0,
    paddingRight: Int = 0
  ) {
    self.batchSize = batchSize
    self.inputDepth = inputDepth
    self.inputHeight = inputHeight
    self.inputWidth = inputWidth
    self.inputChannels = inputChannels
    self.outputChannels = outputChannels
    self.kernelDepth = kernelDepth
    self.kernelHeight = kernelHeight
    self.kernelWidth = kernelWidth
    self.strideZ = strideZ
    self.strideY = strideY
    self.strideX = strideX
    self.dilationZ = dilationZ
    self.dilationY = dilationY
    self.dilationX = dilationX
    self.paddingTop = paddingTop
    self.paddingBottom = paddingBottom
    self.paddingLeft = paddingLeft
    self.paddingRight = paddingRight
  }

  public var outputDepth: Int {
    ((inputDepth - ((kernelDepth - 1) * dilationZ + 1)) / strideZ) + 1
  }

  public var outputHeight: Int {
    ((inputHeight + paddingTop + paddingBottom - ((kernelHeight - 1) * dilationY + 1)) / strideY) + 1
  }

  public var outputWidth: Int {
    ((inputWidth + paddingLeft + paddingRight - ((kernelWidth - 1) * dilationX + 1)) / strideX) + 1
  }

  public var description: String {
    let paddingDescription =
      (paddingTop != 0 || paddingBottom != 0 || paddingLeft != 0 || paddingRight != 0)
      ? ", PT=\(paddingTop), PB=\(paddingBottom), PL=\(paddingLeft), PR=\(paddingRight)" : ""
    return "N=\(batchSize), D=\(inputDepth), H=\(inputHeight), W=\(inputWidth), C=\(inputChannels), O=\(outputChannels), KD=\(kernelDepth), KH=\(kernelHeight), KW=\(kernelWidth), SZ=\(strideZ), SY=\(strideY), SX=\(strideX), DZ=\(dilationZ), DY=\(dilationY), DX=\(dilationX)\(paddingDescription)"
  }
}

public struct BuildOptions: Codable {
  public var executionSIMDGroups: Int
  public var outputTileWidth: Int?
  public var outputTileHeight: Int?
  public var outputBaseX: Int?
  public var outputBaseY: Int?
  public var dispatchGridWidth: Int?
  public var dispatchGridHeight: Int?

  public init(
    executionSIMDGroups: Int,
    outputTileWidth: Int? = nil,
    outputTileHeight: Int? = nil,
    outputBaseX: Int? = nil,
    outputBaseY: Int? = nil,
    dispatchGridWidth: Int? = nil,
    dispatchGridHeight: Int? = nil
  ) {
    self.executionSIMDGroups = executionSIMDGroups
    self.outputTileWidth = outputTileWidth
    self.outputTileHeight = outputTileHeight
    self.outputBaseX = outputBaseX
    self.outputBaseY = outputBaseY
    self.dispatchGridWidth = dispatchGridWidth
    self.dispatchGridHeight = dispatchGridHeight
  }

  public var usesOutputTiling: Bool {
    outputTileWidth != nil && outputTileHeight != nil
  }
}

public struct ValidationResult: Codable {
  public var maxAbsoluteError: Float
  public var mismatches: Int

  public init(maxAbsoluteError: Float, mismatches: Int) {
    self.maxAbsoluteError = maxAbsoluteError
    self.mismatches = mismatches
  }
}

public struct ConvolutionCaseResult: Codable {
  public var dimensions: Conv2DDimensions
  public var tensorOpValidation: ValidationResult
  public var mpsGraphValidation: ValidationResult
  public var backendAgreement: ValidationResult
  public var tensorOpWallLatencyMS: Double?
  public var tensorOpGPULatencyMS: Double?
  public var mpsGraphWallLatencyMS: Double?

  public init(
    dimensions: Conv2DDimensions,
    tensorOpValidation: ValidationResult,
    mpsGraphValidation: ValidationResult,
    backendAgreement: ValidationResult,
    tensorOpWallLatencyMS: Double?,
    tensorOpGPULatencyMS: Double?,
    mpsGraphWallLatencyMS: Double?
  ) {
    self.dimensions = dimensions
    self.tensorOpValidation = tensorOpValidation
    self.mpsGraphValidation = mpsGraphValidation
    self.backendAgreement = backendAgreement
    self.tensorOpWallLatencyMS = tensorOpWallLatencyMS
    self.tensorOpGPULatencyMS = tensorOpGPULatencyMS
    self.mpsGraphWallLatencyMS = mpsGraphWallLatencyMS
  }
}

public struct ConvolutionValidationSuiteResult: Codable {
  public var deviceName: String
  public var allPassed: Bool
  public var cases: [ConvolutionCaseResult]

  public init(deviceName: String, allPassed: Bool, cases: [ConvolutionCaseResult]) {
    self.deviceName = deviceName
    self.allPassed = allPassed
    self.cases = cases
  }
}

public struct ProfileOptions: Codable {
  public var warmupIterations: Int
  public var timedIterations: Int
  public var duplicatedCount: Int

  public init(warmupIterations: Int, timedIterations: Int, duplicatedCount: Int = 1) {
    self.warmupIterations = warmupIterations
    self.timedIterations = timedIterations
    self.duplicatedCount = duplicatedCount
  }
}

public struct BackendProfileResult: Codable {
  public var averageWallLatencyMS: Double
  public var averageGPULatencyMS: Double?
  public var throughputGFLOPS: Double

  public init(
    averageWallLatencyMS: Double,
    averageGPULatencyMS: Double?,
    throughputGFLOPS: Double
  ) {
    self.averageWallLatencyMS = averageWallLatencyMS
    self.averageGPULatencyMS = averageGPULatencyMS
    self.throughputGFLOPS = throughputGFLOPS
  }
}

public struct DispatchGridSize: Codable {
  public var width: Int
  public var height: Int
  public var depth: Int

  public init(width: Int, height: Int, depth: Int) {
    self.width = width
    self.height = height
    self.depth = depth
  }
}

public struct TensorOpTileRegionProfileResult: Codable {
  public var label: String
  public var buildOptions: BuildOptions
  public var dispatchGrid: DispatchGridSize
  public var averageWallLatencyMS: Double
  public var averageGPULatencyMS: Double?

  public init(
    label: String,
    buildOptions: BuildOptions,
    dispatchGrid: DispatchGridSize,
    averageWallLatencyMS: Double,
    averageGPULatencyMS: Double?
  ) {
    self.label = label
    self.buildOptions = buildOptions
    self.dispatchGrid = dispatchGrid
    self.averageWallLatencyMS = averageWallLatencyMS
    self.averageGPULatencyMS = averageGPULatencyMS
  }
}

public struct TensorOpTileProfileResult: Codable {
  public var dimensions: Conv2DDimensions
  public var tileWidth: Int
  public var tileHeight: Int
  public var options: ProfileOptions
  public var regions: [TensorOpTileRegionProfileResult]
  public var averageWallLatencyMS: Double
  public var averageGPULatencyMS: Double?
  public var wallThroughputGFLOPS: Double
  public var gpuThroughputGFLOPS: Double?

  public init(
    dimensions: Conv2DDimensions,
    tileWidth: Int,
    tileHeight: Int,
    options: ProfileOptions,
    regions: [TensorOpTileRegionProfileResult],
    averageWallLatencyMS: Double,
    averageGPULatencyMS: Double?,
    wallThroughputGFLOPS: Double,
    gpuThroughputGFLOPS: Double?
  ) {
    self.dimensions = dimensions
    self.tileWidth = tileWidth
    self.tileHeight = tileHeight
    self.options = options
    self.regions = regions
    self.averageWallLatencyMS = averageWallLatencyMS
    self.averageGPULatencyMS = averageGPULatencyMS
    self.wallThroughputGFLOPS = wallThroughputGFLOPS
    self.gpuThroughputGFLOPS = gpuThroughputGFLOPS
  }
}

public struct Convolution3DCaseResult: Codable {
  public var dimensions: Conv3DDimensions
  public var tensorOpValidation: ValidationResult
  public var tensorOpWallLatencyMS: Double?
  public var tensorOpGPULatencyMS: Double?

  public init(
    dimensions: Conv3DDimensions,
    tensorOpValidation: ValidationResult,
    tensorOpWallLatencyMS: Double?,
    tensorOpGPULatencyMS: Double?
  ) {
    self.dimensions = dimensions
    self.tensorOpValidation = tensorOpValidation
    self.tensorOpWallLatencyMS = tensorOpWallLatencyMS
    self.tensorOpGPULatencyMS = tensorOpGPULatencyMS
  }
}

public struct Convolution3DValidationSuiteResult: Codable {
  public var deviceName: String
  public var allPassed: Bool
  public var cases: [Convolution3DCaseResult]

  public init(deviceName: String, allPassed: Bool, cases: [Convolution3DCaseResult]) {
    self.deviceName = deviceName
    self.allPassed = allPassed
    self.cases = cases
  }
}

public struct Conv3DSpatialPadding: Codable, CustomStringConvertible {
  public var top: Int
  public var bottom: Int
  public var left: Int
  public var right: Int

  public init(top: Int, bottom: Int, left: Int, right: Int) {
    self.top = top
    self.bottom = bottom
    self.left = left
    self.right = right
  }

  public var description: String {
    "top=\(top), bottom=\(bottom), left=\(left), right=\(right)"
  }
}

public struct PaddedConvolution3DCaseResult: Codable {
  public var logicalDimensions: Conv3DDimensions
  public var paddedDimensions: Conv3DDimensions
  public var padding: Conv3DSpatialPadding
  public var tensorOpValidation: ValidationResult
  public var tensorOpWallLatencyMS: Double?
  public var tensorOpGPULatencyMS: Double?

  public init(
    logicalDimensions: Conv3DDimensions,
    paddedDimensions: Conv3DDimensions,
    padding: Conv3DSpatialPadding,
    tensorOpValidation: ValidationResult,
    tensorOpWallLatencyMS: Double?,
    tensorOpGPULatencyMS: Double?
  ) {
    self.logicalDimensions = logicalDimensions
    self.paddedDimensions = paddedDimensions
    self.padding = padding
    self.tensorOpValidation = tensorOpValidation
    self.tensorOpWallLatencyMS = tensorOpWallLatencyMS
    self.tensorOpGPULatencyMS = tensorOpGPULatencyMS
  }
}

public struct PaddedConvolution3DValidationSuiteResult: Codable {
  public var deviceName: String
  public var allPassed: Bool
  public var cases: [PaddedConvolution3DCaseResult]

  public init(deviceName: String, allPassed: Bool, cases: [PaddedConvolution3DCaseResult]) {
    self.deviceName = deviceName
    self.allPassed = allPassed
    self.cases = cases
  }
}

public struct TensorOpConv3DProfileResult: Codable {
  public var dimensions: Conv3DDimensions
  public var buildOptions: BuildOptions
  public var options: ProfileOptions
  public var dispatchGrid: DispatchGridSize
  public var averageWallLatencyMS: Double
  public var averageGPULatencyMS: Double?
  public var wallThroughputGFLOPS: Double
  public var gpuThroughputGFLOPS: Double?

  public init(
    dimensions: Conv3DDimensions,
    buildOptions: BuildOptions,
    options: ProfileOptions,
    dispatchGrid: DispatchGridSize,
    averageWallLatencyMS: Double,
    averageGPULatencyMS: Double?,
    wallThroughputGFLOPS: Double,
    gpuThroughputGFLOPS: Double?
  ) {
    self.dimensions = dimensions
    self.buildOptions = buildOptions
    self.options = options
    self.dispatchGrid = dispatchGrid
    self.averageWallLatencyMS = averageWallLatencyMS
    self.averageGPULatencyMS = averageGPULatencyMS
    self.wallThroughputGFLOPS = wallThroughputGFLOPS
    self.gpuThroughputGFLOPS = gpuThroughputGFLOPS
  }
}

public struct TensorOpConv3DPermutationProfileResult: Codable {
  public var validationDimensions: Conv3DDimensions
  public var profileDimensions: Conv3DDimensions
  public var buildOptions: BuildOptions
  public var options: ProfileOptions
  public var permutationValidation: ValidationResult
  public var combinedValidation: ValidationResult
  public var permutationAverageWallLatencyMS: Double
  public var permutationAverageGPULatencyMS: Double?
  public var permutationWallBandwidthGBPS: Double
  public var permutationGPUBandwidthGBPS: Double?
  public var combinedAverageWallLatencyMS: Double
  public var combinedAverageGPULatencyMS: Double?
  public var combinedWallThroughputGFLOPS: Double
  public var combinedGPUThroughputGFLOPS: Double?

  public init(
    validationDimensions: Conv3DDimensions,
    profileDimensions: Conv3DDimensions,
    buildOptions: BuildOptions,
    options: ProfileOptions,
    permutationValidation: ValidationResult,
    combinedValidation: ValidationResult,
    permutationAverageWallLatencyMS: Double,
    permutationAverageGPULatencyMS: Double?,
    permutationWallBandwidthGBPS: Double,
    permutationGPUBandwidthGBPS: Double?,
    combinedAverageWallLatencyMS: Double,
    combinedAverageGPULatencyMS: Double?,
    combinedWallThroughputGFLOPS: Double,
    combinedGPUThroughputGFLOPS: Double?
  ) {
    self.validationDimensions = validationDimensions
    self.profileDimensions = profileDimensions
    self.buildOptions = buildOptions
    self.options = options
    self.permutationValidation = permutationValidation
    self.combinedValidation = combinedValidation
    self.permutationAverageWallLatencyMS = permutationAverageWallLatencyMS
    self.permutationAverageGPULatencyMS = permutationAverageGPULatencyMS
    self.permutationWallBandwidthGBPS = permutationWallBandwidthGBPS
    self.permutationGPUBandwidthGBPS = permutationGPUBandwidthGBPS
    self.combinedAverageWallLatencyMS = combinedAverageWallLatencyMS
    self.combinedAverageGPULatencyMS = combinedAverageGPULatencyMS
    self.combinedWallThroughputGFLOPS = combinedWallThroughputGFLOPS
    self.combinedGPUThroughputGFLOPS = combinedGPUThroughputGFLOPS
  }
}

public struct MPSGraphWeightLayoutProfileResult: Codable {
  public var validationDimensions: Conv2DDimensions
  public var profileDimensions: Conv2DDimensions
  public var options: ProfileOptions
  public var hwioValidation: ValidationResult
  public var oihwValidation: ValidationResult
  public var validationAgreement: ValidationResult
  public var profileAgreement: ValidationResult
  public var hwio: BackendProfileResult
  public var oihw: BackendProfileResult

  public init(
    validationDimensions: Conv2DDimensions,
    profileDimensions: Conv2DDimensions,
    options: ProfileOptions,
    hwioValidation: ValidationResult,
    oihwValidation: ValidationResult,
    validationAgreement: ValidationResult,
    profileAgreement: ValidationResult,
    hwio: BackendProfileResult,
    oihw: BackendProfileResult
  ) {
    self.validationDimensions = validationDimensions
    self.profileDimensions = profileDimensions
    self.options = options
    self.hwioValidation = hwioValidation
    self.oihwValidation = oihwValidation
    self.validationAgreement = validationAgreement
    self.profileAgreement = profileAgreement
    self.hwio = hwio
    self.oihw = oihw
  }
}

public struct MPSGraphConv3DWeightLayoutProfileResult: Codable {
  public var validationDimensions: Conv3DDimensions
  public var profileDimensions: Conv3DDimensions
  public var options: ProfileOptions
  public var dhwioValidation: ValidationResult
  public var oidhwValidation: ValidationResult
  public var validationAgreement: ValidationResult
  public var profileAgreement: ValidationResult
  public var dhwio: BackendProfileResult
  public var oidhw: BackendProfileResult

  public init(
    validationDimensions: Conv3DDimensions,
    profileDimensions: Conv3DDimensions,
    options: ProfileOptions,
    dhwioValidation: ValidationResult,
    oidhwValidation: ValidationResult,
    validationAgreement: ValidationResult,
    profileAgreement: ValidationResult,
    dhwio: BackendProfileResult,
    oidhw: BackendProfileResult
  ) {
    self.validationDimensions = validationDimensions
    self.profileDimensions = profileDimensions
    self.options = options
    self.dhwioValidation = dhwioValidation
    self.oidhwValidation = oidhwValidation
    self.validationAgreement = validationAgreement
    self.profileAgreement = profileAgreement
    self.dhwio = dhwio
    self.oidhw = oidhw
  }
}

public struct GPUTraceCaptureResult: Codable {
  public var dimensions: Conv2DDimensions
  public var duplicatedCount: Int
  public var tensorOpTracePath: String
  public var mpsGraphTracePath: String

  public init(
    dimensions: Conv2DDimensions,
    duplicatedCount: Int,
    tensorOpTracePath: String,
    mpsGraphTracePath: String
  ) {
    self.dimensions = dimensions
    self.duplicatedCount = duplicatedCount
    self.tensorOpTracePath = tensorOpTracePath
    self.mpsGraphTracePath = mpsGraphTracePath
  }
}

public struct GPUCaptureSupportResult: Codable {
  public var gpuTraceDocumentSupported: Bool
  public var developerToolsSupported: Bool

  public init(gpuTraceDocumentSupported: Bool, developerToolsSupported: Bool) {
    self.gpuTraceDocumentSupported = gpuTraceDocumentSupported
    self.developerToolsSupported = developerToolsSupported
  }
}

public struct ConvolutionProfileResult: Codable {
  public var dimensions: Conv2DDimensions
  public var options: ProfileOptions
  public var tensorOpValidation: ValidationResult
  public var mpsGraphValidation: ValidationResult
  public var backendAgreement: ValidationResult
  public var tensorOp: BackendProfileResult
  public var mpsGraph: BackendProfileResult

  public init(
    dimensions: Conv2DDimensions,
    options: ProfileOptions,
    tensorOpValidation: ValidationResult,
    mpsGraphValidation: ValidationResult,
    backendAgreement: ValidationResult,
    tensorOp: BackendProfileResult,
    mpsGraph: BackendProfileResult
  ) {
    self.dimensions = dimensions
    self.options = options
    self.tensorOpValidation = tensorOpValidation
    self.mpsGraphValidation = mpsGraphValidation
    self.backendAgreement = backendAgreement
    self.tensorOp = tensorOp
    self.mpsGraph = mpsGraph
  }
}

public struct TensorOpVariantCaseResult: Codable {
  public var dimensions: Conv2DDimensions
  public var validation: ValidationResult?
  public var wallLatencyMS: Double?
  public var gpuLatencyMS: Double?
  public var error: String?

  public init(
    dimensions: Conv2DDimensions,
    validation: ValidationResult?,
    wallLatencyMS: Double?,
    gpuLatencyMS: Double?,
    error: String?
  ) {
    self.dimensions = dimensions
    self.validation = validation
    self.wallLatencyMS = wallLatencyMS
    self.gpuLatencyMS = gpuLatencyMS
    self.error = error
  }
}

public struct TensorOpVariantResult: Codable {
  public var name: String
  public var scope: String
  public var destination: String
  public var outputElementType: String
  public var usesOffsets: Bool
  public var executionSIMDGroups: Int
  public var threadExecutionWidth: Int?
  public var threadsPerThreadgroup: Int?
  public var allPassed: Bool
  public var cases: [TensorOpVariantCaseResult]

  public init(
    name: String,
    scope: String,
    destination: String,
    outputElementType: String,
    usesOffsets: Bool,
    executionSIMDGroups: Int,
    threadExecutionWidth: Int?,
    threadsPerThreadgroup: Int?,
    allPassed: Bool,
    cases: [TensorOpVariantCaseResult]
  ) {
    self.name = name
    self.scope = scope
    self.destination = destination
    self.outputElementType = outputElementType
    self.usesOffsets = usesOffsets
    self.executionSIMDGroups = executionSIMDGroups
    self.threadExecutionWidth = threadExecutionWidth
    self.threadsPerThreadgroup = threadsPerThreadgroup
    self.allPassed = allPassed
    self.cases = cases
  }
}

public struct TensorOpVariantSuiteResult: Codable {
  public var deviceName: String
  public var firstPassingVariantName: String?
  public var variants: [TensorOpVariantResult]

  public init(deviceName: String, firstPassingVariantName: String?, variants: [TensorOpVariantResult]) {
    self.deviceName = deviceName
    self.firstPassingVariantName = firstPassingVariantName
    self.variants = variants
  }
}

struct ExecutionResult {
  var output: [Float]
  var wallLatency: Double?
  var gpuLatency: Double?
}

enum ConvolutionWeightLayout {
  case hwio
  case oihw
}

enum Convolution3DWeightLayout {
  case dhwio
  case oidhw
}

enum TensorOpScopeKind {
  case executionSimdgroups
  case executionSimdgroup
}

enum TensorOpDestinationKind {
  case direct
  case cooperative
}

enum TensorOpOutputElementKind {
  case half
  case float
}

enum TensorOpVariant: CaseIterable {
  case baseline
  case directNoOffsets
  case directExecutionSimdgroup
  case directFloat
  case directFloatNoOffsets
  case cooperativeHalf
  case cooperativeHalfNoOffsets

  var name: String {
    switch self {
    case .baseline:
      return "direct-half-offset-simdgroups"
    case .directNoOffsets:
      return "direct-half-no-offset-simdgroups"
    case .directExecutionSimdgroup:
      return "direct-half-offset-simdgroup"
    case .directFloat:
      return "direct-float-offset-simdgroups"
    case .directFloatNoOffsets:
      return "direct-float-no-offset-simdgroups"
    case .cooperativeHalf:
      return "cooperative-half-offset-simdgroups"
    case .cooperativeHalfNoOffsets:
      return "cooperative-half-no-offset-simdgroups"
    }
  }

  var scopeKind: TensorOpScopeKind {
    switch self {
    case .directExecutionSimdgroup:
      return .executionSimdgroup
    default:
      return .executionSimdgroups
    }
  }

  var destinationKind: TensorOpDestinationKind {
    switch self {
    case .cooperativeHalf, .cooperativeHalfNoOffsets:
      return .cooperative
    default:
      return .direct
    }
  }

  var outputElementKind: TensorOpOutputElementKind {
    switch self {
    case .directFloat, .directFloatNoOffsets:
      return .float
    default:
      return .half
    }
  }

  var usesOffsets: Bool {
    switch self {
    case .directNoOffsets, .directFloatNoOffsets, .cooperativeHalfNoOffsets:
      return false
    default:
      return true
    }
  }

  var executionSIMDGroupsOverride: Int? {
    switch scopeKind {
    case .executionSimdgroups:
      return nil
    case .executionSimdgroup:
      return 1
    }
  }
}

func runWithGPUTraceCapture<T>(commandQueue: MTLCommandQueue, outputURL: URL, body: () -> T) throws -> T {
  let captureManager = MTLCaptureManager.shared()
  let gpuTraceDocumentSupported = captureManager.supportsDestination(.gpuTraceDocument)
  let developerToolsSupported = captureManager.supportsDestination(.developerTools)
  guard gpuTraceDocumentSupported else {
    throw NSError(
      domain: "ConvolutionHarness",
      code: 1,
      userInfo: [
        NSLocalizedDescriptionKey:
          "GPU trace document capture is not supported on this device. developerToolsSupported=\(developerToolsSupported)"
      ]
    )
  }

  let fileManager = FileManager.default
  if fileManager.fileExists(atPath: outputURL.path) {
    try fileManager.removeItem(at: outputURL)
  }

  let descriptor = MTLCaptureDescriptor()
  descriptor.captureObject = commandQueue
  descriptor.destination = .gpuTraceDocument
  descriptor.outputURL = outputURL

  try captureManager.startCapture(with: descriptor)
  let result = body()
  captureManager.stopCapture()

  let deadline = Date().addingTimeInterval(10)
  while !fileManager.fileExists(atPath: outputURL.path) && Date() < deadline {
    Thread.sleep(forTimeInterval: 0.1)
  }

  guard fileManager.fileExists(atPath: outputURL.path) else {
    throw NSError(
      domain: "ConvolutionHarness",
      code: 2,
      userInfo: [NSLocalizedDescriptionKey: "GPU trace document was not written to \(outputURL.path)."]
    )
  }
  return result
}

final class TensorOpSession {
  private let dimensions: Conv2DDimensions
  private let buildOptions: BuildOptions
  private let variant: TensorOpVariant
  private let executionSIMDGroups: Int
  private let commandQueue: MTLCommandQueue
  private let pipelineState: MTLComputePipelineState
  private let activationBuffer: MTLBuffer
  private let weightBuffer: MTLBuffer
  private let outputBuffer: MTLBuffer
  private let outputCount: Int

  init(
    device: MTLDevice,
    dimensions: Conv2DDimensions,
    buildOptions: BuildOptions,
    activation: [Float16],
    weights: [Float16],
    variant: TensorOpVariant = .baseline
  ) throws {
    self.dimensions = dimensions
    self.buildOptions = buildOptions
    self.variant = variant
    executionSIMDGroups = variant.executionSIMDGroupsOverride ?? buildOptions.executionSIMDGroups

    guard device.supportsFamily(.metal4) else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 1,
        userInfo: [NSLocalizedDescriptionKey: "This device does not support the tensor operations used in the shader."]
      )
    }

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 2,
        userInfo: [NSLocalizedDescriptionKey: "Could not create command queue."]
      )
    }
    self.commandQueue = commandQueue
    self.commandQueue.label = "ConvolutionBench.TensorOpQueue"

    let effectiveBuildOptions = BuildOptions(
      executionSIMDGroups: executionSIMDGroups,
      outputTileWidth: buildOptions.outputTileWidth,
      outputTileHeight: buildOptions.outputTileHeight,
      outputBaseX: buildOptions.outputBaseX,
      outputBaseY: buildOptions.outputBaseY,
      dispatchGridWidth: buildOptions.dispatchGridWidth,
      dispatchGridHeight: buildOptions.dispatchGridHeight
    )
    let library = try device.makeLibrary(
      source: createSource(dimensions: dimensions, buildOptions: effectiveBuildOptions, variant: variant),
      options: nil
    )

    guard let function = library.makeFunction(name: "conv2d") else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 3,
        userInfo: [NSLocalizedDescriptionKey: "Could not create tensor-op convolution function."]
      )
    }

    pipelineState = try device.makeComputePipelineState(function: function)

    outputCount = dimensions.batchSize * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    activationBuffer = copyToSharedBuffer(device: device, values: activation)
    weightBuffer = copyToSharedBuffer(device: device, values: weights)
    switch variant.outputElementKind {
    case .half:
      outputBuffer = makeZeroedSharedBuffer(device: device, count: outputCount)
    case .float:
      outputBuffer = makeZeroedFloatBuffer(device: device, count: outputCount)
    }
  }

  private func execute(duplicatedCount: Int, readOutput: Bool) -> ExecutionResult {
    precondition(duplicatedCount > 0)

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create command buffer or encoder")
    }

    computeCommandEncoder.setComputePipelineState(pipelineState)
    computeCommandEncoder.setBuffer(activationBuffer, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
    computeCommandEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
    computeCommandEncoder.useResource(activationBuffer, usage: .read)
    computeCommandEncoder.useResource(weightBuffer, usage: .read)
    computeCommandEncoder.useResource(outputBuffer, usage: .write)

    let threadgroups = tensorOpDispatchThreadgroups(dimensions: dimensions, buildOptions: buildOptions)
    let simdgroupWidth = pipelineState.threadExecutionWidth
    let threadsPerThreadgroup = MTLSize(
      width: simdgroupWidth * executionSIMDGroups,
      height: 1,
      depth: 1
    )

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<duplicatedCount {
      computeCommandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    }
    computeCommandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = DispatchTime.now().uptimeNanoseconds

    if commandBuffer.status != .completed {
      let errorDescription = commandBuffer.error.map { "\($0)" } ?? "none"
      fatalError("Tensor-op command buffer did not complete successfully. status=\(commandBuffer.status.rawValue), error=\(errorDescription)")
    }

    return ExecutionResult(
      output: readOutput ? readOutputBuffer(outputBuffer, count: outputCount, elementKind: variant.outputElementKind) : [],
      wallLatency: Double(end - start) / 1e9,
      gpuLatency: commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    )
  }

  func run(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: true)
  }

  func runWithoutOutputReadback(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: false)
  }

  func capture(duplicatedCount: Int, outputURL: URL) throws -> ExecutionResult {
    try runWithGPUTraceCapture(commandQueue: commandQueue, outputURL: outputURL) {
      run(duplicatedCount: duplicatedCount)
    }
  }

  var threadExecutionWidth: Int {
    pipelineState.threadExecutionWidth
  }

  var threadsPerThreadgroup: Int {
    pipelineState.threadExecutionWidth * executionSIMDGroups
  }
}

final class TensorOpSession3D {
  private let dimensions: Conv3DDimensions
  private let spatialDimensions: Conv2DDimensions
  private let buildOptions: BuildOptions
  private let commandQueue: MTLCommandQueue
  private let multiplyPipelineState: MTLComputePipelineState
  private let accumulatePipelineState: MTLComputePipelineState
  private let activationBuffer: MTLBuffer
  private let weightBuffer: MTLBuffer
  private let outputBuffer: MTLBuffer
  private let outputCount: Int

  init(
    device: MTLDevice,
    dimensions: Conv3DDimensions,
    buildOptions: BuildOptions,
    offsets: TensorOpConv3DHorizontalPaddingOffsets? = nil,
    activation: [Float16],
    weights: [Float16]
  ) throws {
    precondition(buildOptions.usesOutputTiling, "Conv3D tensor-op session requires tiled build options.")

    let spatialDimensions = conv3DSpatialSliceDimensions(dimensions)
    self.dimensions = dimensions
    self.spatialDimensions = spatialDimensions
    self.buildOptions = buildOptions

    guard device.supportsFamily(.metal4) else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 10,
        userInfo: [NSLocalizedDescriptionKey: "This device does not support the tensor operations used in the Conv3D shader."]
      )
    }

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 11,
        userInfo: [NSLocalizedDescriptionKey: "Could not create command queue for Conv3D tensor op."]
      )
    }
    self.commandQueue = commandQueue
    self.commandQueue.label = "ConvolutionBench.TensorOp3DQueue"

    let library = try device.makeLibrary(
      source: createConv3DTensorOpSource(dimensions: dimensions, buildOptions: buildOptions),
      options: nil
    )
    guard let multiplyFunction = library.makeFunction(name: TensorOpConv3DMode.multiply.functionName()),
          let accumulateFunction = library.makeFunction(name: TensorOpConv3DMode.multiplyAccumulate.functionName()) else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 12,
        userInfo: [NSLocalizedDescriptionKey: "Could not create Conv3D tensor-op functions."]
      )
    }

    multiplyPipelineState = try device.makeComputePipelineState(function: multiplyFunction)
    accumulatePipelineState = try device.makeComputePipelineState(function: accumulateFunction)

    outputCount = dimensions.batchSize * dimensions.outputDepth * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    activationBuffer = copyToSharedBuffer(device: device, values: activation)
    weightBuffer = copyToSharedBuffer(device: device, values: weights)
    outputBuffer = makeZeroedSharedBuffer(device: device, count: outputCount)
  }

  private func execute(duplicatedCount: Int, readOutput: Bool) -> ExecutionResult {
    precondition(duplicatedCount > 0)

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create Conv3D tensor-op command buffer or encoder")
    }

    computeCommandEncoder.useResource(activationBuffer, usage: .read)
    computeCommandEncoder.useResource(weightBuffer, usage: .read)
    computeCommandEncoder.useResource(outputBuffer, usage: [.read, .write])

    let threadgroups = tensorOpDispatchThreadgroups(dimensions: spatialDimensions, buildOptions: buildOptions)
    let simdgroupWidth = multiplyPipelineState.threadExecutionWidth
    let threadsPerThreadgroup = MTLSize(
      width: simdgroupWidth * buildOptions.executionSIMDGroups,
      height: 1,
      depth: 1
    )

    let inputSliceElementCount = dimensions.inputHeight * dimensions.inputWidth * dimensions.inputChannels
    let weightSliceElementCount = dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * dimensions.outputChannels
    let outputSliceElementCount = dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    let batchInputSliceCount = dimensions.inputDepth * inputSliceElementCount
    let batchOutputSliceCount = dimensions.outputDepth * outputSliceElementCount

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<duplicatedCount {
      for n in 0..<dimensions.batchSize {
        let batchActivationBase = n * batchInputSliceCount
        let batchOutputBase = n * batchOutputSliceCount

        for oz in 0..<dimensions.outputDepth {
          let outputBase = UInt32(batchOutputBase + oz * outputSliceElementCount)

          for kd in 0..<dimensions.kernelDepth {
            let inputDepthIndex = oz * dimensions.strideZ + kd * dimensions.dilationZ
            let activationBase = UInt32(batchActivationBase + inputDepthIndex * inputSliceElementCount)
            let weightsBase = UInt32(kd * weightSliceElementCount)
            let pipelineState = kd == 0 ? multiplyPipelineState : accumulatePipelineState

            computeCommandEncoder.setComputePipelineState(pipelineState)
            computeCommandEncoder.setBuffer(activationBuffer, offset: 0, index: 0)
            computeCommandEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
            computeCommandEncoder.setBuffer(outputBuffer, offset: 0, index: 2)

            var activationBaseVar = activationBase
            var weightsBaseVar = weightsBase
            var outputBaseVar = outputBase
            computeCommandEncoder.setBytes(&activationBaseVar, length: MemoryLayout<UInt32>.size, index: 3)
            computeCommandEncoder.setBytes(&weightsBaseVar, length: MemoryLayout<UInt32>.size, index: 4)
            computeCommandEncoder.setBytes(&outputBaseVar, length: MemoryLayout<UInt32>.size, index: 5)
            computeCommandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
          }
        }
      }
    }
    computeCommandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = DispatchTime.now().uptimeNanoseconds

    if commandBuffer.status != .completed {
      let errorDescription = commandBuffer.error.map { "\($0)" } ?? "none"
      fatalError("Conv3D tensor-op command buffer did not complete successfully. status=\(commandBuffer.status.rawValue), error=\(errorDescription)")
    }

    return ExecutionResult(
      output: readOutput ? readFloat16BufferAsFloat(outputBuffer, count: outputCount) : [],
      wallLatency: Double(end - start) / 1e9,
      gpuLatency: commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    )
  }

  func run(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: true)
  }

  func runWithoutOutputReadback(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: false)
  }

  var threadExecutionWidth: Int {
    multiplyPipelineState.threadExecutionWidth
  }

  var threadsPerThreadgroup: Int {
    multiplyPipelineState.threadExecutionWidth * buildOptions.executionSIMDGroups
  }
}

struct TensorOpConv3DHorizontalPaddingRegion {
  var label: String
  var buildOptions: BuildOptions
  var logicalOutputWidth: Int
  var logicalOutputHeight: Int
  var inputBaseX: Int
  var inputBaseY: Int
  var kernelSliceXStart: Int
  var kernelSliceXWidth: Int
  var offsetX: Int
  var offsetY: Int
}

public struct TensorOpConv3DHorizontalPaddingOffsets {
  var left: Int
  var interior: Int
  var right: Int

  public init(left: Int, interior: Int, right: Int) {
    self.left = left
    self.interior = interior
    self.right = right
  }
}

func tensorOpConv3DHorizontalPaddingRegions(
  dimensions: Conv3DDimensions,
  buildOptions: BuildOptions,
  offsets: TensorOpConv3DHorizontalPaddingOffsets? = nil
) -> [TensorOpConv3DHorizontalPaddingRegion] {
  let outputHeight = dimensions.outputHeight
  let outputWidth = dimensions.outputWidth
  precondition(outputHeight > 0 && outputWidth > 0)
  guard let interiorTileWidth = buildOptions.outputTileWidth,
        let interiorTileHeight = buildOptions.outputTileHeight else {
    fatalError("Horizontal padding regions require tiled build options.")
  }
  let resolvedOffsets =
    offsets
    ?? TensorOpConv3DHorizontalPaddingOffsets(
      left: dimensions.kernelWidth / 2 - 1,
      interior: dimensions.kernelWidth / 2,
      right: dimensions.kernelWidth / 2
    )

  var regions: [TensorOpConv3DHorizontalPaddingRegion] = []
  let leftRegions = offsetTensorOpTileCoverageRegions(
    outputWidth: 1,
    outputHeight: outputHeight,
    tileWidth: 1,
    tileHeight: interiorTileHeight,
    executionSIMDGroups: buildOptions.executionSIMDGroups,
    outputBaseX: 0,
    outputBaseY: 0
  )
  regions.append(contentsOf: leftRegions.map {
    TensorOpConv3DHorizontalPaddingRegion(
      label: "left-\($0.label)",
      buildOptions: $0.buildOptions,
      logicalOutputWidth: 1,
      logicalOutputHeight: outputHeight,
      inputBaseX: 0,
      inputBaseY: 0,
      kernelSliceXStart: 1,
      kernelSliceXWidth: dimensions.kernelWidth - 1,
      offsetX: resolvedOffsets.left,
      offsetY: dimensions.kernelHeight / 2
    )
  })

  if outputWidth > 2 {
    let interiorRegions = offsetTensorOpTileCoverageRegions(
      outputWidth: outputWidth - 2,
      outputHeight: outputHeight,
      tileWidth: interiorTileWidth,
      tileHeight: interiorTileHeight,
      executionSIMDGroups: buildOptions.executionSIMDGroups,
      outputBaseX: 1,
      outputBaseY: 0
    )
    regions.append(contentsOf: interiorRegions.map {
      TensorOpConv3DHorizontalPaddingRegion(
        label: "interior-\($0.label)",
        buildOptions: $0.buildOptions,
        logicalOutputWidth: outputWidth - 2,
        logicalOutputHeight: outputHeight,
        inputBaseX: 0,
        inputBaseY: 0,
        kernelSliceXStart: 0,
        kernelSliceXWidth: dimensions.kernelWidth,
        offsetX: resolvedOffsets.interior,
        offsetY: dimensions.kernelHeight / 2
      )
    })
  }

  let rightRegions = offsetTensorOpTileCoverageRegions(
    outputWidth: 1,
    outputHeight: outputHeight,
    tileWidth: 1,
    tileHeight: interiorTileHeight,
    executionSIMDGroups: buildOptions.executionSIMDGroups,
    outputBaseX: outputWidth - 1,
    outputBaseY: 0
  )
  regions.append(contentsOf: rightRegions.map {
    TensorOpConv3DHorizontalPaddingRegion(
      label: "right-\($0.label)",
      buildOptions: $0.buildOptions,
      logicalOutputWidth: 1,
      logicalOutputHeight: outputHeight,
      inputBaseX: dimensions.inputWidth - (dimensions.kernelWidth - 1),
      inputBaseY: 0,
      kernelSliceXStart: 0,
      kernelSliceXWidth: dimensions.kernelWidth - 1,
      offsetX: resolvedOffsets.right,
      offsetY: dimensions.kernelHeight / 2
    )
  })

  return regions
}

final class TensorOpHorizontalPaddingSession3D {
  private let dimensions: Conv3DDimensions
  private let buildOptions: BuildOptions
  private let commandQueue: MTLCommandQueue
  private let multiplyPipelineState: MTLComputePipelineState
  private let accumulatePipelineState: MTLComputePipelineState
  private let activationBuffer: MTLBuffer
  private let weightBuffer: MTLBuffer
  private let outputBuffer: MTLBuffer
  private let outputCount: Int

  init(
    device: MTLDevice,
    dimensions: Conv3DDimensions,
    buildOptions: BuildOptions,
    activation: [Float16],
    weights: [Float16]
  ) throws {
    precondition(dimensions.paddingTop == 1 && dimensions.paddingBottom == 1, "Readable padded Conv3D session expects padTop=padBottom=1.")
    precondition(dimensions.paddingLeft == 1 && dimensions.paddingRight == 1, "Readable padded Conv3D session expects padLeft=padRight=1.")
    precondition(
      dimensions.kernelWidth == 3 && dimensions.kernelHeight == 3 &&
      dimensions.dilationX == 1 && dimensions.dilationY == 1 &&
      dimensions.strideX == 1 && dimensions.strideY == 1,
      "Readable padded Conv3D session assumes KH=KW=3, dilationX/Y=1, strideX/Y=1."
    )
    self.dimensions = dimensions
    self.buildOptions = buildOptions

    guard device.supportsFamily(.metal4) else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 40,
        userInfo: [NSLocalizedDescriptionKey: "This device does not support the tensor operations used in the horizontal padding Conv3D shader."]
      )
    }

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 41,
        userInfo: [NSLocalizedDescriptionKey: "Could not create command queue for horizontally padded Conv3D tensor op."]
      )
    }
    self.commandQueue = commandQueue
    self.commandQueue.label = "ConvolutionBench.TensorOp3DHorizontalPaddingQueue"
    let library = try device.makeLibrary(
      source: createReadableConv3DHorizontalPaddingSource(dimensions: dimensions, buildOptions: buildOptions),
      options: nil
    )
    guard let multiplyFunction = library.makeFunction(name: TensorOpConv3DMode.multiply.functionName()),
          let accumulateFunction = library.makeFunction(name: TensorOpConv3DMode.multiplyAccumulate.functionName()) else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 42,
        userInfo: [NSLocalizedDescriptionKey: "Could not create Conv3D tensor-op functions for readable horizontal padding."]
      )
    }
    multiplyPipelineState = try device.makeComputePipelineState(function: multiplyFunction)
    accumulatePipelineState = try device.makeComputePipelineState(function: accumulateFunction)

    outputCount = dimensions.batchSize * dimensions.outputDepth * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    activationBuffer = copyToSharedBuffer(device: device, values: activation)
    weightBuffer = copyToSharedBuffer(device: device, values: weights)
    outputBuffer = makeZeroedSharedBuffer(device: device, count: outputCount)
  }

  private func execute(duplicatedCount: Int, readOutput: Bool) -> ExecutionResult {
    precondition(duplicatedCount > 0)

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create horizontally padded Conv3D tensor-op command buffer or encoder")
    }

    computeCommandEncoder.useResource(activationBuffer, usage: .read)
    computeCommandEncoder.useResource(weightBuffer, usage: .read)
    computeCommandEncoder.useResource(outputBuffer, usage: [.read, .write])

    let spatialDimensions = conv3DSpatialSliceDimensions(dimensions)
    let threadgroups = tensorOpDispatchThreadgroups(dimensions: spatialDimensions, buildOptions: buildOptions)
    let threadsPerThreadgroup = MTLSize(
      width: multiplyPipelineState.threadExecutionWidth * buildOptions.executionSIMDGroups,
      height: 1,
      depth: 1
    )
    let inputSliceElementCount = dimensions.inputHeight * dimensions.inputWidth * dimensions.inputChannels
    let weightSliceElementCount = dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * dimensions.outputChannels
    let outputSliceElementCount =
      dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    let batchInputSliceCount = dimensions.inputDepth * inputSliceElementCount
    let batchOutputSliceCount = dimensions.outputDepth * outputSliceElementCount

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<duplicatedCount {
      for n in 0..<dimensions.batchSize {
        let batchActivationBase = n * batchInputSliceCount
        let batchOutputBase = n * batchOutputSliceCount

        for oz in 0..<dimensions.outputDepth {
          let outputBase = UInt32(batchOutputBase + oz * outputSliceElementCount)

          for kd in 0..<dimensions.kernelDepth {
            let inputDepthIndex = oz * dimensions.strideZ + kd * dimensions.dilationZ
            let activationBase = UInt32(batchActivationBase + inputDepthIndex * inputSliceElementCount)
            let weightsBase = UInt32(kd * weightSliceElementCount)
            let pipelineState = kd == 0 ? multiplyPipelineState : accumulatePipelineState
            computeCommandEncoder.setComputePipelineState(pipelineState)
            computeCommandEncoder.setBuffer(activationBuffer, offset: 0, index: 0)
            computeCommandEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
            computeCommandEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
            var activationBaseVar = activationBase
            var weightsBaseVar = weightsBase
            var outputBaseVar = outputBase
            computeCommandEncoder.setBytes(&activationBaseVar, length: MemoryLayout<UInt32>.size, index: 3)
            computeCommandEncoder.setBytes(&weightsBaseVar, length: MemoryLayout<UInt32>.size, index: 4)
            computeCommandEncoder.setBytes(&outputBaseVar, length: MemoryLayout<UInt32>.size, index: 5)
            computeCommandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
          }
        }
      }
    }
    computeCommandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = DispatchTime.now().uptimeNanoseconds

    if commandBuffer.status != .completed {
      let errorDescription = commandBuffer.error.map { "\($0)" } ?? "none"
      fatalError("Horizontally padded Conv3D tensor-op command buffer did not complete successfully. status=\(commandBuffer.status.rawValue), error=\(errorDescription)")
    }

    return ExecutionResult(
      output: readOutput ? readFloat16BufferAsFloat(outputBuffer, count: outputCount) : [],
      wallLatency: Double(end - start) / 1e9,
      gpuLatency: commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    )
  }

  func run(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: true)
  }
}

final class TensorOpBiasSession3D {
  private let dimensions: Conv3DDimensions
  private let spatialDimensions: Conv2DDimensions
  private let buildOptions: BuildOptions
  private let commandQueue: MTLCommandQueue
  private let accumulatePipelineState: MTLComputePipelineState
  private let multiplyBiasPipelineState: MTLComputePipelineState
  private let activationBuffer: MTLBuffer
  private let weightBuffer: MTLBuffer
  private let outputBuffer: MTLBuffer
  private let biasBuffer: MTLBuffer
  private let outputCount: Int

  init(
    device: MTLDevice,
    dimensions: Conv3DDimensions,
    buildOptions: BuildOptions,
    activation: [Float16],
    weights: [Float16],
    bias: [Float16]
  ) throws {
    precondition(buildOptions.usesOutputTiling, "Conv3D bias tensor-op session requires tiled build options.")
    precondition(bias.count == dimensions.outputChannels)

    self.dimensions = dimensions
    self.spatialDimensions = conv3DSpatialSliceDimensions(dimensions)
    self.buildOptions = buildOptions

    guard device.supportsFamily(.metal4) else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 13,
        userInfo: [NSLocalizedDescriptionKey: "This device does not support the tensor operations used in the Conv3D bias shader."]
      )
    }

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 14,
        userInfo: [NSLocalizedDescriptionKey: "Could not create command queue for Conv3D bias tensor op."]
      )
    }
    self.commandQueue = commandQueue
    self.commandQueue.label = "ConvolutionBench.TensorOp3DBiasQueue"

    let library = try device.makeLibrary(
      source: createConv3DTensorOpSource(dimensions: dimensions, buildOptions: buildOptions, includeBias: true),
      options: nil
    )
    guard let accumulateFunction = library.makeFunction(name: TensorOpConv3DMode.multiplyAccumulate.functionName()),
          let multiplyBiasFunction = library.makeFunction(name: TensorOpConv3DMode.multiply.functionName(withBias: true)) else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 15,
        userInfo: [NSLocalizedDescriptionKey: "Could not create Conv3D bias tensor-op functions."]
      )
    }

    accumulatePipelineState = try device.makeComputePipelineState(function: accumulateFunction)
    multiplyBiasPipelineState = try device.makeComputePipelineState(function: multiplyBiasFunction)

    outputCount = dimensions.batchSize * dimensions.outputDepth * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    activationBuffer = copyToSharedBuffer(device: device, values: activation)
    weightBuffer = copyToSharedBuffer(device: device, values: weights)
    outputBuffer = makeZeroedSharedBuffer(device: device, count: outputCount)
    biasBuffer = copyToSharedBuffer(device: device, values: bias)
  }

  private func execute(duplicatedCount: Int, readOutput: Bool) -> ExecutionResult {
    precondition(duplicatedCount > 0)

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create Conv3D bias tensor-op command buffer or encoder")
    }

    computeCommandEncoder.useResource(activationBuffer, usage: .read)
    computeCommandEncoder.useResource(weightBuffer, usage: .read)
    computeCommandEncoder.useResource(outputBuffer, usage: [.read, .write])
    computeCommandEncoder.useResource(biasBuffer, usage: .read)

    let threadgroups = tensorOpDispatchThreadgroups(dimensions: spatialDimensions, buildOptions: buildOptions)
    let simdgroupWidth = max(multiplyBiasPipelineState.threadExecutionWidth, accumulatePipelineState.threadExecutionWidth)
    let threadsPerThreadgroup = MTLSize(
      width: simdgroupWidth * buildOptions.executionSIMDGroups,
      height: 1,
      depth: 1
    )

    let inputSliceElementCount = dimensions.inputHeight * dimensions.inputWidth * dimensions.inputChannels
    let weightSliceElementCount = dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * dimensions.outputChannels
    let outputSliceElementCount = dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    let batchInputSliceCount = dimensions.inputDepth * inputSliceElementCount
    let batchOutputSliceCount = dimensions.outputDepth * outputSliceElementCount

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<duplicatedCount {
      for n in 0..<dimensions.batchSize {
        let batchActivationBase = n * batchInputSliceCount
        let batchOutputBase = n * batchOutputSliceCount

        for oz in 0..<dimensions.outputDepth {
          let outputBase = UInt32(batchOutputBase + oz * outputSliceElementCount)

          for kd in 0..<dimensions.kernelDepth {
            let inputDepthIndex = oz * dimensions.strideZ + kd * dimensions.dilationZ
            let activationBase = UInt32(batchActivationBase + inputDepthIndex * inputSliceElementCount)
            let weightsBase = UInt32(kd * weightSliceElementCount)
            let pipelineState = kd == 0 ? multiplyBiasPipelineState : accumulatePipelineState

            computeCommandEncoder.setComputePipelineState(pipelineState)
            computeCommandEncoder.setBuffer(activationBuffer, offset: 0, index: 0)
            computeCommandEncoder.setBuffer(weightBuffer, offset: 0, index: 1)
            computeCommandEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
            computeCommandEncoder.setBuffer(biasBuffer, offset: 0, index: 6)

            var activationBaseVar = activationBase
            var weightsBaseVar = weightsBase
            var outputBaseVar = outputBase
            computeCommandEncoder.setBytes(&activationBaseVar, length: MemoryLayout<UInt32>.size, index: 3)
            computeCommandEncoder.setBytes(&weightsBaseVar, length: MemoryLayout<UInt32>.size, index: 4)
            computeCommandEncoder.setBytes(&outputBaseVar, length: MemoryLayout<UInt32>.size, index: 5)
            computeCommandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
          }
        }
      }
    }
    computeCommandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = DispatchTime.now().uptimeNanoseconds

    if commandBuffer.status != .completed {
      let errorDescription = commandBuffer.error.map { "\($0)" } ?? "none"
      fatalError("Conv3D bias tensor-op command buffer did not complete successfully. status=\(commandBuffer.status.rawValue), error=\(errorDescription)")
    }

    return ExecutionResult(
      output: readOutput ? readFloat16BufferAsFloat(outputBuffer, count: outputCount) : [],
      wallLatency: Double(end - start) / 1e9,
      gpuLatency: commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    )
  }

  func run(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: true)
  }

  func runWithoutOutputReadback(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: false)
  }

  var threadExecutionWidth: Int {
    multiplyBiasPipelineState.threadExecutionWidth
  }

  var threadsPerThreadgroup: Int {
    max(multiplyBiasPipelineState.threadExecutionWidth, accumulatePipelineState.threadExecutionWidth) * buildOptions.executionSIMDGroups
  }
}

final class WeightPermutationSession3D {
  private let dimensions: Conv3DDimensions
  private let commandQueue: MTLCommandQueue
  private let pipelineState: MTLComputePipelineState
  private let sourceBuffer: MTLBuffer
  private let destinationBuffer: MTLBuffer
  private let elementCount: Int

  init(
    device: MTLDevice,
    dimensions: Conv3DDimensions,
    sourceWeightsOIDHW: [Float16]
  ) throws {
    self.dimensions = dimensions

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 20,
        userInfo: [NSLocalizedDescriptionKey: "Could not create command queue for Conv3D weight permutation."]
      )
    }
    self.commandQueue = commandQueue
    self.commandQueue.label = "ConvolutionBench.Conv3DWeightPermutationQueue"

    let library = try device.makeLibrary(source: createConv3DWeightPermutationSource(), options: nil)
    guard let function = library.makeFunction(name: "permute_oidhw_to_dhwio") else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 21,
        userInfo: [NSLocalizedDescriptionKey: "Could not create Conv3D weight permutation function."]
      )
    }
    pipelineState = try device.makeComputePipelineState(function: function)

    elementCount = dimensions.kernelDepth * dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * dimensions.outputChannels
    sourceBuffer = copyToSharedBuffer(device: device, values: sourceWeightsOIDHW)
    destinationBuffer = makeZeroedSharedBuffer(device: device, count: elementCount)
  }

  private func execute(duplicatedCount: Int, readOutput: Bool) -> ExecutionResult {
    precondition(duplicatedCount > 0)

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create Conv3D weight permutation command buffer or encoder")
    }

    computeCommandEncoder.setComputePipelineState(pipelineState)
    computeCommandEncoder.setBuffer(sourceBuffer, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(destinationBuffer, offset: 0, index: 1)
    computeCommandEncoder.useResource(sourceBuffer, usage: .read)
    computeCommandEncoder.useResource(destinationBuffer, usage: .write)

    var outputChannels = UInt32(dimensions.outputChannels)
    var inputChannels = UInt32(dimensions.inputChannels)
    var kernelDepth = UInt32(dimensions.kernelDepth)
    var kernelHeight = UInt32(dimensions.kernelHeight)
    var kernelWidth = UInt32(dimensions.kernelWidth)
    computeCommandEncoder.setBytes(&outputChannels, length: MemoryLayout<UInt32>.size, index: 2)
    computeCommandEncoder.setBytes(&inputChannels, length: MemoryLayout<UInt32>.size, index: 3)
    computeCommandEncoder.setBytes(&kernelDepth, length: MemoryLayout<UInt32>.size, index: 4)
    computeCommandEncoder.setBytes(&kernelHeight, length: MemoryLayout<UInt32>.size, index: 5)
    computeCommandEncoder.setBytes(&kernelWidth, length: MemoryLayout<UInt32>.size, index: 6)

    let width = min(max(pipelineState.threadExecutionWidth, 1), pipelineState.maxTotalThreadsPerThreadgroup)
    let threadsPerThreadgroup = MTLSize(width: width, height: 1, depth: 1)
    let threads = MTLSize(width: elementCount, height: 1, depth: 1)

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<duplicatedCount {
      computeCommandEncoder.dispatchThreads(threads, threadsPerThreadgroup: threadsPerThreadgroup)
    }
    computeCommandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = DispatchTime.now().uptimeNanoseconds

    if commandBuffer.status != .completed {
      let errorDescription = commandBuffer.error.map { "\($0)" } ?? "none"
      fatalError("Conv3D weight permutation command buffer did not complete successfully. status=\(commandBuffer.status.rawValue), error=\(errorDescription)")
    }

    return ExecutionResult(
      output: readOutput ? readFloat16BufferAsFloat(destinationBuffer, count: elementCount) : [],
      wallLatency: Double(end - start) / 1e9,
      gpuLatency: commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    )
  }

  func run(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: true)
  }

  func runWithoutOutputReadback(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: false)
  }
}

final class PermuteThenTensorOpSession3D {
  private let dimensions: Conv3DDimensions
  private let spatialDimensions: Conv2DDimensions
  private let buildOptions: BuildOptions
  private let commandQueue: MTLCommandQueue
  private let permutationPipelineState: MTLComputePipelineState
  private let multiplyPipelineState: MTLComputePipelineState
  private let accumulatePipelineState: MTLComputePipelineState
  private let activationBuffer: MTLBuffer
  private let sourceWeightBuffer: MTLBuffer
  private let permutedWeightBuffer: MTLBuffer
  private let outputBuffer: MTLBuffer
  private let outputCount: Int
  private let permutedWeightCount: Int

  init(
    device: MTLDevice,
    dimensions: Conv3DDimensions,
    buildOptions: BuildOptions,
    activation: [Float16],
    sourceWeightsOIDHW: [Float16]
  ) throws {
    precondition(buildOptions.usesOutputTiling, "Permute+Conv3D session requires tiled build options.")

    self.dimensions = dimensions
    self.spatialDimensions = conv3DSpatialSliceDimensions(dimensions)
    self.buildOptions = buildOptions

    guard device.supportsFamily(.metal4) else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 22,
        userInfo: [NSLocalizedDescriptionKey: "This device does not support the tensor operations used in the Conv3D shader."]
      )
    }

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 23,
        userInfo: [NSLocalizedDescriptionKey: "Could not create command queue for permute+Conv3D tensor op."]
      )
    }
    self.commandQueue = commandQueue
    self.commandQueue.label = "ConvolutionBench.PermuteThenTensorOp3DQueue"

    let permutationLibrary = try device.makeLibrary(source: createConv3DWeightPermutationSource(), options: nil)
    guard let permutationFunction = permutationLibrary.makeFunction(name: "permute_oidhw_to_dhwio") else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 24,
        userInfo: [NSLocalizedDescriptionKey: "Could not create Conv3D weight permutation function."]
      )
    }
    permutationPipelineState = try device.makeComputePipelineState(function: permutationFunction)

    let convLibrary = try device.makeLibrary(
      source: createConv3DTensorOpSource(dimensions: dimensions, buildOptions: buildOptions),
      options: nil
    )
    guard let multiplyFunction = convLibrary.makeFunction(name: TensorOpConv3DMode.multiply.functionName()),
          let accumulateFunction = convLibrary.makeFunction(name: TensorOpConv3DMode.multiplyAccumulate.functionName()) else {
      throw NSError(
        domain: "ConvolutionHarness",
        code: 25,
        userInfo: [NSLocalizedDescriptionKey: "Could not create Conv3D tensor-op functions for permute+conv."]
      )
    }
    multiplyPipelineState = try device.makeComputePipelineState(function: multiplyFunction)
    accumulatePipelineState = try device.makeComputePipelineState(function: accumulateFunction)

    activationBuffer = copyToSharedBuffer(device: device, values: activation)
    permutedWeightCount = dimensions.kernelDepth * dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * dimensions.outputChannels
    sourceWeightBuffer = copyToSharedBuffer(device: device, values: sourceWeightsOIDHW)
    permutedWeightBuffer = makeZeroedSharedBuffer(device: device, count: permutedWeightCount)
    outputCount = dimensions.batchSize * dimensions.outputDepth * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    outputBuffer = makeZeroedSharedBuffer(device: device, count: outputCount)
  }

  private func execute(duplicatedCount: Int, readOutput: Bool) -> ExecutionResult {
    precondition(duplicatedCount > 0)

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create permute+Conv3D command buffer or encoder")
    }

    computeCommandEncoder.useResource(activationBuffer, usage: .read)
    computeCommandEncoder.useResource(sourceWeightBuffer, usage: .read)
    computeCommandEncoder.useResource(permutedWeightBuffer, usage: [.read, .write])
    computeCommandEncoder.useResource(outputBuffer, usage: [.read, .write])

    var outputChannels = UInt32(dimensions.outputChannels)
    var inputChannels = UInt32(dimensions.inputChannels)
    var kernelDepth = UInt32(dimensions.kernelDepth)
    var kernelHeight = UInt32(dimensions.kernelHeight)
    var kernelWidth = UInt32(dimensions.kernelWidth)
    let permutationThreadsPerThreadgroup = MTLSize(
      width: min(max(permutationPipelineState.threadExecutionWidth, 1), permutationPipelineState.maxTotalThreadsPerThreadgroup),
      height: 1,
      depth: 1
    )
    let permutationThreads = MTLSize(width: permutedWeightCount, height: 1, depth: 1)

    let convThreadgroups = tensorOpDispatchThreadgroups(dimensions: spatialDimensions, buildOptions: buildOptions)
    let convThreadsPerThreadgroup = MTLSize(
      width: multiplyPipelineState.threadExecutionWidth * buildOptions.executionSIMDGroups,
      height: 1,
      depth: 1
    )
    let inputSliceElementCount = dimensions.inputHeight * dimensions.inputWidth * dimensions.inputChannels
    let weightSliceElementCount = dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * dimensions.outputChannels
    let outputSliceElementCount = dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    let batchInputSliceCount = dimensions.inputDepth * inputSliceElementCount
    let batchOutputSliceCount = dimensions.outputDepth * outputSliceElementCount

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<duplicatedCount {
      computeCommandEncoder.setComputePipelineState(permutationPipelineState)
      computeCommandEncoder.setBuffer(sourceWeightBuffer, offset: 0, index: 0)
      computeCommandEncoder.setBuffer(permutedWeightBuffer, offset: 0, index: 1)
      computeCommandEncoder.setBytes(&outputChannels, length: MemoryLayout<UInt32>.size, index: 2)
      computeCommandEncoder.setBytes(&inputChannels, length: MemoryLayout<UInt32>.size, index: 3)
      computeCommandEncoder.setBytes(&kernelDepth, length: MemoryLayout<UInt32>.size, index: 4)
      computeCommandEncoder.setBytes(&kernelHeight, length: MemoryLayout<UInt32>.size, index: 5)
      computeCommandEncoder.setBytes(&kernelWidth, length: MemoryLayout<UInt32>.size, index: 6)
      computeCommandEncoder.dispatchThreads(permutationThreads, threadsPerThreadgroup: permutationThreadsPerThreadgroup)

      for n in 0..<dimensions.batchSize {
        let batchActivationBase = n * batchInputSliceCount
        let batchOutputBase = n * batchOutputSliceCount

        for oz in 0..<dimensions.outputDepth {
          let outputBase = UInt32(batchOutputBase + oz * outputSliceElementCount)

          for kd in 0..<dimensions.kernelDepth {
            let inputDepthIndex = oz * dimensions.strideZ + kd * dimensions.dilationZ
            let activationBase = UInt32(batchActivationBase + inputDepthIndex * inputSliceElementCount)
            let weightsBase = UInt32(kd * weightSliceElementCount)
            let pipelineState = kd == 0 ? multiplyPipelineState : accumulatePipelineState

            computeCommandEncoder.setComputePipelineState(pipelineState)
            computeCommandEncoder.setBuffer(activationBuffer, offset: 0, index: 0)
            computeCommandEncoder.setBuffer(permutedWeightBuffer, offset: 0, index: 1)
            computeCommandEncoder.setBuffer(outputBuffer, offset: 0, index: 2)

            var activationBaseVar = activationBase
            var weightsBaseVar = weightsBase
            var outputBaseVar = outputBase
            computeCommandEncoder.setBytes(&activationBaseVar, length: MemoryLayout<UInt32>.size, index: 3)
            computeCommandEncoder.setBytes(&weightsBaseVar, length: MemoryLayout<UInt32>.size, index: 4)
            computeCommandEncoder.setBytes(&outputBaseVar, length: MemoryLayout<UInt32>.size, index: 5)
            computeCommandEncoder.dispatchThreadgroups(convThreadgroups, threadsPerThreadgroup: convThreadsPerThreadgroup)
          }
        }
      }
    }
    computeCommandEncoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let end = DispatchTime.now().uptimeNanoseconds

    if commandBuffer.status != .completed {
      let errorDescription = commandBuffer.error.map { "\($0)" } ?? "none"
      fatalError("Permute+Conv3D command buffer did not complete successfully. status=\(commandBuffer.status.rawValue), error=\(errorDescription)")
    }

    return ExecutionResult(
      output: readOutput ? readFloat16BufferAsFloat(outputBuffer, count: outputCount) : [],
      wallLatency: Double(end - start) / 1e9,
      gpuLatency: commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    )
  }

  func run(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: true)
  }

  func runWithoutOutputReadback(duplicatedCount: Int) -> ExecutionResult {
    execute(duplicatedCount: duplicatedCount, readOutput: false)
  }
}

final class MPSGraphSession {
  private let dimensions: Conv2DDimensions
  private let commandQueue: MTLCommandQueue
  private let executable: MPSGraphExecutable
  private let activationData: MPSGraphTensorData
  private let weightsData: MPSGraphTensorData
  private let outputData: MPSGraphTensorData
  private let outputBuffer: MTLBuffer
  private let executionDescriptor: MPSGraphExecutableExecutionDescriptor
  private let outputCount: Int

  init(
    device: MTLDevice,
    dimensions: Conv2DDimensions,
    activation: [Float16],
    weights: [Float16],
    weightLayout: ConvolutionWeightLayout
  ) {
    self.dimensions = dimensions

    guard let commandQueue = device.makeCommandQueue() else {
      fatalError("Could not create command queue")
    }
    self.commandQueue = commandQueue
    self.commandQueue.label = "ConvolutionBench.MPSGraph2DQueue"

    let activationShape = makeShape([
      dimensions.batchSize,
      dimensions.inputHeight,
      dimensions.inputWidth,
      dimensions.inputChannels,
    ])
    let weightShape: [NSNumber]
    switch weightLayout {
    case .hwio:
      weightShape = makeShape([
        dimensions.kernelHeight,
        dimensions.kernelWidth,
        dimensions.inputChannels,
        dimensions.outputChannels,
      ])
    case .oihw:
      weightShape = makeShape([
        dimensions.outputChannels,
        dimensions.inputChannels,
        dimensions.kernelHeight,
        dimensions.kernelWidth,
      ])
    }
    let outputShape = makeShape([
      dimensions.batchSize,
      dimensions.outputHeight,
      dimensions.outputWidth,
      dimensions.outputChannels,
    ])

    let graph = MPSGraph()
    let activationTensor = graph.placeholder(shape: activationShape, dataType: .float16, name: "activation")
    let weightsTensor = graph.placeholder(shape: weightShape, dataType: .float16, name: "weights")

    let descriptor = MPSGraphConvolution2DOpDescriptor()
    descriptor.strideInX = dimensions.strideX
    descriptor.strideInY = dimensions.strideY
    descriptor.dilationRateInX = dimensions.dilationX
    descriptor.dilationRateInY = dimensions.dilationY
    descriptor.groups = 1
    descriptor.paddingLeft = 0
    descriptor.paddingRight = 0
    descriptor.paddingTop = 0
    descriptor.paddingBottom = 0
    descriptor.paddingStyle = .explicit
    descriptor.dataLayout = .NHWC
    switch weightLayout {
    case .hwio:
      descriptor.weightsLayout = .HWIO
    case .oihw:
      descriptor.weightsLayout = .OIHW
    }

    let outputTensor = graph.convolution2D(activationTensor, weights: weightsTensor, descriptor: descriptor, name: "conv2d")
    let feeds: [MPSGraphTensor: MPSGraphShapedType] = [
      activationTensor: MPSGraphShapedType(shape: activationShape, dataType: .float16),
      weightsTensor: MPSGraphShapedType(shape: weightShape, dataType: .float16),
    ]
    let compilationDescriptor = MPSGraphCompilationDescriptor()
    compilationDescriptor.optimizationLevel = .level0
    if #unavailable(iOS 17.0, macOS 14.0, tvOS 17.0) {
      compilationDescriptor.optimizationProfile = .performance
    }
    executable = graph.compile(
      with: nil,
      feeds: feeds,
      targetTensors: [outputTensor],
      targetOperations: nil,
      compilationDescriptor: compilationDescriptor
    )

    let activationBuffer = copyToSharedBuffer(device: device, values: activation)
    let weightBuffer = copyToSharedBuffer(device: device, values: weights)
    outputCount = dimensions.batchSize * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    outputBuffer = makeZeroedSharedBuffer(device: device, count: outputCount)

    activationData = MPSGraphTensorData(activationBuffer, shape: activationShape, dataType: .float16)
    weightsData = MPSGraphTensorData(weightBuffer, shape: weightShape, dataType: .float16)
    outputData = MPSGraphTensorData(outputBuffer, shape: outputShape, dataType: .float16)
    executionDescriptor = MPSGraphExecutableExecutionDescriptor()
    executionDescriptor.waitUntilCompleted = true
  }

  func run(duplicatedCount: Int) -> ExecutionResult {
    precondition(duplicatedCount > 0)

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<duplicatedCount {
      _ = executable.run(
        with: commandQueue,
        inputs: [activationData, weightsData],
        results: [outputData],
        executionDescriptor: executionDescriptor
      )
    }
    let end = DispatchTime.now().uptimeNanoseconds

    return ExecutionResult(
      output: readFloat16BufferAsFloat(outputBuffer, count: outputCount),
      wallLatency: Double(end - start) / 1e9,
      gpuLatency: nil
    )
  }

  func capture(duplicatedCount: Int, outputURL: URL) throws -> ExecutionResult {
    try runWithGPUTraceCapture(commandQueue: commandQueue, outputURL: outputURL) {
      run(duplicatedCount: duplicatedCount)
    }
  }
}

final class MPSGraphSession3D {
  private let dimensions: Conv3DDimensions
  private let commandQueue: MTLCommandQueue
  private let executable: MPSGraphExecutable
  private let activationData: MPSGraphTensorData
  private let weightsData: MPSGraphTensorData
  private let outputData: MPSGraphTensorData
  private let outputBuffer: MTLBuffer
  private let outputCount: Int
  private let executionDescriptor: MPSGraphExecutableExecutionDescriptor

  init(
    device: MTLDevice,
    dimensions: Conv3DDimensions,
    activation: [Float16],
    weights: [Float16],
    weightLayout: Convolution3DWeightLayout
  ) {
    self.dimensions = dimensions

    guard let commandQueue = device.makeCommandQueue() else {
      fatalError("Could not create command queue")
    }
    self.commandQueue = commandQueue

    let activationShape = makeShape([
      dimensions.batchSize,
      dimensions.inputDepth,
      dimensions.inputHeight,
      dimensions.inputWidth,
      dimensions.inputChannels,
    ])
    let weightShape: [NSNumber]
    switch weightLayout {
    case .dhwio:
      weightShape = makeShape([
        dimensions.kernelDepth,
        dimensions.kernelHeight,
        dimensions.kernelWidth,
        dimensions.inputChannels,
        dimensions.outputChannels,
      ])
    case .oidhw:
      weightShape = makeShape([
        dimensions.outputChannels,
        dimensions.inputChannels,
        dimensions.kernelDepth,
        dimensions.kernelHeight,
        dimensions.kernelWidth,
      ])
    }
    let outputShape = makeShape([
      dimensions.batchSize,
      dimensions.outputDepth,
      dimensions.outputHeight,
      dimensions.outputWidth,
      dimensions.outputChannels,
    ])

    let graph = MPSGraph()
    let activationTensor = graph.placeholder(shape: activationShape, dataType: .float16, name: "activation3d")
    let weightsTensor = graph.placeholder(shape: weightShape, dataType: .float16, name: "weights3d")

    let descriptor = MPSGraphConvolution3DOpDescriptor()
    descriptor.strideInX = dimensions.strideX
    descriptor.strideInY = dimensions.strideY
    descriptor.strideInZ = dimensions.strideZ
    descriptor.dilationRateInX = dimensions.dilationX
    descriptor.dilationRateInY = dimensions.dilationY
    descriptor.dilationRateInZ = dimensions.dilationZ
    descriptor.groups = 1
    descriptor.paddingLeft = 0
    descriptor.paddingRight = 0
    descriptor.paddingTop = 0
    descriptor.paddingBottom = 0
    descriptor.paddingFront = 0
    descriptor.paddingBack = 0
    descriptor.paddingStyle = .explicit
    descriptor.dataLayout = .NDHWC
    switch weightLayout {
    case .dhwio:
      descriptor.weightsLayout = .DHWIO
    case .oidhw:
      descriptor.weightsLayout = .OIDHW
    }

    let outputTensor = graph.convolution3D(activationTensor, weights: weightsTensor, descriptor: descriptor, name: "conv3d")
    let feeds: [MPSGraphTensor: MPSGraphShapedType] = [
      activationTensor: MPSGraphShapedType(shape: activationShape, dataType: .float16),
      weightsTensor: MPSGraphShapedType(shape: weightShape, dataType: .float16),
    ]
    let compilationDescriptor = MPSGraphCompilationDescriptor()
    compilationDescriptor.optimizationLevel = .level0
    if #unavailable(iOS 17.0, macOS 14.0, tvOS 17.0) {
      compilationDescriptor.optimizationProfile = .performance
    }
    executable = graph.compile(
      with: nil,
      feeds: feeds,
      targetTensors: [outputTensor],
      targetOperations: nil,
      compilationDescriptor: compilationDescriptor
    )

    let activationBuffer = copyToSharedBuffer(device: device, values: activation)
    let weightBuffer = copyToSharedBuffer(device: device, values: weights)
    outputCount = dimensions.batchSize * dimensions.outputDepth * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    outputBuffer = makeZeroedSharedBuffer(device: device, count: outputCount)

    activationData = MPSGraphTensorData(activationBuffer, shape: activationShape, dataType: .float16)
    weightsData = MPSGraphTensorData(weightBuffer, shape: weightShape, dataType: .float16)
    outputData = MPSGraphTensorData(outputBuffer, shape: outputShape, dataType: .float16)
    executionDescriptor = MPSGraphExecutableExecutionDescriptor()
    executionDescriptor.waitUntilCompleted = true
  }

  func run(duplicatedCount: Int) -> ExecutionResult {
    precondition(duplicatedCount > 0)

    let start = DispatchTime.now().uptimeNanoseconds
    for _ in 0..<duplicatedCount {
      _ = executable.run(
        with: commandQueue,
        inputs: [activationData, weightsData],
        results: [outputData],
        executionDescriptor: executionDescriptor
      )
    }
    let end = DispatchTime.now().uptimeNanoseconds

    return ExecutionResult(
      output: readFloat16BufferAsFloat(outputBuffer, count: outputCount),
      wallLatency: Double(end - start) / 1e9,
      gpuLatency: nil
    )
  }
}

public enum ConvolutionHarness {
  public static func gpuCaptureSupport() -> GPUCaptureSupportResult {
    let captureManager = MTLCaptureManager.shared()
    return GPUCaptureSupportResult(
      gpuTraceDocumentSupported: captureManager.supportsDestination(.gpuTraceDocument),
      developerToolsSupported: captureManager.supportsDestination(.developerTools)
    )
  }

  public static func defaultValidationCases() -> [Conv2DDimensions] {
    [
      Conv2DDimensions(
        batchSize: 1,
        inputHeight: 5,
        inputWidth: 5,
        inputChannels: 1,
        outputChannels: 1,
        kernelHeight: 3,
        kernelWidth: 3,
        strideY: 1,
        strideX: 1,
        dilationY: 1,
        dilationX: 1
      ),
      Conv2DDimensions(
        batchSize: 1,
        inputHeight: 5,
        inputWidth: 5,
        inputChannels: 4,
        outputChannels: 8,
        kernelHeight: 3,
        kernelWidth: 3,
        strideY: 1,
        strideX: 1,
        dilationY: 1,
        dilationX: 1
      ),
      Conv2DDimensions(
        batchSize: 1,
        inputHeight: 6,
        inputWidth: 7,
        inputChannels: 3,
        outputChannels: 5,
        kernelHeight: 3,
        kernelWidth: 3,
        strideY: 1,
        strideX: 1,
        dilationY: 1,
        dilationX: 1
      ),
      Conv2DDimensions(
        batchSize: 1,
        inputHeight: 7,
        inputWidth: 8,
        inputChannels: 3,
        outputChannels: 4,
        kernelHeight: 3,
        kernelWidth: 3,
        strideY: 2,
        strideX: 2,
        dilationY: 1,
        dilationX: 1
      ),
      Conv2DDimensions(
        batchSize: 1,
        inputHeight: 9,
        inputWidth: 10,
        inputChannels: 2,
        outputChannels: 3,
        kernelHeight: 3,
        kernelWidth: 3,
        strideY: 1,
        strideX: 1,
        dilationY: 2,
        dilationX: 2
      ),
    ]
  }

  public static func defaultProfileDimensions() -> Conv2DDimensions {
    Conv2DDimensions(
      batchSize: 1,
      inputHeight: 128,
      inputWidth: 128,
      inputChannels: 32,
      outputChannels: 32,
      kernelHeight: 3,
      kernelWidth: 3,
      strideY: 1,
      strideX: 1,
      dilationY: 1,
      dilationX: 1
    )
  }

  public static func largeMPSGraphProfileDimensions() -> Conv2DDimensions {
    Conv2DDimensions(
      batchSize: 1,
      inputHeight: 256,
      inputWidth: 256,
      inputChannels: 512,
      outputChannels: 512,
      kernelHeight: 3,
      kernelWidth: 3,
      strideY: 1,
      strideX: 1,
      dilationY: 1,
      dilationX: 1
    )
  }

  public static func largeMPSGraphValidationDimensions() -> Conv2DDimensions {
    defaultProfileDimensions()
  }

  public static func largeMPSGraphProfileOptions() -> ProfileOptions {
    ProfileOptions(warmupIterations: 1, timedIterations: 3, duplicatedCount: 1)
  }

  @discardableResult
  public static func debugTensorOpVariants(
    dimensionsList: [Conv2DDimensions] = defaultValidationCases(),
    buildOptions: BuildOptions = BuildOptions(executionSIMDGroups: 1),
    log: (String) -> Void = { _ in }
  ) -> TensorOpVariantSuiteResult {
    let device = makeDevice()
    var results: [TensorOpVariantResult] = []
    var firstPassingVariantName: String?

    for variant in TensorOpVariant.allCases {
      var caseResults: [TensorOpVariantCaseResult] = []
      var allPassed = true
      var threadExecutionWidth: Int?
      var threadsPerThreadgroup: Int?

      for dimensions in dimensionsList {
        let activation = createActivationData(dimensions: dimensions)
        let weights = createWeightData(dimensions: dimensions)
        let expected = referenceConvolution(activation: activation, weights: weights, dimensions: dimensions)

        do {
          let session = try TensorOpSession(
            device: device,
            dimensions: dimensions,
            buildOptions: buildOptions,
            activation: activation,
            weights: weights,
            variant: variant
          )
          threadExecutionWidth = session.threadExecutionWidth
          threadsPerThreadgroup = session.threadsPerThreadgroup

          let result = session.run(duplicatedCount: 1)
          let validation = validateOutput(
            actual: result.output,
            expected: expected,
            dimensions: dimensions,
            label: "Tensor op variant \(variant.name)",
            log: log
          )
          caseResults.append(
            TensorOpVariantCaseResult(
              dimensions: dimensions,
              validation: validation,
              wallLatencyMS: result.wallLatency.map { $0 * 1e3 },
              gpuLatencyMS: result.gpuLatency.map { $0 * 1e3 },
              error: nil
            )
          )
          if validation.mismatches > 0 {
            allPassed = false
          }
          log(
            "Tensor-op variant \(variant.name) on \(dimensions): max error \(validation.maxAbsoluteError), mismatches \(validation.mismatches)"
          )
        } catch {
          allPassed = false
          caseResults.append(
            TensorOpVariantCaseResult(
              dimensions: dimensions,
              validation: nil,
              wallLatencyMS: nil,
              gpuLatencyMS: nil,
              error: error.localizedDescription
            )
          )
          log("Tensor-op variant \(variant.name) failed on \(dimensions): \(error.localizedDescription)")
        }
      }

      if allPassed && firstPassingVariantName == nil {
        firstPassingVariantName = variant.name
      }

      let executionSIMDGroups = variant.executionSIMDGroupsOverride ?? buildOptions.executionSIMDGroups
      let scopeDescription: String
      switch variant.scopeKind {
      case .executionSimdgroups:
        scopeDescription = "execution_simdgroups"
      case .executionSimdgroup:
        scopeDescription = "execution_simdgroup"
      }

      let destinationDescription: String
      switch variant.destinationKind {
      case .direct:
        destinationDescription = "direct"
      case .cooperative:
        destinationDescription = "cooperative"
      }

      let outputElementType: String
      switch variant.outputElementKind {
      case .half:
        outputElementType = "half"
      case .float:
        outputElementType = "float"
      }

      results.append(
        TensorOpVariantResult(
          name: variant.name,
          scope: scopeDescription,
          destination: destinationDescription,
          outputElementType: outputElementType,
          usesOffsets: variant.usesOffsets,
          executionSIMDGroups: executionSIMDGroups,
          threadExecutionWidth: threadExecutionWidth,
          threadsPerThreadgroup: threadsPerThreadgroup,
          allPassed: allPassed,
          cases: caseResults
        )
      )
    }

    if let firstPassingVariantName {
      log("First fully passing tensor-op variant: \(firstPassingVariantName)")
    } else {
      log("No tensor-op debug variant fully passed.")
    }

    return TensorOpVariantSuiteResult(
      deviceName: device.name,
      firstPassingVariantName: firstPassingVariantName,
      variants: results
    )
  }

  @discardableResult
  public static func validateCorrectness(
    buildOptions: BuildOptions = BuildOptions(executionSIMDGroups: 1),
    log: (String) -> Void = { _ in }
  ) -> ConvolutionValidationSuiteResult {
    let device = makeDevice()
    var caseResults: [ConvolutionCaseResult] = []
    var allPassed = true

    for dimensions in defaultValidationCases() {
      let activation = createActivationData(dimensions: dimensions)
      let weights = createWeightData(dimensions: dimensions)
      let expected = referenceConvolution(activation: activation, weights: weights, dimensions: dimensions)

      let tensorOpSession = makeTensorOpSessionOrFatal(
        device: device,
        dimensions: dimensions,
        buildOptions: buildOptions,
        activation: activation,
        weights: weights
      )
      let mpsGraphSession = MPSGraphSession(
        device: device,
        dimensions: dimensions,
        activation: activation,
        weights: weights,
        weightLayout: .hwio
      )

      let tensorOpResult = tensorOpSession.run(duplicatedCount: 1)
      let mpsGraphResult = mpsGraphSession.run(duplicatedCount: 1)

      let tensorOpValidation = validateOutput(
        actual: tensorOpResult.output,
        expected: expected,
        dimensions: dimensions,
        label: "Tensor op",
        log: log
      )
      let mpsGraphValidation = validateOutput(
        actual: mpsGraphResult.output,
        expected: expected,
        dimensions: dimensions,
        label: "MPSGraph",
        log: log
      )
      let backendAgreement = compareOutputs(
        lhs: tensorOpResult.output,
        rhs: mpsGraphResult.output,
        dimensions: dimensions,
        lhsLabel: "Tensor op",
        rhsLabel: "MPSGraph",
        log: log
      )

      let caseResult = ConvolutionCaseResult(
        dimensions: dimensions,
        tensorOpValidation: tensorOpValidation,
        mpsGraphValidation: mpsGraphValidation,
        backendAgreement: backendAgreement,
        tensorOpWallLatencyMS: tensorOpResult.wallLatency.map { $0 * 1e3 },
        tensorOpGPULatencyMS: tensorOpResult.gpuLatency.map { $0 * 1e3 },
        mpsGraphWallLatencyMS: mpsGraphResult.wallLatency.map { $0 * 1e3 }
      )
      caseResults.append(caseResult)

      log("Validated \(dimensions)")
      log("Tensor op max absolute error: \(tensorOpValidation.maxAbsoluteError)")
      log("Tensor op mismatches above tolerance: \(tensorOpValidation.mismatches)")
      log("MPSGraph max absolute error: \(mpsGraphValidation.maxAbsoluteError)")
      log("MPSGraph mismatches above tolerance: \(mpsGraphValidation.mismatches)")
      log("Tensor op vs MPSGraph max absolute diff: \(backendAgreement.maxAbsoluteError)")
      log("Tensor op vs MPSGraph mismatches above tolerance: \(backendAgreement.mismatches)")
      printSingleRunThroughput(label: "Tensor op", dimensions: dimensions, result: tensorOpResult, duplicatedCount: 1, log: log)
      printSingleRunThroughput(label: "MPSGraph", dimensions: dimensions, result: mpsGraphResult, duplicatedCount: 1, log: log)

      if tensorOpValidation.mismatches > 0 || mpsGraphValidation.mismatches > 0 || backendAgreement.mismatches > 0 {
        allPassed = false
      }
    }

    log(allPassed ? "Convolution validation passed." : "Convolution validation failed.")
    return ConvolutionValidationSuiteResult(deviceName: device.name, allPassed: allPassed, cases: caseResults)
  }

  public static func profile(
    dimensions: Conv2DDimensions = defaultProfileDimensions(),
    buildOptions: BuildOptions = BuildOptions(executionSIMDGroups: 1),
    options: ProfileOptions = ProfileOptions(warmupIterations: 3, timedIterations: 10),
    log: (String) -> Void = { _ in }
  ) -> ConvolutionProfileResult {
    precondition(options.warmupIterations >= 0)
    precondition(options.timedIterations > 0)
    precondition(options.duplicatedCount > 0)

    let device = makeDevice()
    let activation = createActivationData(dimensions: dimensions)
    let weights = createWeightData(dimensions: dimensions)
    let expected = referenceConvolution(activation: activation, weights: weights, dimensions: dimensions)

    let tensorOpSession = makeTensorOpSessionOrFatal(
      device: device,
      dimensions: dimensions,
      buildOptions: buildOptions,
      activation: activation,
      weights: weights
    )
    let mpsGraphSession = MPSGraphSession(
      device: device,
      dimensions: dimensions,
      activation: activation,
      weights: weights,
      weightLayout: .hwio
    )

    let tensorOpReference = tensorOpSession.run(duplicatedCount: 1)
    let mpsGraphReference = mpsGraphSession.run(duplicatedCount: 1)
    let tensorOpValidation = validateOutput(
      actual: tensorOpReference.output,
      expected: expected,
      dimensions: dimensions,
      label: "Tensor op",
      log: log
    )
    let mpsGraphValidation = validateOutput(
      actual: mpsGraphReference.output,
      expected: expected,
      dimensions: dimensions,
      label: "MPSGraph",
      log: log
    )
    let backendAgreement = compareOutputs(
      lhs: tensorOpReference.output,
      rhs: mpsGraphReference.output,
      dimensions: dimensions,
      lhsLabel: "Tensor op",
      rhsLabel: "MPSGraph",
      log: log
    )

    for _ in 0..<options.warmupIterations {
      _ = tensorOpSession.run(duplicatedCount: options.duplicatedCount)
      _ = mpsGraphSession.run(duplicatedCount: options.duplicatedCount)
    }

    var tensorOpWallLatency: Double = 0
    var tensorOpGPULatency: Double = 0
    for _ in 0..<options.timedIterations {
      let result = tensorOpSession.run(duplicatedCount: options.duplicatedCount)
      tensorOpWallLatency += result.wallLatency ?? 0
      tensorOpGPULatency += result.gpuLatency ?? 0
    }

    var mpsGraphWallLatency: Double = 0
    for _ in 0..<options.timedIterations {
      let result = mpsGraphSession.run(duplicatedCount: options.duplicatedCount)
      mpsGraphWallLatency += result.wallLatency ?? 0
    }

    let averageTensorOpWallLatency = tensorOpWallLatency / Double(options.timedIterations)
    let averageTensorOpGPULatency = tensorOpGPULatency / Double(options.timedIterations)
    let averageMPSGraphWallLatency = mpsGraphWallLatency / Double(options.timedIterations)

    let tensorOpProfile = BackendProfileResult(
      averageWallLatencyMS: averageTensorOpWallLatency * 1e3,
      averageGPULatencyMS: averageTensorOpGPULatency * 1e3,
      throughputGFLOPS: throughputGFLOPS(dimensions: dimensions, latency: averageTensorOpWallLatency, duplicatedCount: options.duplicatedCount)
    )
    let mpsGraphProfile = BackendProfileResult(
      averageWallLatencyMS: averageMPSGraphWallLatency * 1e3,
      averageGPULatencyMS: nil,
      throughputGFLOPS: throughputGFLOPS(dimensions: dimensions, latency: averageMPSGraphWallLatency, duplicatedCount: options.duplicatedCount)
    )

    log("Profiled \(dimensions)")
    log("Profile options: warmup=\(options.warmupIterations), timed=\(options.timedIterations), duplicatedCount=\(options.duplicatedCount)")
    log(String(format: "Tensor op average wall latency: %.6f ms, average GPU latency: %.6f ms, throughput: %.3f GFLOP/s", tensorOpProfile.averageWallLatencyMS, tensorOpProfile.averageGPULatencyMS ?? 0, tensorOpProfile.throughputGFLOPS))
    log(String(format: "MPSGraph average wall latency: %.6f ms, throughput: %.3f GFLOP/s", mpsGraphProfile.averageWallLatencyMS, mpsGraphProfile.throughputGFLOPS))

    return ConvolutionProfileResult(
      dimensions: dimensions,
      options: options,
      tensorOpValidation: tensorOpValidation,
      mpsGraphValidation: mpsGraphValidation,
      backendAgreement: backendAgreement,
      tensorOp: tensorOpProfile,
      mpsGraph: mpsGraphProfile
    )
  }

  public static func profileTensorOpOnly(
    dimensions: Conv2DDimensions = defaultProfileDimensions(),
    buildOptions: BuildOptions = BuildOptions(executionSIMDGroups: 1),
    options: ProfileOptions = ProfileOptions(warmupIterations: 3, timedIterations: 10),
    log: (String) -> Void = { _ in }
  ) -> BackendProfileResult {
    precondition(options.warmupIterations >= 0)
    precondition(options.timedIterations > 0)
    precondition(options.duplicatedCount > 0)

    let device = makeDevice()
    let activation = createActivationData(dimensions: dimensions)
    let weights = createWeightData(dimensions: dimensions)

    let tensorOpSession = makeTensorOpSessionOrFatal(
      device: device,
      dimensions: dimensions,
      buildOptions: buildOptions,
      activation: activation,
      weights: weights
    )
    let dispatchThreadgroups = tensorOpDispatchThreadgroups(dimensions: dimensions, buildOptions: buildOptions)

    for _ in 0..<options.warmupIterations {
      _ = tensorOpSession.runWithoutOutputReadback(duplicatedCount: options.duplicatedCount)
    }

    var tensorOpWallLatency: Double = 0
    var tensorOpGPULatency: Double = 0
    for _ in 0..<options.timedIterations {
      let result = tensorOpSession.runWithoutOutputReadback(duplicatedCount: options.duplicatedCount)
      tensorOpWallLatency += result.wallLatency ?? 0
      tensorOpGPULatency += result.gpuLatency ?? 0
    }

    let averageTensorOpWallLatency = tensorOpWallLatency / Double(options.timedIterations)
    let averageTensorOpGPULatency = tensorOpGPULatency / Double(options.timedIterations)
    let wallThroughput = throughputGFLOPS(
      dimensions: dimensions,
      latency: averageTensorOpWallLatency,
      duplicatedCount: options.duplicatedCount
    )
    let gpuThroughput = throughputGFLOPS(
      dimensions: dimensions,
      latency: averageTensorOpGPULatency,
      duplicatedCount: options.duplicatedCount
    )

    log("Tensor-op only profile on \(dimensions)")
    if let outputTileWidth = buildOptions.outputTileWidth, let outputTileHeight = buildOptions.outputTileHeight {
      log("Build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(outputTileWidth)x\(outputTileHeight)")
    } else {
      log("Build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=full")
    }
    log(
      "Dispatch threadgroups: \(dispatchThreadgroups.width)x\(dispatchThreadgroups.height)x\(dispatchThreadgroups.depth), threadsPerThreadgroup: \(tensorOpSession.threadsPerThreadgroup)"
    )
    log("Profile options: warmup=\(options.warmupIterations), timed=\(options.timedIterations), duplicatedCount=\(options.duplicatedCount)")
    log(String(format: "Tensor op average wall latency: %.6f ms, throughput: %.3f GFLOP/s", averageTensorOpWallLatency * 1e3, wallThroughput))
    log(String(format: "Tensor op average GPU latency: %.6f ms, throughput: %.3f GFLOP/s", averageTensorOpGPULatency * 1e3, gpuThroughput))

    return BackendProfileResult(
      averageWallLatencyMS: averageTensorOpWallLatency * 1e3,
      averageGPULatencyMS: averageTensorOpGPULatency * 1e3,
      throughputGFLOPS: wallThroughput
    )
  }

  public static func profileTensorOpTileCoverage(
    dimensions: Conv2DDimensions = largeMPSGraphProfileDimensions(),
    tileWidth: Int,
    tileHeight: Int,
    buildOptions: BuildOptions = BuildOptions(executionSIMDGroups: 1),
    options: ProfileOptions = ProfileOptions(warmupIterations: 1, timedIterations: 1),
    log: (String) -> Void = { _ in }
  ) -> TensorOpTileProfileResult {
    precondition(tileWidth > 0 && tileHeight > 0)
    let tiledBuildOptions = BuildOptions(
      executionSIMDGroups: buildOptions.executionSIMDGroups,
      outputTileWidth: tileWidth,
      outputTileHeight: tileHeight
    )
    let dispatchThreadgroups = tensorOpDispatchThreadgroups(dimensions: dimensions, buildOptions: tiledBuildOptions)

    log("Tensor-op tiled coverage profile on \(dimensions)")
    log("Tile size: \(tileWidth)x\(tileHeight), executionSIMDGroups=\(buildOptions.executionSIMDGroups)")
    let profile = profileTensorOpOnly(
      dimensions: dimensions,
      buildOptions: tiledBuildOptions,
      options: options,
      log: log
    )
    let gpuThroughput = profile.averageGPULatencyMS.map {
      throughputGFLOPS(dimensions: dimensions, latency: $0 / 1e3, duplicatedCount: options.duplicatedCount)
    }

    return TensorOpTileProfileResult(
      dimensions: dimensions,
      tileWidth: tileWidth,
      tileHeight: tileHeight,
      options: options,
      regions: [
        TensorOpTileRegionProfileResult(
          label: "full-grid",
          buildOptions: tiledBuildOptions,
          dispatchGrid: DispatchGridSize(
            width: dispatchThreadgroups.width,
            height: dispatchThreadgroups.height,
            depth: dispatchThreadgroups.depth
          ),
          averageWallLatencyMS: profile.averageWallLatencyMS,
          averageGPULatencyMS: profile.averageGPULatencyMS
        )
      ],
      averageWallLatencyMS: profile.averageWallLatencyMS,
      averageGPULatencyMS: profile.averageGPULatencyMS,
      wallThroughputGFLOPS: profile.throughputGFLOPS,
      gpuThroughputGFLOPS: gpuThroughput
    )
  }

  public static func defaultTensorOpConv3DValidationCases() -> [Conv3DDimensions] {
    [
      Conv3DDimensions(
        batchSize: 1,
        inputDepth: 5,
        inputHeight: 5,
        inputWidth: 5,
        inputChannels: 1,
        outputChannels: 1,
        kernelDepth: 3,
        kernelHeight: 3,
        kernelWidth: 3,
        strideZ: 1,
        strideY: 1,
        strideX: 1,
        dilationZ: 1,
        dilationY: 1,
        dilationX: 1
      ),
      Conv3DDimensions(
        batchSize: 1,
        inputDepth: 4,
        inputHeight: 5,
        inputWidth: 5,
        inputChannels: 4,
        outputChannels: 8,
        kernelDepth: 3,
        kernelHeight: 3,
        kernelWidth: 3,
        strideZ: 1,
        strideY: 1,
        strideX: 1,
        dilationZ: 1,
        dilationY: 1,
        dilationX: 1
      ),
      Conv3DDimensions(
        batchSize: 2,
        inputDepth: 7,
        inputHeight: 6,
        inputWidth: 7,
        inputChannels: 3,
        outputChannels: 5,
        kernelDepth: 3,
        kernelHeight: 3,
        kernelWidth: 3,
        strideZ: 2,
        strideY: 1,
        strideX: 1,
        dilationZ: 1,
        dilationY: 1,
        dilationX: 1
      ),
      Conv3DDimensions(
        batchSize: 1,
        inputDepth: 7,
        inputHeight: 7,
        inputWidth: 7,
        inputChannels: 2,
        outputChannels: 3,
        kernelDepth: 3,
        kernelHeight: 3,
        kernelWidth: 3,
        strideZ: 1,
        strideY: 1,
        strideX: 1,
        dilationZ: 2,
        dilationY: 2,
        dilationX: 2
      ),
    ]
  }

  public static func defaultTensorOpConv3DBuildOptions() -> BuildOptions {
    BuildOptions(executionSIMDGroups: 4, outputTileWidth: 8, outputTileHeight: 8)
  }

  public static func defaultTensorOpConv3DLeftRightPaddingValidationCases() -> [Conv3DDimensions] {
    [
      Conv3DDimensions(
        batchSize: 1,
        inputDepth: 5,
        inputHeight: 5,
        inputWidth: 5,
        inputChannels: 1,
        outputChannels: 1,
        kernelDepth: 3,
        kernelHeight: 3,
        kernelWidth: 3,
        strideZ: 1,
        strideY: 1,
        strideX: 1,
        dilationZ: 1,
        dilationY: 1,
        dilationX: 1,
        paddingTop: 1,
        paddingBottom: 1,
        paddingLeft: 1,
        paddingRight: 1
      ),
      Conv3DDimensions(
        batchSize: 1,
        inputDepth: 4,
        inputHeight: 5,
        inputWidth: 5,
        inputChannels: 4,
        outputChannels: 8,
        kernelDepth: 3,
        kernelHeight: 3,
        kernelWidth: 3,
        strideZ: 1,
        strideY: 1,
        strideX: 1,
        dilationZ: 1,
        dilationY: 1,
        dilationX: 1,
        paddingTop: 1,
        paddingBottom: 1,
        paddingLeft: 1,
        paddingRight: 1
      ),
      Conv3DDimensions(
        batchSize: 2,
        inputDepth: 7,
        inputHeight: 6,
        inputWidth: 7,
        inputChannels: 3,
        outputChannels: 5,
        kernelDepth: 3,
        kernelHeight: 3,
        kernelWidth: 3,
        strideZ: 2,
        strideY: 1,
        strideX: 1,
        dilationZ: 1,
        dilationY: 1,
        dilationX: 1,
        paddingTop: 1,
        paddingBottom: 1,
        paddingLeft: 1,
        paddingRight: 1
      ),
      Conv3DDimensions(
        batchSize: 1,
        inputDepth: 5,
        inputHeight: 17,
        inputWidth: 19,
        inputChannels: 8,
        outputChannels: 12,
        kernelDepth: 3,
        kernelHeight: 3,
        kernelWidth: 3,
        strideZ: 1,
        strideY: 1,
        strideX: 1,
        dilationZ: 1,
        dilationY: 1,
        dilationX: 1,
        paddingTop: 1,
        paddingBottom: 1,
        paddingLeft: 1,
        paddingRight: 1
      ),
      Conv3DDimensions(
        batchSize: 1,
        inputDepth: 4,
        inputHeight: 33,
        inputWidth: 35,
        inputChannels: 4,
        outputChannels: 8,
        kernelDepth: 3,
        kernelHeight: 3,
        kernelWidth: 3,
        strideZ: 1,
        strideY: 1,
        strideX: 1,
        dilationZ: 1,
        dilationY: 1,
        dilationX: 1,
        paddingTop: 1,
        paddingBottom: 1,
        paddingLeft: 1,
        paddingRight: 1
      ),
    ]
  }

  public static func largeTensorOpConv3DProfileDimensions() -> Conv3DDimensions {
    Conv3DDimensions(
      batchSize: 1,
      inputDepth: 3,
      inputHeight: 256,
      inputWidth: 256,
      inputChannels: 512,
      outputChannels: 512,
      kernelDepth: 3,
      kernelHeight: 3,
      kernelWidth: 3,
      strideZ: 1,
      strideY: 1,
      strideX: 1,
      dilationZ: 1,
      dilationY: 1,
      dilationX: 1
    )
  }

  public static func largeTensorOpConv3DProfileOptions() -> ProfileOptions {
    ProfileOptions(warmupIterations: 2, timedIterations: 5, duplicatedCount: 1)
  }

  @discardableResult
  public static func validateTensorOpConv3D(
    dimensionsList: [Conv3DDimensions] = defaultTensorOpConv3DValidationCases(),
    buildOptions: BuildOptions = defaultTensorOpConv3DBuildOptions(),
    log: (String) -> Void = { _ in }
  ) -> Convolution3DValidationSuiteResult {
    let device = makeDevice()
    var caseResults: [Convolution3DCaseResult] = []
    var allPassed = true

    for dimensions in dimensionsList {
      let activation = createActivationData(dimensions: dimensions)
      let weights = createWeightData(dimensions: dimensions)
      let expected = referenceConvolution(activation: activation, weights: weights, dimensions: dimensions)

      let session: TensorOpHorizontalPaddingSession3D
      do {
        session = try TensorOpHorizontalPaddingSession3D(
          device: device,
          dimensions: dimensions,
          buildOptions: buildOptions,
          activation: activation,
          weights: weights
        )
      } catch {
        fatalError("Could not create Conv3D tensor-op session: \(error)")
      }

      let result = session.run(duplicatedCount: 1)
      let validation = validateOutput(
        actual: result.output,
        expected: expected,
        dimensions: dimensions,
        label: "Tensor-op Conv3D",
        log: log
      )
      if validation.mismatches > 0 {
        allPassed = false
      }
      caseResults.append(
        Convolution3DCaseResult(
          dimensions: dimensions,
          tensorOpValidation: validation,
          tensorOpWallLatencyMS: result.wallLatency.map { $0 * 1e3 },
          tensorOpGPULatencyMS: result.gpuLatency.map { $0 * 1e3 }
        )
      )
    }

    return Convolution3DValidationSuiteResult(
      deviceName: device.name,
      allPassed: allPassed,
      cases: caseResults
    )
  }

  @discardableResult
  public static func validateTensorOpConv3DLeftRightPadding(
    dimensionsList: [Conv3DDimensions] = defaultTensorOpConv3DLeftRightPaddingValidationCases(),
    buildOptions: BuildOptions = defaultTensorOpConv3DBuildOptions(),
    log: (String) -> Void = { _ in }
  ) -> Convolution3DValidationSuiteResult {
    let device = makeDevice()
    var caseResults: [Convolution3DCaseResult] = []
    var allPassed = true

    for dimensions in dimensionsList {
      let activation = createActivationData(dimensions: dimensions)
      let weights = createWeightData(dimensions: dimensions)
      let expected = referenceConvolution(activation: activation, weights: weights, dimensions: dimensions)

      let session: TensorOpHorizontalPaddingSession3D
      do {
        session = try TensorOpHorizontalPaddingSession3D(
          device: device,
          dimensions: dimensions,
          buildOptions: buildOptions,
          activation: activation,
          weights: weights
        )
      } catch {
        fatalError("Could not create left/right padded Conv3D tensor-op session: \(error)")
      }

      let result = session.run(duplicatedCount: 1)
      let validation = validateOutput(
        actual: result.output,
        expected: expected,
        dimensions: dimensions,
        label: "Tensor-op Conv3D left/right padding",
        log: log
      )
      if validation.mismatches > 0 {
        allPassed = false
      }
      caseResults.append(
        Convolution3DCaseResult(
          dimensions: dimensions,
          tensorOpValidation: validation,
          tensorOpWallLatencyMS: result.wallLatency.map { $0 * 1e3 },
          tensorOpGPULatencyMS: result.gpuLatency.map { $0 * 1e3 }
        )
      )
    }

    return Convolution3DValidationSuiteResult(
      deviceName: device.name,
      allPassed: allPassed,
      cases: caseResults
    )
  }

  @discardableResult
  public static func validateTensorOpConv3DBias(
    dimensionsList: [Conv3DDimensions] = defaultTensorOpConv3DValidationCases(),
    buildOptions: BuildOptions = defaultTensorOpConv3DBuildOptions(),
    log: (String) -> Void = { _ in }
  ) -> Convolution3DValidationSuiteResult {
    let device = makeDevice()
    var caseResults: [Convolution3DCaseResult] = []
    var allPassed = true

    for dimensions in dimensionsList {
      let activation = createActivationData(dimensions: dimensions)
      let weights = createWeightData(dimensions: dimensions)
      let bias = createBiasData(dimensions: dimensions)
      let expected = referenceConvolutionWithBias(
        activation: activation,
        weights: weights,
        bias: bias,
        dimensions: dimensions
      )

      let session: TensorOpBiasSession3D
      do {
        session = try TensorOpBiasSession3D(
          device: device,
          dimensions: dimensions,
          buildOptions: buildOptions,
          activation: activation,
          weights: weights,
          bias: bias
        )
      } catch {
        fatalError("Could not create Conv3D bias tensor-op session: \(error)")
      }

      let result = session.run(duplicatedCount: 1)
      let validation = validateOutput(
        actual: result.output,
        expected: expected,
        dimensions: dimensions,
        label: "Tensor-op Conv3D Bias",
        log: log
      )
      if validation.mismatches > 0 {
        allPassed = false
      }
      caseResults.append(
        Convolution3DCaseResult(
          dimensions: dimensions,
          tensorOpValidation: validation,
          tensorOpWallLatencyMS: result.wallLatency.map { $0 * 1e3 },
          tensorOpGPULatencyMS: result.gpuLatency.map { $0 * 1e3 }
        )
      )
    }

    return Convolution3DValidationSuiteResult(
      deviceName: device.name,
      allPassed: allPassed,
      cases: caseResults
    )
  }

  public static func profileTensorOpConv3D(
    dimensions: Conv3DDimensions = largeTensorOpConv3DProfileDimensions(),
    buildOptions: BuildOptions = defaultTensorOpConv3DBuildOptions(),
    options: ProfileOptions = largeTensorOpConv3DProfileOptions(),
    log: (String) -> Void = { _ in }
  ) -> TensorOpConv3DProfileResult {
    precondition(options.warmupIterations >= 0)
    precondition(options.timedIterations > 0)
    precondition(options.duplicatedCount > 0)

    let device = makeDevice()
    let activation = createActivationData(dimensions: dimensions)
    let weights = createWeightData(dimensions: dimensions)

    let session: TensorOpSession3D
    do {
      session = try TensorOpSession3D(
        device: device,
        dimensions: dimensions,
        buildOptions: buildOptions,
        activation: activation,
        weights: weights
      )
    } catch {
      fatalError("Could not create Conv3D tensor-op session: \(error)")
    }

    for _ in 0..<options.warmupIterations {
      _ = session.runWithoutOutputReadback(duplicatedCount: options.duplicatedCount)
    }

    var wallLatency: Double = 0
    var gpuLatency: Double = 0
    for _ in 0..<options.timedIterations {
      let result = session.runWithoutOutputReadback(duplicatedCount: options.duplicatedCount)
      wallLatency += result.wallLatency ?? 0
      gpuLatency += result.gpuLatency ?? 0
    }

    let averageWallLatency = wallLatency / Double(options.timedIterations)
    let averageGPULatency = gpuLatency / Double(options.timedIterations)
    let wallThroughput = throughputGFLOPS(
      dimensions: dimensions,
      latency: averageWallLatency,
      duplicatedCount: options.duplicatedCount
    )
    let gpuThroughput = throughputGFLOPS(
      dimensions: dimensions,
      latency: averageGPULatency,
      duplicatedCount: options.duplicatedCount
    )
    let dispatchThreadgroups = tensorOpDispatchThreadgroups(
      dimensions: conv3DSpatialSliceDimensions(dimensions),
      buildOptions: buildOptions
    )

    if let outputTileWidth = buildOptions.outputTileWidth, let outputTileHeight = buildOptions.outputTileHeight {
      log("Conv3D tensor-op build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(outputTileWidth)x\(outputTileHeight)")
    } else {
      log("Conv3D tensor-op build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=full")
    }
    log(
      "Conv3D dispatch threadgroups: \(dispatchThreadgroups.width)x\(dispatchThreadgroups.height)x\(dispatchThreadgroups.depth), threadsPerThreadgroup: \(session.threadsPerThreadgroup)"
    )
    log("Conv3D profile options: warmup=\(options.warmupIterations), timed=\(options.timedIterations), duplicatedCount=\(options.duplicatedCount)")
    log(String(format: "Conv3D tensor op average wall latency: %.6f ms, throughput: %.3f GFLOP/s", averageWallLatency * 1e3, wallThroughput))
    log(String(format: "Conv3D tensor op average GPU latency: %.6f ms, throughput: %.3f GFLOP/s", averageGPULatency * 1e3, gpuThroughput))

    return TensorOpConv3DProfileResult(
      dimensions: dimensions,
      buildOptions: buildOptions,
      options: options,
      dispatchGrid: DispatchGridSize(
        width: dispatchThreadgroups.width,
        height: dispatchThreadgroups.height,
        depth: dispatchThreadgroups.depth
      ),
      averageWallLatencyMS: averageWallLatency * 1e3,
      averageGPULatencyMS: averageGPULatency * 1e3,
      wallThroughputGFLOPS: wallThroughput,
      gpuThroughputGFLOPS: gpuThroughput
    )
  }

  public static func profileTensorOpConv3DBias(
    dimensions: Conv3DDimensions = largeTensorOpConv3DProfileDimensions(),
    buildOptions: BuildOptions = defaultTensorOpConv3DBuildOptions(),
    options: ProfileOptions = largeTensorOpConv3DProfileOptions(),
    log: (String) -> Void = { _ in }
  ) -> TensorOpConv3DProfileResult {
    precondition(options.warmupIterations >= 0)
    precondition(options.timedIterations > 0)
    precondition(options.duplicatedCount > 0)

    let device = makeDevice()
    let activation = createActivationData(dimensions: dimensions)
    let weights = createWeightData(dimensions: dimensions)
    let bias = createBiasData(dimensions: dimensions)

    let session: TensorOpBiasSession3D
    do {
      session = try TensorOpBiasSession3D(
        device: device,
        dimensions: dimensions,
        buildOptions: buildOptions,
        activation: activation,
        weights: weights,
        bias: bias
      )
    } catch {
      fatalError("Could not create Conv3D bias tensor-op session: \(error)")
    }

    for _ in 0..<options.warmupIterations {
      _ = session.runWithoutOutputReadback(duplicatedCount: options.duplicatedCount)
    }

    var wallLatency: Double = 0
    var gpuLatency: Double = 0
    for _ in 0..<options.timedIterations {
      let result = session.runWithoutOutputReadback(duplicatedCount: options.duplicatedCount)
      wallLatency += result.wallLatency ?? 0
      gpuLatency += result.gpuLatency ?? 0
    }

    let averageWallLatency = wallLatency / Double(options.timedIterations)
    let averageGPULatency = gpuLatency / Double(options.timedIterations)
    let wallThroughput = throughputGFLOPS(
      dimensions: dimensions,
      latency: averageWallLatency,
      duplicatedCount: options.duplicatedCount
    )
    let gpuThroughput = throughputGFLOPS(
      dimensions: dimensions,
      latency: averageGPULatency,
      duplicatedCount: options.duplicatedCount
    )
    let dispatchThreadgroups = tensorOpDispatchThreadgroups(
      dimensions: conv3DSpatialSliceDimensions(dimensions),
      buildOptions: buildOptions
    )

    if let outputTileWidth = buildOptions.outputTileWidth, let outputTileHeight = buildOptions.outputTileHeight {
      log("Conv3D bias tensor-op build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(outputTileWidth)x\(outputTileHeight)")
    } else {
      log("Conv3D bias tensor-op build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=full")
    }
    log(
      "Conv3D bias dispatch threadgroups: \(dispatchThreadgroups.width)x\(dispatchThreadgroups.height)x\(dispatchThreadgroups.depth), threadsPerThreadgroup: \(session.threadsPerThreadgroup)"
    )
    log("Conv3D bias profile options: warmup=\(options.warmupIterations), timed=\(options.timedIterations), duplicatedCount=\(options.duplicatedCount)")
    log(String(format: "Conv3D bias tensor op average wall latency: %.6f ms, throughput: %.3f GFLOP/s", averageWallLatency * 1e3, wallThroughput))
    log(String(format: "Conv3D bias tensor op average GPU latency: %.6f ms, throughput: %.3f GFLOP/s", averageGPULatency * 1e3, gpuThroughput))

    return TensorOpConv3DProfileResult(
      dimensions: dimensions,
      buildOptions: buildOptions,
      options: options,
      dispatchGrid: DispatchGridSize(
        width: dispatchThreadgroups.width,
        height: dispatchThreadgroups.height,
        depth: dispatchThreadgroups.depth
      ),
      averageWallLatencyMS: averageWallLatency * 1e3,
      averageGPULatencyMS: averageGPULatency * 1e3,
      wallThroughputGFLOPS: wallThroughput,
      gpuThroughputGFLOPS: gpuThroughput
    )
  }

  public static func defaultTensorOpConv3DPermutationValidationDimensions() -> Conv3DDimensions {
    defaultTensorOpConv3DValidationCases()[1]
  }

  public static func profileTensorOpConv3DWithPermutation(
    validationDimensions: Conv3DDimensions = defaultTensorOpConv3DPermutationValidationDimensions(),
    profileDimensions: Conv3DDimensions = largeTensorOpConv3DProfileDimensions(),
    buildOptions: BuildOptions = defaultTensorOpConv3DBuildOptions(),
    options: ProfileOptions = largeTensorOpConv3DProfileOptions(),
    log: (String) -> Void = { _ in }
  ) -> TensorOpConv3DPermutationProfileResult {
    precondition(options.warmupIterations >= 0)
    precondition(options.timedIterations > 0)
    precondition(options.duplicatedCount > 0)

    let device = makeDevice()

    let validationWeightsOIDHW = createWeightData(dimensions: validationDimensions, layout: .oidhw)
    let validationWeightsDHWIO = createWeightData(dimensions: validationDimensions, layout: .dhwio)
    let validationActivation = createActivationData(dimensions: validationDimensions)
    let validationExpected = referenceConvolution(
      activation: validationActivation,
      weights: validationWeightsDHWIO,
      dimensions: validationDimensions
    )

    let permutationValidationSession: WeightPermutationSession3D
    let combinedValidationSession: PermuteThenTensorOpSession3D
    do {
      permutationValidationSession = try WeightPermutationSession3D(
        device: device,
        dimensions: validationDimensions,
        sourceWeightsOIDHW: validationWeightsOIDHW
      )
      combinedValidationSession = try PermuteThenTensorOpSession3D(
        device: device,
        dimensions: validationDimensions,
        buildOptions: buildOptions,
        activation: validationActivation,
        sourceWeightsOIDHW: validationWeightsOIDHW
      )
    } catch {
      fatalError("Could not create Conv3D permutation sessions: \(error)")
    }

    let permutationValidationOutput = permutationValidationSession.run(duplicatedCount: 1)
    let permutationValidation = validateFlatOutput(
      actual: permutationValidationOutput.output,
      expected: validationWeightsDHWIO,
      label: "OIDHW->DHWIO permutation",
      log: log
    )
    let combinedValidationOutput = combinedValidationSession.run(duplicatedCount: 1)
    let combinedValidation = validateOutput(
      actual: combinedValidationOutput.output,
      expected: validationExpected,
      dimensions: validationDimensions,
      label: "Permute+Tensor-op Conv3D",
      log: log
    )

    let profileActivation = createActivationData(dimensions: profileDimensions)
    let profileWeightsOIDHW = createWeightData(dimensions: profileDimensions, layout: .oidhw)

    let permutationProfileSession: WeightPermutationSession3D
    let combinedProfileSession: PermuteThenTensorOpSession3D
    do {
      permutationProfileSession = try WeightPermutationSession3D(
        device: device,
        dimensions: profileDimensions,
        sourceWeightsOIDHW: profileWeightsOIDHW
      )
      combinedProfileSession = try PermuteThenTensorOpSession3D(
        device: device,
        dimensions: profileDimensions,
        buildOptions: buildOptions,
        activation: profileActivation,
        sourceWeightsOIDHW: profileWeightsOIDHW
      )
    } catch {
      fatalError("Could not create Conv3D permutation profile sessions: \(error)")
    }

    for _ in 0..<options.warmupIterations {
      _ = permutationProfileSession.runWithoutOutputReadback(duplicatedCount: options.duplicatedCount)
      _ = combinedProfileSession.runWithoutOutputReadback(duplicatedCount: options.duplicatedCount)
    }

    var permutationWallLatency: Double = 0
    var permutationGPULatency: Double = 0
    var combinedWallLatency: Double = 0
    var combinedGPULatency: Double = 0
    for _ in 0..<options.timedIterations {
      let permutationResult = permutationProfileSession.runWithoutOutputReadback(duplicatedCount: options.duplicatedCount)
      permutationWallLatency += permutationResult.wallLatency ?? 0
      permutationGPULatency += permutationResult.gpuLatency ?? 0

      let combinedResult = combinedProfileSession.runWithoutOutputReadback(duplicatedCount: options.duplicatedCount)
      combinedWallLatency += combinedResult.wallLatency ?? 0
      combinedGPULatency += combinedResult.gpuLatency ?? 0
    }

    let averagePermutationWallLatency = permutationWallLatency / Double(options.timedIterations)
    let averagePermutationGPULatency = permutationGPULatency / Double(options.timedIterations)
    let averageCombinedWallLatency = combinedWallLatency / Double(options.timedIterations)
    let averageCombinedGPULatency = combinedGPULatency / Double(options.timedIterations)

    let permutationWallBandwidth = bandwidthGBPS(
      dimensions: profileDimensions,
      latency: averagePermutationWallLatency,
      duplicatedCount: options.duplicatedCount
    )
    let permutationGPUBandwidth = bandwidthGBPS(
      dimensions: profileDimensions,
      latency: averagePermutationGPULatency,
      duplicatedCount: options.duplicatedCount
    )
    let combinedWallThroughput = throughputGFLOPS(
      dimensions: profileDimensions,
      latency: averageCombinedWallLatency,
      duplicatedCount: options.duplicatedCount
    )
    let combinedGPUThroughput = throughputGFLOPS(
      dimensions: profileDimensions,
      latency: averageCombinedGPULatency,
      duplicatedCount: options.duplicatedCount
    )

    log("Validated Conv3D weight permutation on \(validationDimensions)")
    log("Permutation max absolute error: \(permutationValidation.maxAbsoluteError)")
    log("Permutation mismatches above tolerance: \(permutationValidation.mismatches)")
    log("Permute+Conv3D max absolute error: \(combinedValidation.maxAbsoluteError)")
    log("Permute+Conv3D mismatches above tolerance: \(combinedValidation.mismatches)")
    if let outputTileWidth = buildOptions.outputTileWidth, let outputTileHeight = buildOptions.outputTileHeight {
      log("Permute+Conv3D build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(outputTileWidth)x\(outputTileHeight)")
    }
    log("Permute+Conv3D profile dimensions: \(profileDimensions)")
    log("Permute+Conv3D profile options: warmup=\(options.warmupIterations), timed=\(options.timedIterations), duplicatedCount=\(options.duplicatedCount)")
    log(String(format: "Permutation average wall latency: %.6f ms, bandwidth: %.3f GB/s", averagePermutationWallLatency * 1e3, permutationWallBandwidth))
    log(String(format: "Permutation average GPU latency: %.6f ms, bandwidth: %.3f GB/s", averagePermutationGPULatency * 1e3, permutationGPUBandwidth))
    log(String(format: "Permute+Conv3D average wall latency: %.6f ms, throughput: %.3f GFLOP/s", averageCombinedWallLatency * 1e3, combinedWallThroughput))
    log(String(format: "Permute+Conv3D average GPU latency: %.6f ms, throughput: %.3f GFLOP/s", averageCombinedGPULatency * 1e3, combinedGPUThroughput))

    return TensorOpConv3DPermutationProfileResult(
      validationDimensions: validationDimensions,
      profileDimensions: profileDimensions,
      buildOptions: buildOptions,
      options: options,
      permutationValidation: permutationValidation,
      combinedValidation: combinedValidation,
      permutationAverageWallLatencyMS: averagePermutationWallLatency * 1e3,
      permutationAverageGPULatencyMS: averagePermutationGPULatency * 1e3,
      permutationWallBandwidthGBPS: permutationWallBandwidth,
      permutationGPUBandwidthGBPS: permutationGPUBandwidth,
      combinedAverageWallLatencyMS: averageCombinedWallLatency * 1e3,
      combinedAverageGPULatencyMS: averageCombinedGPULatency * 1e3,
      combinedWallThroughputGFLOPS: combinedWallThroughput,
      combinedGPUThroughputGFLOPS: combinedGPUThroughput
    )
  }

  public static func profileMPSGraphWeightLayouts(
    validationDimensions: Conv2DDimensions? = nil,
    profileDimensions: Conv2DDimensions = defaultProfileDimensions(),
    options: ProfileOptions = ProfileOptions(warmupIterations: 5, timedIterations: 20, duplicatedCount: 20),
    log: (String) -> Void = { _ in }
  ) -> MPSGraphWeightLayoutProfileResult {
    precondition(options.warmupIterations >= 0)
    precondition(options.timedIterations > 0)
    precondition(options.duplicatedCount > 0)

    let device = makeDevice()
    let validationDimensions = validationDimensions ?? profileDimensions

    let validationActivation = createActivationData(dimensions: validationDimensions)
    let validationWeightsHWIO = createWeightData(dimensions: validationDimensions, layout: .hwio)
    let validationWeightsOIHW = createWeightData(dimensions: validationDimensions, layout: .oihw)
    let validationExpected = referenceConvolution(
      activation: validationActivation,
      weights: validationWeightsHWIO,
      dimensions: validationDimensions
    )

    let hwioSession = MPSGraphSession(
      device: device,
      dimensions: validationDimensions,
      activation: validationActivation,
      weights: validationWeightsHWIO,
      weightLayout: .hwio
    )
    let oihwSession = MPSGraphSession(
      device: device,
      dimensions: validationDimensions,
      activation: validationActivation,
      weights: validationWeightsOIHW,
      weightLayout: .oihw
    )

    let hwioReference = hwioSession.run(duplicatedCount: 1)
    let oihwReference = oihwSession.run(duplicatedCount: 1)
    let hwioValidation = validateOutput(
      actual: hwioReference.output,
      expected: validationExpected,
      dimensions: validationDimensions,
      label: "MPSGraph HWIO",
      log: log
    )
    let oihwValidation = validateOutput(
      actual: oihwReference.output,
      expected: validationExpected,
      dimensions: validationDimensions,
      label: "MPSGraph OIHW",
      log: log
    )
    let validationAgreement = compareOutputs(
      lhs: hwioReference.output,
      rhs: oihwReference.output,
      dimensions: validationDimensions,
      lhsLabel: "MPSGraph HWIO",
      rhsLabel: "MPSGraph OIHW",
      log: log
    )

    let profileActivation = createActivationData(dimensions: profileDimensions)
    let profileWeightsHWIO = createWeightData(dimensions: profileDimensions, layout: .hwio)
    let profileWeightsOIHW = createWeightData(dimensions: profileDimensions, layout: .oihw)

    let profileHWIOSession = MPSGraphSession(
      device: device,
      dimensions: profileDimensions,
      activation: profileActivation,
      weights: profileWeightsHWIO,
      weightLayout: .hwio
    )
    let profileOIHWSession = MPSGraphSession(
      device: device,
      dimensions: profileDimensions,
      activation: profileActivation,
      weights: profileWeightsOIHW,
      weightLayout: .oihw
    )

    let profileHWIOReference = profileHWIOSession.run(duplicatedCount: 1)
    let profileOIHWReference = profileOIHWSession.run(duplicatedCount: 1)
    let profileAgreement = compareOutputs(
      lhs: profileHWIOReference.output,
      rhs: profileOIHWReference.output,
      dimensions: profileDimensions,
      lhsLabel: "MPSGraph HWIO",
      rhsLabel: "MPSGraph OIHW",
      log: log
    )

    for _ in 0..<options.warmupIterations {
      _ = profileHWIOSession.run(duplicatedCount: options.duplicatedCount)
      _ = profileOIHWSession.run(duplicatedCount: options.duplicatedCount)
    }

    var hwioWallLatency: Double = 0
    for _ in 0..<options.timedIterations {
      let result = profileHWIOSession.run(duplicatedCount: options.duplicatedCount)
      hwioWallLatency += result.wallLatency ?? 0
    }

    var oihwWallLatency: Double = 0
    for _ in 0..<options.timedIterations {
      let result = profileOIHWSession.run(duplicatedCount: options.duplicatedCount)
      oihwWallLatency += result.wallLatency ?? 0
    }

    let averageHWIOWallLatency = hwioWallLatency / Double(options.timedIterations)
    let averageOIHWWallLatency = oihwWallLatency / Double(options.timedIterations)

    let hwioProfile = BackendProfileResult(
      averageWallLatencyMS: averageHWIOWallLatency * 1e3,
      averageGPULatencyMS: nil,
      throughputGFLOPS: throughputGFLOPS(dimensions: profileDimensions, latency: averageHWIOWallLatency, duplicatedCount: options.duplicatedCount)
    )
    let oihwProfile = BackendProfileResult(
      averageWallLatencyMS: averageOIHWWallLatency * 1e3,
      averageGPULatencyMS: nil,
      throughputGFLOPS: throughputGFLOPS(dimensions: profileDimensions, latency: averageOIHWWallLatency, duplicatedCount: options.duplicatedCount)
    )

    log("Validated MPSGraph weight layouts on \(validationDimensions)")
    log("MPSGraph HWIO max absolute error: \(hwioValidation.maxAbsoluteError)")
    log("MPSGraph HWIO mismatches above tolerance: \(hwioValidation.mismatches)")
    log("MPSGraph OIHW max absolute error: \(oihwValidation.maxAbsoluteError)")
    log("MPSGraph OIHW mismatches above tolerance: \(oihwValidation.mismatches)")
    log("MPSGraph HWIO vs OIHW validation diff: \(validationAgreement.maxAbsoluteError)")
    log("MPSGraph HWIO vs OIHW validation mismatches above tolerance: \(validationAgreement.mismatches)")
    log("MPSGraph HWIO vs OIHW profile diff on \(profileDimensions): \(profileAgreement.maxAbsoluteError)")
    log("MPSGraph HWIO vs OIHW profile mismatches above tolerance: \(profileAgreement.mismatches)")
    log("Profiled MPSGraph weight layouts for \(profileDimensions)")
    log("Profile options: warmup=\(options.warmupIterations), timed=\(options.timedIterations), duplicatedCount=\(options.duplicatedCount)")
    log(String(format: "MPSGraph HWIO average wall latency: %.6f ms, throughput: %.3f GFLOP/s", hwioProfile.averageWallLatencyMS, hwioProfile.throughputGFLOPS))
    log(String(format: "MPSGraph OIHW average wall latency: %.6f ms, throughput: %.3f GFLOP/s", oihwProfile.averageWallLatencyMS, oihwProfile.throughputGFLOPS))
    if oihwProfile.averageWallLatencyMS > 0 {
      log(String(format: "HWIO/OIHW wall latency ratio: %.3f", hwioProfile.averageWallLatencyMS / oihwProfile.averageWallLatencyMS))
    }

    return MPSGraphWeightLayoutProfileResult(
      validationDimensions: validationDimensions,
      profileDimensions: profileDimensions,
      options: options,
      hwioValidation: hwioValidation,
      oihwValidation: oihwValidation,
      validationAgreement: validationAgreement,
      profileAgreement: profileAgreement,
      hwio: hwioProfile,
      oihw: oihwProfile
    )
  }

  public static func captureGPUTraces(
    dimensions: Conv2DDimensions = defaultProfileDimensions(),
    buildOptions: BuildOptions = BuildOptions(executionSIMDGroups: 1),
    duplicatedCount: Int = 1,
    outputDirectoryURL: URL,
    log: (String) -> Void = { _ in }
  ) throws -> GPUTraceCaptureResult {
    precondition(duplicatedCount > 0)

    let device = makeDevice()
    let activation = createActivationData(dimensions: dimensions)
    let weights = createWeightData(dimensions: dimensions)

    let tensorOpSession = makeTensorOpSessionOrFatal(
      device: device,
      dimensions: dimensions,
      buildOptions: buildOptions,
      activation: activation,
      weights: weights
    )
    let mpsGraphSession = MPSGraphSession(
      device: device,
      dimensions: dimensions,
      activation: activation,
      weights: weights,
      weightLayout: .hwio
    )

    _ = tensorOpSession.run(duplicatedCount: 1)
    _ = mpsGraphSession.run(duplicatedCount: 1)

    let fileManager = FileManager.default
    try fileManager.createDirectory(at: outputDirectoryURL, withIntermediateDirectories: true)

    let tensorOpTraceURL = outputDirectoryURL.appendingPathComponent("tensorop-2d.gputrace", isDirectory: true)
    let mpsGraphTraceURL = outputDirectoryURL.appendingPathComponent("mpsgraph-2d.gputrace", isDirectory: true)

    _ = try tensorOpSession.capture(duplicatedCount: duplicatedCount, outputURL: tensorOpTraceURL)
    log("Saved tensor-op GPU trace to \(tensorOpTraceURL.path)")

    _ = try mpsGraphSession.capture(duplicatedCount: duplicatedCount, outputURL: mpsGraphTraceURL)
    log("Saved MPSGraph GPU trace to \(mpsGraphTraceURL.path)")

    return GPUTraceCaptureResult(
      dimensions: dimensions,
      duplicatedCount: duplicatedCount,
      tensorOpTracePath: tensorOpTraceURL.path,
      mpsGraphTracePath: mpsGraphTraceURL.path
    )
  }

  public static func defaultConv3DValidationDimensions() -> Conv3DDimensions {
    Conv3DDimensions(
      batchSize: 1,
      inputDepth: 5,
      inputHeight: 6,
      inputWidth: 7,
      inputChannels: 4,
      outputChannels: 6,
      kernelDepth: 3,
      kernelHeight: 3,
      kernelWidth: 3,
      strideZ: 1,
      strideY: 1,
      strideX: 1,
      dilationZ: 1,
      dilationY: 1,
      dilationX: 1
    )
  }

  public static func defaultConv3DProfileDimensions() -> Conv3DDimensions {
    Conv3DDimensions(
      batchSize: 1,
      inputDepth: 16,
      inputHeight: 64,
      inputWidth: 64,
      inputChannels: 32,
      outputChannels: 32,
      kernelDepth: 3,
      kernelHeight: 3,
      kernelWidth: 3,
      strideZ: 1,
      strideY: 1,
      strideX: 1,
      dilationZ: 1,
      dilationY: 1,
      dilationX: 1
    )
  }

  public static func largeConv3DProfileDimensions() -> Conv3DDimensions {
    Conv3DDimensions(
      batchSize: 1,
      inputDepth: 4,
      inputHeight: 128,
      inputWidth: 128,
      inputChannels: 512,
      outputChannels: 512,
      kernelDepth: 3,
      kernelHeight: 3,
      kernelWidth: 3,
      strideZ: 1,
      strideY: 1,
      strideX: 1,
      dilationZ: 1,
      dilationY: 1,
      dilationX: 1
    )
  }

  public static func largeConv3DProfileOptions() -> ProfileOptions {
    ProfileOptions(warmupIterations: 1, timedIterations: 3, duplicatedCount: 1)
  }

  public static func profileMPSGraphConv3DWeightLayouts(
    validationDimensions: Conv3DDimensions = defaultConv3DValidationDimensions(),
    profileDimensions: Conv3DDimensions = defaultConv3DProfileDimensions(),
    options: ProfileOptions = ProfileOptions(warmupIterations: 3, timedIterations: 10, duplicatedCount: 5),
    log: (String) -> Void = { _ in }
  ) -> MPSGraphConv3DWeightLayoutProfileResult {
    precondition(options.warmupIterations >= 0)
    precondition(options.timedIterations > 0)
    precondition(options.duplicatedCount > 0)

    let device = makeDevice()

    let validationActivation = createActivationData(dimensions: validationDimensions)
    let validationWeightsDHWIO = createWeightData(dimensions: validationDimensions, layout: .dhwio)
    let validationWeightsOIDHW = createWeightData(dimensions: validationDimensions, layout: .oidhw)
    let validationExpected = referenceConvolution(
      activation: validationActivation,
      weights: validationWeightsDHWIO,
      dimensions: validationDimensions
    )

    let validationDHWIOSession = MPSGraphSession3D(
      device: device,
      dimensions: validationDimensions,
      activation: validationActivation,
      weights: validationWeightsDHWIO,
      weightLayout: .dhwio
    )
    let validationOIDHWSession = MPSGraphSession3D(
      device: device,
      dimensions: validationDimensions,
      activation: validationActivation,
      weights: validationWeightsOIDHW,
      weightLayout: .oidhw
    )

    let validationDHWIOResult = validationDHWIOSession.run(duplicatedCount: 1)
    let validationOIDHWResult = validationOIDHWSession.run(duplicatedCount: 1)
    let dhwioValidation = validateOutput(
      actual: validationDHWIOResult.output,
      expected: validationExpected,
      dimensions: validationDimensions,
      label: "MPSGraph Conv3D DHWIO",
      log: log
    )
    let oidhwValidation = validateOutput(
      actual: validationOIDHWResult.output,
      expected: validationExpected,
      dimensions: validationDimensions,
      label: "MPSGraph Conv3D OIDHW",
      log: log
    )
    let validationAgreement = compareOutputs(
      lhs: validationDHWIOResult.output,
      rhs: validationOIDHWResult.output,
      dimensions: validationDimensions,
      lhsLabel: "MPSGraph Conv3D DHWIO",
      rhsLabel: "MPSGraph Conv3D OIDHW",
      log: log
    )

    let profileActivation = createActivationData(dimensions: profileDimensions)
    let profileWeightsDHWIO = createWeightData(dimensions: profileDimensions, layout: .dhwio)
    let profileWeightsOIDHW = createWeightData(dimensions: profileDimensions, layout: .oidhw)

    let profileDHWIOSession = MPSGraphSession3D(
      device: device,
      dimensions: profileDimensions,
      activation: profileActivation,
      weights: profileWeightsDHWIO,
      weightLayout: .dhwio
    )
    let profileOIDHWSession = MPSGraphSession3D(
      device: device,
      dimensions: profileDimensions,
      activation: profileActivation,
      weights: profileWeightsOIDHW,
      weightLayout: .oidhw
    )

    let profileDHWIOReference = profileDHWIOSession.run(duplicatedCount: 1)
    let profileOIDHWReference = profileOIDHWSession.run(duplicatedCount: 1)
    let profileAgreement = compareOutputs(
      lhs: profileDHWIOReference.output,
      rhs: profileOIDHWReference.output,
      dimensions: profileDimensions,
      lhsLabel: "MPSGraph Conv3D DHWIO",
      rhsLabel: "MPSGraph Conv3D OIDHW",
      log: log
    )

    for _ in 0..<options.warmupIterations {
      _ = profileDHWIOSession.run(duplicatedCount: options.duplicatedCount)
      _ = profileOIDHWSession.run(duplicatedCount: options.duplicatedCount)
    }

    var dhwioWallLatency: Double = 0
    for _ in 0..<options.timedIterations {
      let result = profileDHWIOSession.run(duplicatedCount: options.duplicatedCount)
      dhwioWallLatency += result.wallLatency ?? 0
    }

    var oidhwWallLatency: Double = 0
    for _ in 0..<options.timedIterations {
      let result = profileOIDHWSession.run(duplicatedCount: options.duplicatedCount)
      oidhwWallLatency += result.wallLatency ?? 0
    }

    let averageDHWIOWallLatency = dhwioWallLatency / Double(options.timedIterations)
    let averageOIDHWWallLatency = oidhwWallLatency / Double(options.timedIterations)

    let dhwioProfile = BackendProfileResult(
      averageWallLatencyMS: averageDHWIOWallLatency * 1e3,
      averageGPULatencyMS: nil,
      throughputGFLOPS: throughputGFLOPS(dimensions: profileDimensions, latency: averageDHWIOWallLatency, duplicatedCount: options.duplicatedCount)
    )
    let oidhwProfile = BackendProfileResult(
      averageWallLatencyMS: averageOIDHWWallLatency * 1e3,
      averageGPULatencyMS: nil,
      throughputGFLOPS: throughputGFLOPS(dimensions: profileDimensions, latency: averageOIDHWWallLatency, duplicatedCount: options.duplicatedCount)
    )

    log("Validated MPSGraph Conv3D layouts on \(validationDimensions)")
    log("Conv3D DHWIO max absolute error: \(dhwioValidation.maxAbsoluteError)")
    log("Conv3D DHWIO mismatches above tolerance: \(dhwioValidation.mismatches)")
    log("Conv3D OIDHW max absolute error: \(oidhwValidation.maxAbsoluteError)")
    log("Conv3D OIDHW mismatches above tolerance: \(oidhwValidation.mismatches)")
    log("Conv3D DHWIO vs OIDHW validation diff: \(validationAgreement.maxAbsoluteError)")
    log("Conv3D DHWIO vs OIDHW validation mismatches above tolerance: \(validationAgreement.mismatches)")
    log("Profiled MPSGraph Conv3D weight layouts for \(profileDimensions)")
    log("Profile options: warmup=\(options.warmupIterations), timed=\(options.timedIterations), duplicatedCount=\(options.duplicatedCount)")
    log("Conv3D profile DHWIO vs OIDHW max absolute diff: \(profileAgreement.maxAbsoluteError)")
    log("Conv3D profile DHWIO vs OIDHW mismatches above tolerance: \(profileAgreement.mismatches)")
    log(String(format: "MPSGraph Conv3D DHWIO average wall latency: %.6f ms, throughput: %.3f GFLOP/s", dhwioProfile.averageWallLatencyMS, dhwioProfile.throughputGFLOPS))
    log(String(format: "MPSGraph Conv3D OIDHW average wall latency: %.6f ms, throughput: %.3f GFLOP/s", oidhwProfile.averageWallLatencyMS, oidhwProfile.throughputGFLOPS))
    if oidhwProfile.averageWallLatencyMS > 0 {
      log(String(format: "Conv3D DHWIO/OIDHW wall latency ratio: %.3f", dhwioProfile.averageWallLatencyMS / oidhwProfile.averageWallLatencyMS))
    }

    return MPSGraphConv3DWeightLayoutProfileResult(
      validationDimensions: validationDimensions,
      profileDimensions: profileDimensions,
      options: options,
      dhwioValidation: dhwioValidation,
      oidhwValidation: oidhwValidation,
      validationAgreement: validationAgreement,
      profileAgreement: profileAgreement,
      dhwio: dhwioProfile,
      oidhw: oidhwProfile
    )
  }
}

func makeDevice() -> MTLDevice {
  guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device")
  }
  return device
}

struct TensorOpTileCoverageRegion {
  var label: String
  var buildOptions: BuildOptions
}

func offsetTensorOpTileCoverageRegions(
  outputWidth: Int,
  outputHeight: Int,
  tileWidth: Int,
  tileHeight: Int,
  executionSIMDGroups: Int,
  outputBaseX: Int,
  outputBaseY: Int
) -> [TensorOpTileCoverageRegion] {
  let fullTilesX = outputWidth / tileWidth
  let fullTilesY = outputHeight / tileHeight
  let remX = outputWidth % tileWidth
  let remY = outputHeight % tileHeight

  var regions: [TensorOpTileCoverageRegion] = []

  if fullTilesX > 0 && fullTilesY > 0 {
    regions.append(
      TensorOpTileCoverageRegion(
        label: "interior",
        buildOptions: BuildOptions(
          executionSIMDGroups: executionSIMDGroups,
          outputTileWidth: tileWidth,
          outputTileHeight: tileHeight,
          outputBaseX: outputBaseX,
          outputBaseY: outputBaseY,
          dispatchGridWidth: fullTilesX,
          dispatchGridHeight: fullTilesY
        )
      )
    )
  }
  if remX > 0 && fullTilesY > 0 {
    regions.append(
      TensorOpTileCoverageRegion(
        label: "right-edge",
        buildOptions: BuildOptions(
          executionSIMDGroups: executionSIMDGroups,
          outputTileWidth: remX,
          outputTileHeight: tileHeight,
          outputBaseX: outputBaseX + fullTilesX * tileWidth,
          outputBaseY: outputBaseY,
          dispatchGridWidth: 1,
          dispatchGridHeight: fullTilesY
        )
      )
    )
  }
  if remY > 0 && fullTilesX > 0 {
    regions.append(
      TensorOpTileCoverageRegion(
        label: "bottom-edge",
        buildOptions: BuildOptions(
          executionSIMDGroups: executionSIMDGroups,
          outputTileWidth: tileWidth,
          outputTileHeight: remY,
          outputBaseX: outputBaseX,
          outputBaseY: outputBaseY + fullTilesY * tileHeight,
          dispatchGridWidth: fullTilesX,
          dispatchGridHeight: 1
        )
      )
    )
  }
  if remX > 0 && remY > 0 {
    regions.append(
      TensorOpTileCoverageRegion(
        label: "corner",
        buildOptions: BuildOptions(
          executionSIMDGroups: executionSIMDGroups,
          outputTileWidth: remX,
          outputTileHeight: remY,
          outputBaseX: outputBaseX + fullTilesX * tileWidth,
          outputBaseY: outputBaseY + fullTilesY * tileHeight,
          dispatchGridWidth: 1,
          dispatchGridHeight: 1
        )
      )
    )
  }

  return regions
}

func tensorOpTileCoverageRegions(
  dimensions: Conv2DDimensions,
  tileWidth: Int,
  tileHeight: Int,
  executionSIMDGroups: Int
) -> [TensorOpTileCoverageRegion] {
  offsetTensorOpTileCoverageRegions(
    outputWidth: dimensions.outputWidth,
    outputHeight: dimensions.outputHeight,
    tileWidth: tileWidth,
    tileHeight: tileHeight,
    executionSIMDGroups: executionSIMDGroups,
    outputBaseX: 0,
    outputBaseY: 0
  )
}

func tensorOpDispatchThreadgroups(
  dimensions: Conv2DDimensions,
  buildOptions: BuildOptions
) -> MTLSize {
  if let dispatchGridWidth = buildOptions.dispatchGridWidth,
     let dispatchGridHeight = buildOptions.dispatchGridHeight {
    return MTLSize(width: dispatchGridWidth, height: dispatchGridHeight, depth: max(dimensions.batchSize, 1))
  }
  if let outputTileWidth = buildOptions.outputTileWidth,
     let outputTileHeight = buildOptions.outputTileHeight {
    return MTLSize(
      width: (dimensions.outputWidth + outputTileWidth - 1) / outputTileWidth,
      height: (dimensions.outputHeight + outputTileHeight - 1) / outputTileHeight,
      depth: max(dimensions.batchSize, 1)
    )
  }
  return MTLSize(width: 1, height: 1, depth: 1)
}

func makeTensorOpSessionOrFatal(
  device: MTLDevice,
  dimensions: Conv2DDimensions,
  buildOptions: BuildOptions,
  activation: [Float16],
  weights: [Float16],
  variant: TensorOpVariant = .cooperativeHalf
) -> TensorOpSession {
  do {
    return try TensorOpSession(
      device: device,
      dimensions: dimensions,
      buildOptions: buildOptions,
      activation: activation,
      weights: weights,
      variant: variant
    )
  } catch {
    fatalError("Could not create tensor-op session for \(variant.name): \(error)")
  }
}

func conv3DSpatialSliceDimensions(_ dimensions: Conv3DDimensions) -> Conv2DDimensions {
  Conv2DDimensions(
    batchSize: 1,
    inputHeight: dimensions.inputHeight,
    inputWidth: dimensions.inputWidth,
    inputChannels: dimensions.inputChannels,
    outputChannels: dimensions.outputChannels,
    kernelHeight: dimensions.kernelHeight,
    kernelWidth: dimensions.kernelWidth,
    strideY: dimensions.strideY,
    strideX: dimensions.strideX,
    dilationY: dimensions.dilationY,
    dilationX: dimensions.dilationX,
    paddingTop: dimensions.paddingTop,
    paddingBottom: dimensions.paddingBottom,
    paddingLeft: dimensions.paddingLeft,
    paddingRight: dimensions.paddingRight
  )
}

enum TensorOpConv3DMode {
  case multiply
  case multiplyAccumulate

  func functionName(withBias: Bool = false) -> String {
    switch self {
    case .multiply:
      return withBias ? "conv3d_multiply_bias" : "conv3d_multiply"
    case .multiplyAccumulate:
      return "conv3d_multiply_accumulate"
    }
  }

  var descriptorMode: String {
    switch self {
    case .multiply:
      return "multiply"
    case .multiplyAccumulate:
      return "multiply_accumulate"
    }
  }

  var destinationInitialization: String {
    switch self {
    case .multiply:
      return """
  #pragma clang loop unroll(full)
  for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
    if (cOutput.is_valid_element(i)) {
      cOutput[i] = 0;
    }
  }
"""
    case .multiplyAccumulate:
      return "  cOutput.load(output);"
    }
  }
}

func createConv3DHorizontalPaddingHelperSource(
  paddedDimensions: Conv3DDimensions,
  interiorDimensions: Conv3DDimensions
) -> String {
  """

#include <metal_stdlib>

using namespace metal;

kernel void copy_conv3d_horizontal_padding_interior(device const half *interior_output [[buffer(0)]],
                                                    device half *padded_output [[buffer(1)]],
                                                    uint gid [[thread_position_in_grid]])
{
  const uint elementCount = \(interiorDimensions.batchSize * interiorDimensions.outputDepth * interiorDimensions.outputHeight * interiorDimensions.outputWidth * interiorDimensions.outputChannels);
  if (gid >= elementCount) {
    return;
  }

  uint linear = gid;
  const uint oc = linear % \(interiorDimensions.outputChannels);
  linear /= \(interiorDimensions.outputChannels);
  const uint ow = linear % \(interiorDimensions.outputWidth);
  linear /= \(interiorDimensions.outputWidth);
  const uint oh = linear % \(interiorDimensions.outputHeight);
  linear /= \(interiorDimensions.outputHeight);
  const uint oz = linear % \(interiorDimensions.outputDepth);
  const uint n = linear / \(interiorDimensions.outputDepth);

  uint paddedIndex = n;
  paddedIndex = paddedIndex * \(paddedDimensions.outputDepth) + oz;
  paddedIndex = paddedIndex * \(paddedDimensions.outputHeight) + oh;
  paddedIndex = paddedIndex * \(paddedDimensions.outputWidth) + (ow + 1);
  paddedIndex = paddedIndex * \(paddedDimensions.outputChannels) + oc;
  padded_output[paddedIndex] = interior_output[gid];
}

kernel void conv3d_horizontal_padding_edges(device const half *activation [[buffer(0)]],
                                            device const half *weights [[buffer(1)]],
                                            device half *output [[buffer(2)]],
                                            uint gid [[thread_position_in_grid]])
{
  const uint elementCount = \(paddedDimensions.batchSize * paddedDimensions.outputDepth * paddedDimensions.outputHeight * 2 * paddedDimensions.outputChannels);
  if (gid >= elementCount) {
    return;
  }

  uint linear = gid;
  const uint oc = linear % \(paddedDimensions.outputChannels);
  linear /= \(paddedDimensions.outputChannels);
  const uint edge = linear % 2;
  linear /= 2;
  const uint oh = linear % \(paddedDimensions.outputHeight);
  linear /= \(paddedDimensions.outputHeight);
  const uint oz = linear % \(paddedDimensions.outputDepth);
  const uint n = linear / \(paddedDimensions.outputDepth);

  const int ow = edge == 0 ? 0 : \(paddedDimensions.outputWidth - 1);
  half accumulator = half(0.0h);

  for (int kd = 0; kd < \(paddedDimensions.kernelDepth); ++kd) {
    const int iz = int(oz) * \(paddedDimensions.strideZ) + kd * \(paddedDimensions.dilationZ);
    for (int kh = 0; kh < \(paddedDimensions.kernelHeight); ++kh) {
      const int ih = int(oh) * \(paddedDimensions.strideY) + kh * \(paddedDimensions.dilationY);
      for (int kw = 0; kw < \(paddedDimensions.kernelWidth); ++kw) {
        const int iw = ow * \(paddedDimensions.strideX) + kw * \(paddedDimensions.dilationX) - \(paddedDimensions.paddingLeft);
        if (iz < 0 || iz >= \(paddedDimensions.inputDepth) || ih < 0 || ih >= \(paddedDimensions.inputHeight) || iw < 0 || iw >= \(paddedDimensions.inputWidth)) {
          continue;
        }
        uint activationBase = n;
        activationBase = activationBase * \(paddedDimensions.inputDepth) + uint(iz);
        activationBase = activationBase * \(paddedDimensions.inputHeight) + uint(ih);
        activationBase = activationBase * \(paddedDimensions.inputWidth) + uint(iw);
        activationBase = activationBase * \(paddedDimensions.inputChannels);
        uint weightBase = uint(kd);
        weightBase = weightBase * \(paddedDimensions.kernelHeight) + uint(kh);
        weightBase = weightBase * \(paddedDimensions.kernelWidth) + uint(kw);
        weightBase = weightBase * \(paddedDimensions.inputChannels) * \(paddedDimensions.outputChannels) + oc;
        for (uint ic = 0; ic < \(paddedDimensions.inputChannels); ++ic) {
          accumulator += activation[activationBase + ic] * weights[weightBase + ic * \(paddedDimensions.outputChannels)];
        }
      }
    }
  }

  uint outputIndex = n;
  outputIndex = outputIndex * \(paddedDimensions.outputDepth) + oz;
  outputIndex = outputIndex * \(paddedDimensions.outputHeight) + oh;
  outputIndex = outputIndex * \(paddedDimensions.outputWidth) + uint(ow);
  outputIndex = outputIndex * \(paddedDimensions.outputChannels) + oc;
  output[outputIndex] = accumulator;
}
"""
}

func createConv3DHorizontalPaddingRegionSource(
  dimensions: Conv3DDimensions,
  region: TensorOpConv3DHorizontalPaddingRegion
) -> String {
  guard let outputTileWidth = region.buildOptions.outputTileWidth,
        let outputTileHeight = region.buildOptions.outputTileHeight else {
    fatalError("Conv3D horizontal padding tensor-op source requires tiled build options.")
  }

  let spatialDimensions = conv3DSpatialSliceDimensions(dimensions)
  let inputTileWidth =
    (outputTileWidth - 1) * dimensions.strideX + (region.kernelSliceXWidth - 1) * dimensions.dilationX + 1
  let inputTileHeight =
    (outputTileHeight - 1) * dimensions.strideY + (dimensions.kernelHeight - 1) * dimensions.dilationY + 1
  let scopeType = "execution_simdgroups<\(region.buildOptions.executionSIMDGroups)>"
  let outputBaseX = region.buildOptions.outputBaseX ?? 0
  let outputBaseY = region.buildOptions.outputBaseY ?? 0
  let offsetSetup =
    "  conv2d_op.set_offsets(int2(\(region.offsetX), \(region.offsetY)));"

  func destinationSetup(outputTensorSetup: String, mode: TensorOpConv3DMode) -> String {
    """
\(outputTensorSetup)
  auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
\(mode.destinationInitialization)
  conv2d_op.run(activation, weights, cOutput);
  #pragma clang loop unroll(full)
  for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
    if (cOutput.is_valid_element(i)) {
      auto idx = cOutput.get_multidimensional_index(i);
      const uint outputIndex =
          ((uint(physical_output_origin_y + idx[2]) * uint(\(spatialDimensions.outputWidth))
            + uint(physical_output_origin_x + idx[1]))
           * uint(\(spatialDimensions.outputChannels)))
          + uint(idx[0]);
      output_buf[outputIndex] = cOutput[i];
    }
  }
"""
  }

  func createKernel(mode: TensorOpConv3DMode) -> String {
    let tiledStaticOutputSetup = """
  auto output = output_base.slice<\(dimensions.outputChannels), \(outputTileWidth), \(outputTileHeight), 1>(
      0,
      physical_output_origin_x,
      physical_output_origin_y,
      0);
"""
    let tiledDynamicOutputSetup = """
  auto output = output_base.slice(
      0,
      physical_output_origin_x,
      physical_output_origin_y,
      0);
"""

    return """
kernel void \(mode.functionName())(device half *activation_buf [[buffer(0)]],
                                 device half *weights_buf [[buffer(1)]],
                                 device half *output_buf [[buffer(2)]],
                                 constant uint& activation_base [[buffer(3)]],
                                 constant uint& weights_base_offset [[buffer(4)]],
                                 constant uint& output_base_offset [[buffer(5)]],
                                 uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]])
{
  activation_buf += activation_base;
  weights_buf += weights_base_offset;
  output_buf += output_base_offset;

  const int logical_output_origin_x = int(threadgroup_position_in_grid.x) * \(outputTileWidth);
  const int logical_output_origin_y = int(threadgroup_position_in_grid.y) * \(outputTileHeight);
  if (logical_output_origin_x >= \(region.logicalOutputWidth) || logical_output_origin_y >= \(region.logicalOutputHeight)) {
    return;
  }

  const int physical_output_origin_x = \(outputBaseX) + logical_output_origin_x;
  const int physical_output_origin_y = \(outputBaseY) + logical_output_origin_y;
  const int input_origin_x = \(region.inputBaseX) + logical_output_origin_x * \(spatialDimensions.strideX);
  const int input_origin_y = \(region.inputBaseY) + logical_output_origin_y * \(spatialDimensions.strideY);

  auto activation_base_tensor = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(\(spatialDimensions.inputChannels), \(spatialDimensions.inputWidth), \(spatialDimensions.inputHeight), 1));
  auto output_base = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      output_buf,
      dextents<int32_t, 4>(\(spatialDimensions.outputChannels), \(spatialDimensions.outputWidth), \(spatialDimensions.outputHeight), 1));
  auto weights_base = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      weights_buf,
      dextents<int32_t, 4>(\(spatialDimensions.outputChannels), \(spatialDimensions.inputChannels), \(spatialDimensions.kernelWidth), \(spatialDimensions.kernelHeight)));
  auto weights = weights_base.slice<\(spatialDimensions.outputChannels), \(spatialDimensions.inputChannels), \(region.kernelSliceXWidth), \(spatialDimensions.kernelHeight)>(
      0,
      0,
      \(region.kernelSliceXStart),
      0);
  constexpr auto descriptor = convolution2d_descriptor(
      int4(\(spatialDimensions.outputChannels), \(outputTileWidth), \(outputTileHeight), 1),
      int4(\(spatialDimensions.inputChannels), \(inputTileWidth), \(inputTileHeight), 1),
      int2(\(region.kernelSliceXWidth), \(spatialDimensions.kernelHeight)),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(\(spatialDimensions.strideX), \(spatialDimensions.strideY)),
      int2(\(spatialDimensions.dilationX), \(spatialDimensions.dilationY)),
      1,
      false,
      convolution2d_descriptor::mode::\(mode.descriptorMode));
  convolution2d<descriptor, \(scopeType)> conv2d_op;
\(offsetSetup)

  if (logical_output_origin_x + \(outputTileWidth) <= \(region.logicalOutputWidth) &&
      logical_output_origin_y + \(outputTileHeight) <= \(region.logicalOutputHeight)) {
    auto activation = activation_base_tensor.slice<\(spatialDimensions.inputChannels), \(inputTileWidth), \(inputTileHeight), 1>(
        0,
        input_origin_x,
        input_origin_y,
        0);
\(destinationSetup(outputTensorSetup: tiledStaticOutputSetup, mode: mode))
  } else {
    auto activation = activation_base_tensor.slice(
        0,
        input_origin_x,
        input_origin_y,
        0);
\(destinationSetup(outputTensorSetup: tiledDynamicOutputSetup, mode: mode))
  }
}
"""
  }

  return """

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

\(createKernel(mode: .multiply))
\(createKernel(mode: .multiplyAccumulate))
"""
}

func createReadableConv3DHorizontalPaddingSource(
  dimensions: Conv3DDimensions,
  buildOptions: BuildOptions
) -> String {
  guard let outputTileWidth = buildOptions.outputTileWidth,
        let outputTileHeight = buildOptions.outputTileHeight else {
    fatalError("Readable Conv3D horizontal padding source requires tiled build options.")
  }

  let spatialDimensions = conv3DSpatialSliceDimensions(dimensions)
  let inputTileWidth =
    (outputTileWidth - 1) * spatialDimensions.strideX
    + (spatialDimensions.kernelWidth - 1) * spatialDimensions.dilationX + 1
  let inputTileHeight =
    (outputTileHeight - 1) * spatialDimensions.strideY
    + (spatialDimensions.kernelHeight - 1) * spatialDimensions.dilationY + 1
  let baseOffsetX = (spatialDimensions.kernelWidth - 1) * spatialDimensions.dilationX / 2
  let baseOffsetY = (spatialDimensions.kernelHeight - 1) * spatialDimensions.dilationY / 2
  let scopeType = "execution_simdgroups<\(buildOptions.executionSIMDGroups)>"

  func destinationSetup(mode: TensorOpConv3DMode) -> String {
    let initialization: String
    if mode == .multiply {
      initialization = """
  #pragma clang loop unroll(full)
  for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
    if (cOutput.is_valid_element(i)) {
      cOutput[i] = 0;
    }
  }
"""
    } else {
      initialization = """
  #pragma clang loop unroll(full)
  for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
    if (cOutput.is_valid_element(i)) {
      auto idx = cOutput.get_multidimensional_index(i);
      const int ox = output_origin_x + int(idx[1]);
      const int oy = output_origin_y + int(idx[2]);
      if (ox < \(spatialDimensions.outputWidth) && oy < \(spatialDimensions.outputHeight)) {
        cOutput[i] = output_buf[((oy * \(spatialDimensions.outputWidth) + ox) * \(spatialDimensions.outputChannels)) + idx[0]];
      } else {
        cOutput[i] = 0;
      }
    }
  }
"""
    }

    return """
  auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
\(initialization)
  conv2d_op.run(activation, weights, cOutput);
  #pragma clang loop unroll(full)
  for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
    if (cOutput.is_valid_element(i)) {
      auto idx = cOutput.get_multidimensional_index(i);
      const int ox = output_origin_x + int(idx[1]);
      const int oy = output_origin_y + int(idx[2]);
      if (ox < \(spatialDimensions.outputWidth) && oy < \(spatialDimensions.outputHeight)) {
        output_buf[((oy * \(spatialDimensions.outputWidth) + ox) * \(spatialDimensions.outputChannels)) + idx[0]] = cOutput[i];
      }
    }
  }
"""
  }

  func createKernel(mode: TensorOpConv3DMode) -> String {
    return """
kernel void \(mode.functionName())(device half *activation_buf [[buffer(0)]],
                                   device half *weights_buf [[buffer(1)]],
                                   device half *output_buf [[buffer(2)]],
                                   constant uint& activation_base [[buffer(3)]],
                                   constant uint& weights_base_offset [[buffer(4)]],
                                   constant uint& output_base_offset [[buffer(5)]],
                                   uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]])
{
  activation_buf += activation_base;
  weights_buf += weights_base_offset;
  output_buf += output_base_offset;

  const int output_origin_x = int(threadgroup_position_in_grid.x) * \(outputTileWidth);
  const int output_origin_y = int(threadgroup_position_in_grid.y) * \(outputTileHeight);
  if (output_origin_x >= \(spatialDimensions.outputWidth) || output_origin_y >= \(spatialDimensions.outputHeight)) {
    return;
  }

  const int unclamped_input_origin_x = output_origin_x * \(spatialDimensions.strideX) - \(spatialDimensions.paddingLeft);
  const int unclamped_input_origin_y = output_origin_y * \(spatialDimensions.strideY) - \(spatialDimensions.paddingTop);
  const int clamped_input_origin_x = max(0, min(unclamped_input_origin_x, max(0, \(spatialDimensions.inputWidth - inputTileWidth))));
  const int clamped_input_origin_y = max(0, min(unclamped_input_origin_y, max(0, \(spatialDimensions.inputHeight - inputTileHeight))));
  const int adjusted_offset_x = \(baseOffsetX) + (unclamped_input_origin_x - clamped_input_origin_x);
  const int adjusted_offset_y = \(baseOffsetY) + (unclamped_input_origin_y - clamped_input_origin_y);

  auto activation_base_tensor = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(\(spatialDimensions.inputChannels), \(spatialDimensions.inputWidth), \(spatialDimensions.inputHeight), 1));
  auto weights = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      weights_buf,
      dextents<int32_t, 4>(\(spatialDimensions.outputChannels), \(spatialDimensions.inputChannels), \(spatialDimensions.kernelWidth), \(spatialDimensions.kernelHeight)));

  constexpr auto descriptor = convolution2d_descriptor(
      int4(\(spatialDimensions.outputChannels), \(outputTileWidth), \(outputTileHeight), 1),
      int4(\(spatialDimensions.inputChannels), \(inputTileWidth), \(inputTileHeight), 1),
      int2(\(spatialDimensions.kernelWidth), \(spatialDimensions.kernelHeight)),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(\(spatialDimensions.strideX), \(spatialDimensions.strideY)),
      int2(\(spatialDimensions.dilationX), \(spatialDimensions.dilationY)),
      1,
      false,
      convolution2d_descriptor::mode::\(mode.descriptorMode));
  convolution2d<descriptor, \(scopeType)> conv2d_op;
  conv2d_op.set_offsets(int2(adjusted_offset_x, adjusted_offset_y));

  if (output_origin_x + \(outputTileWidth) <= \(spatialDimensions.outputWidth) &&
      output_origin_y + \(outputTileHeight) <= \(spatialDimensions.outputHeight)) {
    auto activation = activation_base_tensor.slice<\(spatialDimensions.inputChannels), \(inputTileWidth), \(inputTileHeight), 1>(
        0,
        clamped_input_origin_x,
        clamped_input_origin_y,
        0);
\(destinationSetup(mode: mode))
  } else {
    auto activation = activation_base_tensor.slice(
        0,
        clamped_input_origin_x,
        clamped_input_origin_y,
        0);
\(destinationSetup(mode: mode))
  }
}
"""
  }

  return """

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

\(createKernel(mode: .multiply))
\(createKernel(mode: .multiplyAccumulate))
"""
}

func createConv3DTensorOpSource(
  dimensions: Conv3DDimensions,
  buildOptions: BuildOptions,
  includeBias: Bool = false,
  kernelSliceXStart: Int = 0,
  kernelSliceXWidth: Int? = nil
) -> String {
  guard let outputTileWidth = buildOptions.outputTileWidth,
        let outputTileHeight = buildOptions.outputTileHeight else {
    fatalError("Conv3D tensor-op source currently requires tiled build options.")
  }

  let spatialDimensions = conv3DSpatialSliceDimensions(dimensions)
  let slicedKernelWidth = kernelSliceXWidth ?? dimensions.kernelWidth
  let inputTileWidth = (outputTileWidth - 1) * dimensions.strideX + (slicedKernelWidth - 1) * dimensions.dilationX + 1
  let inputTileHeight = (outputTileHeight - 1) * dimensions.strideY + (dimensions.kernelHeight - 1) * dimensions.dilationY + 1
  let scopeType = "execution_simdgroups<\(buildOptions.executionSIMDGroups)>"
  let outputBaseX = buildOptions.outputBaseX ?? 0
  let outputBaseY = buildOptions.outputBaseY ?? 0
  let offsetSetup =
    "  conv2d_op.set_offsets(int2(\(((slicedKernelWidth - 1) * dimensions.dilationX) / 2), \(((dimensions.kernelHeight - 1) * dimensions.dilationY) / 2)));"

  func destinationSetup(outputTensorSetup: String, mode: TensorOpConv3DMode, withBias: Bool) -> String {
    let biasAdd: String
    if withBias {
      biasAdd = """
  auto biasOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
  biasOutput.load(bias);
  #pragma clang loop unroll(full)
  for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
    if (cOutput.is_valid_element(i) && biasOutput.is_valid_element(i)) {
      cOutput[i] += biasOutput[i];
    }
  }
"""
    } else {
      biasAdd = ""
    }

    return """
\(outputTensorSetup)
  auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
\(mode.destinationInitialization)
  conv2d_op.run(activation, weights, cOutput);
\(biasAdd)
  cOutput.store(output);
"""
  }

  func createKernel(mode: TensorOpConv3DMode, withBias: Bool) -> String {
    let tiledStaticOutputSetup = """
  auto output = output_base.slice<\(dimensions.outputChannels), \(outputTileWidth), \(outputTileHeight), 1>(
      0,
      output_origin_x,
      output_origin_y,
      0);
"""
    let tiledDynamicOutputSetup = """
  auto output = output_base.slice(
      0,
      output_origin_x,
      output_origin_y,
      0);
"""
    let biasArguments = withBias ? ",\n                                 device half *bias_buf [[buffer(6)]]" : ""
    let biasTensorSetup: String
    if withBias {
      biasTensorSetup = """
  auto bias = tensor<device half, dextents<int32_t, 1>, tensor_inline>(
      bias_buf,
      dextents<int32_t, 1>(\(spatialDimensions.outputChannels)));
"""
    } else {
      biasTensorSetup = ""
    }

    return """
kernel void \(mode.functionName(withBias: withBias))(device half *activation_buf [[buffer(0)]],
                                 device half *weights_buf [[buffer(1)]],
                                 device half *output_buf [[buffer(2)]],
                                 constant uint& activation_base [[buffer(3)]],
                                 constant uint& weights_base_offset [[buffer(4)]],
                                 constant uint& output_base_offset [[buffer(5)]]\(biasArguments),
                                 uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]])
{
  activation_buf += activation_base;
  weights_buf += weights_base_offset;
  output_buf += output_base_offset;

  const int output_origin_x = \(outputBaseX) + int(threadgroup_position_in_grid.x) * \(outputTileWidth);
  const int output_origin_y = \(outputBaseY) + int(threadgroup_position_in_grid.y) * \(outputTileHeight);
  if (output_origin_x >= \(spatialDimensions.outputWidth) || output_origin_y >= \(spatialDimensions.outputHeight)) {
    return;
  }

  const int input_origin_x = output_origin_x * \(spatialDimensions.strideX) - \(spatialDimensions.paddingLeft) + \(kernelSliceXStart * dimensions.dilationX);
  const int input_origin_y = output_origin_y * \(spatialDimensions.strideY) - \(spatialDimensions.paddingTop);

  auto activation_base_tensor = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(\(spatialDimensions.inputChannels), \(spatialDimensions.inputWidth), \(spatialDimensions.inputHeight), 1));
  auto output_base = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      output_buf,
      dextents<int32_t, 4>(\(spatialDimensions.outputChannels), \(spatialDimensions.outputWidth), \(spatialDimensions.outputHeight), 1));
\(biasTensorSetup)
  auto weights_base = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      weights_buf,
      dextents<int32_t, 4>(\(spatialDimensions.outputChannels), \(spatialDimensions.inputChannels), \(spatialDimensions.kernelWidth), \(spatialDimensions.kernelHeight)));
  auto weights = weights_base.slice<\(spatialDimensions.outputChannels), \(spatialDimensions.inputChannels), \(slicedKernelWidth), \(spatialDimensions.kernelHeight)>(
      0,
      0,
      \(kernelSliceXStart),
      0);
  constexpr auto descriptor = convolution2d_descriptor(
      int4(\(spatialDimensions.outputChannels), \(outputTileWidth), \(outputTileHeight), 1),
      int4(\(spatialDimensions.inputChannels), \(inputTileWidth), \(inputTileHeight), 1),
      int2(\(slicedKernelWidth), \(spatialDimensions.kernelHeight)),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(\(spatialDimensions.strideX), \(spatialDimensions.strideY)),
      int2(\(spatialDimensions.dilationX), \(spatialDimensions.dilationY)),
      1,
      false,
      convolution2d_descriptor::mode::\(mode.descriptorMode));
  convolution2d<descriptor, \(scopeType)> conv2d_op;
\(offsetSetup)

  if (output_origin_x + \(outputTileWidth) <= \(spatialDimensions.outputWidth) &&
      output_origin_y + \(outputTileHeight) <= \(spatialDimensions.outputHeight)) {
    auto activation = activation_base_tensor.slice<\(spatialDimensions.inputChannels), \(inputTileWidth), \(inputTileHeight), 1>(
        0,
        input_origin_x,
        input_origin_y,
        0);
\(destinationSetup(outputTensorSetup: tiledStaticOutputSetup, mode: mode, withBias: withBias))
  } else {
    auto activation = activation_base_tensor.slice(
        0,
        input_origin_x,
        input_origin_y,
        0);
\(destinationSetup(outputTensorSetup: tiledDynamicOutputSetup, mode: mode, withBias: withBias))
  }
}
"""
  }

  return """

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

\(createKernel(mode: .multiply, withBias: false))
\(createKernel(mode: .multiplyAccumulate, withBias: false))
\(includeBias ? createKernel(mode: .multiply, withBias: true) : "")
"""
}

func createConv3DWeightPermutationSource() -> String {
  """

#include <metal_stdlib>

using namespace metal;

kernel void permute_oidhw_to_dhwio(device const half *source [[buffer(0)]],
                                   device half *destination [[buffer(1)]],
                                   constant uint &outputChannels [[buffer(2)]],
                                   constant uint &inputChannels [[buffer(3)]],
                                   constant uint &kernelDepth [[buffer(4)]],
                                   constant uint &kernelHeight [[buffer(5)]],
                                   constant uint &kernelWidth [[buffer(6)]],
                                   uint gid [[thread_position_in_grid]])
{
  const uint elementCount = outputChannels * inputChannels * kernelDepth * kernelHeight * kernelWidth;
  if (gid >= elementCount) {
    return;
  }

  uint linear = gid;
  const uint oc = linear % outputChannels;
  linear /= outputChannels;
  const uint ic = linear % inputChannels;
  linear /= inputChannels;
  const uint kw = linear % kernelWidth;
  linear /= kernelWidth;
  const uint kh = linear % kernelHeight;
  linear /= kernelHeight;
  const uint kd = linear;

  const uint sourceIndex = ((((oc * inputChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth) + kw;
  destination[gid] = source[sourceIndex];
}
"""
}

func createSource(dimensions: Conv2DDimensions, buildOptions: BuildOptions, variant: TensorOpVariant = .baseline) -> String {
  let outputElementType: String
  switch variant.outputElementKind {
  case .half:
    outputElementType = "half"
  case .float:
    outputElementType = "float"
  }

  let scopeType: String
  switch variant.scopeKind {
  case .executionSimdgroups:
    scopeType = "execution_simdgroups<\(buildOptions.executionSIMDGroups)>"
  case .executionSimdgroup:
    scopeType = "execution_simdgroup"
  }

  let offsetSetup: String
  if variant.usesOffsets {
    offsetSetup =
      "  conv2d_op.set_offsets(int2(\((dimensions.kernelWidth - 1) * dimensions.dilationX / 2), \((dimensions.kernelHeight - 1) * dimensions.dilationY / 2)));"
  } else {
    offsetSetup = ""
  }

  func destinationSetup(outputTensorSetup: String) -> String {
    switch variant.destinationKind {
    case .direct:
      return """
\(outputTensorSetup)
  conv2d_op.run(activation, weights, output);
"""
    case .cooperative:
      return """
\(outputTensorSetup)
  auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), \(outputElementType)>();
  #pragma clang loop unroll(full)
  for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
    if (cOutput.is_valid_element(i)) {
      cOutput[i] = 0;
    }
  }
  conv2d_op.run(activation, weights, cOutput);
  cOutput.store(output);
"""
    }
  }

  let activationSetup: String
  let outputSetupPrefix: String
  let kernelArguments: String
  let descriptor: String

  if let outputTileWidth = buildOptions.outputTileWidth,
     let outputTileHeight = buildOptions.outputTileHeight {
    let inputTileWidth = (outputTileWidth - 1) * dimensions.strideX + (dimensions.kernelWidth - 1) * dimensions.dilationX + 1
    let inputTileHeight = (outputTileHeight - 1) * dimensions.strideY + (dimensions.kernelHeight - 1) * dimensions.dilationY + 1

    kernelArguments = """
device half *activation_buf [[buffer(0)]],
                   device half *weights_buf [[buffer(1)]],
                   device \(outputElementType) *output_buf [[buffer(2)]],
                   uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
"""

    let tiledStaticOutputSetup = """
  auto output = output_base.slice<\(dimensions.outputChannels), \(outputTileWidth), \(outputTileHeight), 1>(
      0,
      output_origin_x,
      output_origin_y,
      batch_index);
"""
    let tiledDynamicOutputSetup = """
  auto output = output_base.slice(
      0,
      output_origin_x,
      output_origin_y,
      batch_index);
"""
    let tiledStaticDestinationSetup = destinationSetup(outputTensorSetup: tiledStaticOutputSetup)
    let tiledDynamicDestinationSetup = destinationSetup(outputTensorSetup: tiledDynamicOutputSetup)

    return """

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void conv2d(\(kernelArguments))
{
  const int output_origin_x = int(threadgroup_position_in_grid.x) * \(outputTileWidth);
  const int output_origin_y = int(threadgroup_position_in_grid.y) * \(outputTileHeight);
  const int batch_index = int(threadgroup_position_in_grid.z);
  if (output_origin_x >= \(dimensions.outputWidth) || output_origin_y >= \(dimensions.outputHeight) || batch_index >= \(dimensions.batchSize)) {
    return;
  }

  const int input_origin_x = output_origin_x * \(dimensions.strideX);
  const int input_origin_y = output_origin_y * \(dimensions.strideY);

  auto activation_base = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(\(dimensions.inputChannels), \(dimensions.inputWidth), \(dimensions.inputHeight), \(dimensions.batchSize)));
  auto output_base = tensor<device \(outputElementType), dextents<int32_t, 4>, tensor_inline>(
      output_buf,
      dextents<int32_t, 4>(\(dimensions.outputChannels), \(dimensions.outputWidth), \(dimensions.outputHeight), \(dimensions.batchSize)));
  auto weights = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      weights_buf,
      dextents<int32_t, 4>(\(dimensions.outputChannels), \(dimensions.inputChannels), \(dimensions.kernelWidth), \(dimensions.kernelHeight)));
  constexpr auto descriptor = convolution2d_descriptor(
      int4(\(dimensions.outputChannels), \(outputTileWidth), \(outputTileHeight), 1),
      int4(\(dimensions.inputChannels), \(inputTileWidth), \(inputTileHeight), 1),
      int2(\(dimensions.kernelWidth), \(dimensions.kernelHeight)),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(\(dimensions.strideX), \(dimensions.strideY)),
      int2(\(dimensions.dilationX), \(dimensions.dilationY)),
      1,
      false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<descriptor, \(scopeType)> conv2d_op;
\(offsetSetup)

  if (output_origin_x + \(outputTileWidth) <= \(dimensions.outputWidth) &&
      output_origin_y + \(outputTileHeight) <= \(dimensions.outputHeight)) {
    auto activation = activation_base.slice<\(dimensions.inputChannels), \(inputTileWidth), \(inputTileHeight), 1>(
        0,
        input_origin_x,
        input_origin_y,
        batch_index);
\(tiledStaticDestinationSetup)
  } else {
    auto activation = activation_base.slice(
        0,
        input_origin_x,
        input_origin_y,
        batch_index);
\(tiledDynamicDestinationSetup)
  }
}

"""
  } else {
    kernelArguments = """
device half *activation_buf [[buffer(0)]],
                   device half *weights_buf [[buffer(1)]],
                   device \(outputElementType) *output_buf [[buffer(2)]]
"""

    activationSetup = """
  auto activation = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(\(dimensions.inputChannels), \(dimensions.inputWidth), \(dimensions.inputHeight), \(dimensions.batchSize)));
"""

    outputSetupPrefix = """
  auto output = tensor<device \(outputElementType), dextents<int32_t, 4>, tensor_inline>(
      output_buf,
      dextents<int32_t, 4>(\(dimensions.outputChannels), \(dimensions.outputWidth), \(dimensions.outputHeight), \(dimensions.batchSize)));
"""

    descriptor = """
  constexpr auto descriptor = convolution2d_descriptor(
      int4(\(dimensions.outputChannels), \(dimensions.outputWidth), \(dimensions.outputHeight), \(dimensions.batchSize)),
      int4(\(dimensions.inputChannels), \(dimensions.inputWidth), \(dimensions.inputHeight), \(dimensions.batchSize)),
      int2(\(dimensions.kernelWidth), \(dimensions.kernelHeight)),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(\(dimensions.strideX), \(dimensions.strideY)),
      int2(\(dimensions.dilationX), \(dimensions.dilationY)),
      1,
      false,
      convolution2d_descriptor::mode::multiply);
"""
  }

  let destinationSetup = destinationSetup(outputTensorSetup: outputSetupPrefix)

  return """

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void conv2d(\(kernelArguments))
{
\(activationSetup)
  auto weights = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      weights_buf,
      dextents<int32_t, 4>(\(dimensions.outputChannels), \(dimensions.inputChannels), \(dimensions.kernelWidth), \(dimensions.kernelHeight)));

\(descriptor)

  convolution2d<descriptor, \(scopeType)> conv2d_op;
\(offsetSetup)
\(destinationSetup)
}

"""
}

func activationIndex(
  n: Int,
  h: Int,
  w: Int,
  c: Int,
  dimensions: Conv2DDimensions
) -> Int {
  (((n * dimensions.inputHeight + h) * dimensions.inputWidth + w) * dimensions.inputChannels) + c
}

func weightIndex(
  kh: Int,
  kw: Int,
  ic: Int,
  oc: Int,
  dimensions: Conv2DDimensions,
  layout: ConvolutionWeightLayout = .hwio
) -> Int {
  switch layout {
  case .hwio:
    return (((kh * dimensions.kernelWidth + kw) * dimensions.inputChannels + ic) * dimensions.outputChannels) + oc
  case .oihw:
    return (((oc * dimensions.inputChannels + ic) * dimensions.kernelHeight + kh) * dimensions.kernelWidth) + kw
  }
}

func outputIndex(
  n: Int,
  h: Int,
  w: Int,
  oc: Int,
  dimensions: Conv2DDimensions
) -> Int {
  (((n * dimensions.outputHeight + h) * dimensions.outputWidth + w) * dimensions.outputChannels) + oc
}

func createActivationData(dimensions: Conv2DDimensions) -> [Float16] {
  let count = dimensions.batchSize * dimensions.inputHeight * dimensions.inputWidth * dimensions.inputChannels
  if dimensions.inputChannels == 1 && dimensions.outputChannels == 1 && dimensions.kernelHeight == 3 && dimensions.kernelWidth == 3 {
    return (0..<count).map { Float16($0 + 1) }
  }
  return (0..<count).map {
    let value = Float((($0 * 7) + 3) % 19 - 9) * 0.0625
    return Float16(value)
  }
}

func createWeightData(
  dimensions: Conv2DDimensions,
  layout: ConvolutionWeightLayout = .hwio
) -> [Float16] {
  let count = dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * dimensions.outputChannels
  if dimensions.inputChannels == 1 && dimensions.outputChannels == 1 && dimensions.kernelHeight == 3 && dimensions.kernelWidth == 3 {
    var weights = [Float16](repeating: 0, count: count)
    weights[weightIndex(kh: 1, kw: 1, ic: 0, oc: 0, dimensions: dimensions, layout: layout)] = 1
    return weights
  }
  var weights = [Float16](repeating: 0, count: count)
  for kh in 0..<dimensions.kernelHeight {
    for kw in 0..<dimensions.kernelWidth {
      for ic in 0..<dimensions.inputChannels {
        for oc in 0..<dimensions.outputChannels {
          let canonicalIndex = weightIndex(kh: kh, kw: kw, ic: ic, oc: oc, dimensions: dimensions, layout: .hwio)
          let value = Float((((canonicalIndex * 5) + 1) % 17) - 8) * 0.03125
          weights[weightIndex(kh: kh, kw: kw, ic: ic, oc: oc, dimensions: dimensions, layout: layout)] = Float16(value)
        }
      }
    }
  }
  return weights
}

func activationIndex(
  n: Int,
  z: Int,
  h: Int,
  w: Int,
  c: Int,
  dimensions: Conv3DDimensions
) -> Int {
  ((((n * dimensions.inputDepth + z) * dimensions.inputHeight + h) * dimensions.inputWidth + w) * dimensions.inputChannels) + c
}

func weightIndex(
  kd: Int,
  kh: Int,
  kw: Int,
  ic: Int,
  oc: Int,
  dimensions: Conv3DDimensions,
  layout: Convolution3DWeightLayout = .dhwio
) -> Int {
  switch layout {
  case .dhwio:
    return ((((kd * dimensions.kernelHeight + kh) * dimensions.kernelWidth + kw) * dimensions.inputChannels + ic) * dimensions.outputChannels) + oc
  case .oidhw:
    return ((((oc * dimensions.inputChannels + ic) * dimensions.kernelDepth + kd) * dimensions.kernelHeight + kh) * dimensions.kernelWidth) + kw
  }
}

func outputIndex(
  n: Int,
  z: Int,
  h: Int,
  w: Int,
  oc: Int,
  dimensions: Conv3DDimensions
) -> Int {
  ((((n * dimensions.outputDepth + z) * dimensions.outputHeight + h) * dimensions.outputWidth + w) * dimensions.outputChannels) + oc
}

func createActivationData(dimensions: Conv3DDimensions) -> [Float16] {
  let count = dimensions.batchSize * dimensions.inputDepth * dimensions.inputHeight * dimensions.inputWidth * dimensions.inputChannels
  return (0..<count).map {
    let value = Float((($0 * 7) + 3) % 29 - 14) * 0.0625
    return Float16(value)
  }
}

func createWeightData(
  dimensions: Conv3DDimensions,
  layout: Convolution3DWeightLayout = .dhwio
) -> [Float16] {
  let count = dimensions.kernelDepth * dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * dimensions.outputChannels
  var weights = [Float16](repeating: 0, count: count)
  for kd in 0..<dimensions.kernelDepth {
    for kh in 0..<dimensions.kernelHeight {
      for kw in 0..<dimensions.kernelWidth {
        for ic in 0..<dimensions.inputChannels {
          for oc in 0..<dimensions.outputChannels {
            let canonicalIndex = weightIndex(
              kd: kd,
              kh: kh,
              kw: kw,
              ic: ic,
              oc: oc,
              dimensions: dimensions,
              layout: .dhwio
            )
            let value = Float((((canonicalIndex * 5) + 1) % 23) - 11) * 0.03125
            weights[weightIndex(kd: kd, kh: kh, kw: kw, ic: ic, oc: oc, dimensions: dimensions, layout: layout)] = Float16(value)
          }
        }
      }
    }
  }
  return weights
}

func createBiasData(dimensions: Conv3DDimensions) -> [Float16] {
  (0..<dimensions.outputChannels).map {
    let value = Float((($0 * 11) + 5) % 19 - 9) * 0.0625
    return Float16(value)
  }
}

func referenceConvolution(
  activation: [Float16],
  weights: [Float16],
  dimensions: Conv2DDimensions
) -> [Float] {
  var output = [Float](repeating: 0, count: dimensions.batchSize * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels)

  for n in 0..<dimensions.batchSize {
    for oh in 0..<dimensions.outputHeight {
      for ow in 0..<dimensions.outputWidth {
        for oc in 0..<dimensions.outputChannels {
          var sum: Float = 0
          for kh in 0..<dimensions.kernelHeight {
            let ih = oh * dimensions.strideY + kh * dimensions.dilationY - dimensions.paddingTop
            for kw in 0..<dimensions.kernelWidth {
              let iw = ow * dimensions.strideX + kw * dimensions.dilationX - dimensions.paddingLeft
              for ic in 0..<dimensions.inputChannels {
                if ih >= 0 && ih < dimensions.inputHeight && iw >= 0 && iw < dimensions.inputWidth {
                  let a = Float(activation[activationIndex(n: n, h: ih, w: iw, c: ic, dimensions: dimensions)])
                  let w = Float(weights[weightIndex(kh: kh, kw: kw, ic: ic, oc: oc, dimensions: dimensions)])
                  sum += a * w
                }
              }
            }
          }
          output[outputIndex(n: n, h: oh, w: ow, oc: oc, dimensions: dimensions)] = sum
        }
      }
    }
  }

  return output
}

func referenceConvolution(
  activation: [Float16],
  weights: [Float16],
  dimensions: Conv3DDimensions
) -> [Float] {
  var output = [Float](
    repeating: 0,
    count: dimensions.batchSize * dimensions.outputDepth * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
  )

  for n in 0..<dimensions.batchSize {
    for oz in 0..<dimensions.outputDepth {
      for oh in 0..<dimensions.outputHeight {
        for ow in 0..<dimensions.outputWidth {
          for oc in 0..<dimensions.outputChannels {
            var sum: Float = 0
            for kd in 0..<dimensions.kernelDepth {
              let iz = oz * dimensions.strideZ + kd * dimensions.dilationZ
              for kh in 0..<dimensions.kernelHeight {
                let ih = oh * dimensions.strideY + kh * dimensions.dilationY - dimensions.paddingTop
                for kw in 0..<dimensions.kernelWidth {
                  let iw = ow * dimensions.strideX + kw * dimensions.dilationX - dimensions.paddingLeft
                  for ic in 0..<dimensions.inputChannels {
                    if ih >= 0 && ih < dimensions.inputHeight && iw >= 0 && iw < dimensions.inputWidth {
                      let a = Float(activation[activationIndex(n: n, z: iz, h: ih, w: iw, c: ic, dimensions: dimensions)])
                      let w = Float(weights[weightIndex(kd: kd, kh: kh, kw: kw, ic: ic, oc: oc, dimensions: dimensions)])
                      sum += a * w
                    }
                  }
                }
              }
            }
            output[outputIndex(n: n, z: oz, h: oh, w: ow, oc: oc, dimensions: dimensions)] = sum
          }
        }
      }
    }
  }

  return output
}

func referenceConvolutionWithBias(
  activation: [Float16],
  weights: [Float16],
  bias: [Float16],
  dimensions: Conv3DDimensions
) -> [Float] {
  precondition(bias.count == dimensions.outputChannels)
  var output = referenceConvolution(
    activation: activation,
    weights: weights,
    dimensions: dimensions
  )

  for n in 0..<dimensions.batchSize {
    for oz in 0..<dimensions.outputDepth {
      for oh in 0..<dimensions.outputHeight {
        for ow in 0..<dimensions.outputWidth {
          for oc in 0..<dimensions.outputChannels {
            output[outputIndex(n: n, z: oz, h: oh, w: ow, oc: oc, dimensions: dimensions)] += Float(bias[oc])
          }
        }
      }
    }
  }

  return output
}

func makeShape(_ dimensions: [Int]) -> [NSNumber] {
  dimensions.map { NSNumber(value: $0) }
}

func copyToSharedBuffer(
  device: MTLDevice,
  values: [Float16]
) -> MTLBuffer {
  let size = values.count * MemoryLayout<Float16>.size
  guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
    fatalError("Could not allocate shared buffer")
  }
  buffer.contents().copyMemory(from: values, byteCount: size)
  return buffer
}

func makeZeroedSharedBuffer(
  device: MTLDevice,
  count: Int
) -> MTLBuffer {
  let size = count * MemoryLayout<Float16>.size
  guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
    fatalError("Could not allocate shared buffer")
  }
  buffer.contents().initializeMemory(as: Float16.self, repeating: 0, count: count)
  return buffer
}

func makeZeroedFloatBuffer(
  device: MTLDevice,
  count: Int
) -> MTLBuffer {
  let size = count * MemoryLayout<Float>.size
  guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
    fatalError("Could not allocate shared float buffer")
  }
  buffer.contents().initializeMemory(as: Float.self, repeating: 0, count: count)
  return buffer
}

func readFloat16Buffer(
  _ buffer: MTLBuffer,
  count: Int
) -> [Float16] {
  let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
  return Array(UnsafeBufferPointer(start: pointer, count: count))
}

func readFloat16BufferAsFloat(
  _ buffer: MTLBuffer,
  count: Int
) -> [Float] {
  readFloat16Buffer(buffer, count: count).map(Float.init)
}

func readFloatBuffer(
  _ buffer: MTLBuffer,
  count: Int
) -> [Float] {
  let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
  return Array(UnsafeBufferPointer(start: pointer, count: count))
}

func readOutputBuffer(
  _ buffer: MTLBuffer,
  count: Int,
  elementKind: TensorOpOutputElementKind
) -> [Float] {
  switch elementKind {
  case .half:
    return readFloat16BufferAsFloat(buffer, count: count)
  case .float:
    return readFloatBuffer(buffer, count: count)
  }
}

func validateOutput(
  actual: [Float],
  expected: [Float],
  dimensions: Conv2DDimensions,
  label: String,
  log: (String) -> Void
) -> ValidationResult {
  precondition(actual.count == expected.count)

  var mismatches = 0
  var maxAbsoluteError: Float = 0
  let tolerance: Float = 2e-2

  for n in 0..<dimensions.batchSize {
    for oh in 0..<dimensions.outputHeight {
      for ow in 0..<dimensions.outputWidth {
        for oc in 0..<dimensions.outputChannels {
          let index = outputIndex(n: n, h: oh, w: ow, oc: oc, dimensions: dimensions)
          let actualValue = actual[index]
          let expectedValue = expected[index]
          let error = abs(expectedValue - actualValue)
          maxAbsoluteError = max(maxAbsoluteError, error)
          if error > tolerance {
            mismatches += 1
            if mismatches <= 8 {
              log("\(label) mismatch at n=\(n), h=\(oh), w=\(ow), o=\(oc): expected \(expectedValue), got \(actualValue)")
            }
          }
        }
      }
    }
  }

  return ValidationResult(maxAbsoluteError: maxAbsoluteError, mismatches: mismatches)
}

func validateOutput(
  actual: [Float],
  expected: [Float],
  dimensions: Conv3DDimensions,
  label: String,
  log: (String) -> Void
) -> ValidationResult {
  precondition(actual.count == expected.count)

  var mismatches = 0
  var maxAbsoluteError: Float = 0
  let tolerance: Float = 2e-2

  for n in 0..<dimensions.batchSize {
    for oz in 0..<dimensions.outputDepth {
      for oh in 0..<dimensions.outputHeight {
        for ow in 0..<dimensions.outputWidth {
          for oc in 0..<dimensions.outputChannels {
            let index = outputIndex(n: n, z: oz, h: oh, w: ow, oc: oc, dimensions: dimensions)
            let actualValue = actual[index]
            let expectedValue = expected[index]
            let error = abs(expectedValue - actualValue)
            maxAbsoluteError = max(maxAbsoluteError, error)
            if error > tolerance {
              mismatches += 1
              if mismatches <= 8 {
                log("\(label) mismatch at n=\(n), z=\(oz), h=\(oh), w=\(ow), o=\(oc): expected \(expectedValue), got \(actualValue)")
              }
            }
          }
        }
      }
    }
  }

  return ValidationResult(maxAbsoluteError: maxAbsoluteError, mismatches: mismatches)
}

func compareOutputs(
  lhs: [Float],
  rhs: [Float],
  dimensions: Conv2DDimensions,
  lhsLabel: String,
  rhsLabel: String,
  log: (String) -> Void
) -> ValidationResult {
  precondition(lhs.count == rhs.count)

  var mismatches = 0
  var maxAbsoluteError: Float = 0
  let tolerance: Float = 2e-2

  for n in 0..<dimensions.batchSize {
    for oh in 0..<dimensions.outputHeight {
      for ow in 0..<dimensions.outputWidth {
        for oc in 0..<dimensions.outputChannels {
          let index = outputIndex(n: n, h: oh, w: ow, oc: oc, dimensions: dimensions)
          let lhsValue = lhs[index]
          let rhsValue = rhs[index]
          let error = abs(lhsValue - rhsValue)
          maxAbsoluteError = max(maxAbsoluteError, error)
          if error > tolerance {
            mismatches += 1
            if mismatches <= 8 {
              log("\(lhsLabel) vs \(rhsLabel) mismatch at n=\(n), h=\(oh), w=\(ow), o=\(oc): \(lhsValue) vs \(rhsValue)")
            }
          }
        }
      }
    }
  }

  return ValidationResult(maxAbsoluteError: maxAbsoluteError, mismatches: mismatches)
}

func compareOutputs(
  lhs: [Float],
  rhs: [Float],
  dimensions: Conv3DDimensions,
  lhsLabel: String,
  rhsLabel: String,
  log: (String) -> Void
) -> ValidationResult {
  precondition(lhs.count == rhs.count)

  var mismatches = 0
  var maxAbsoluteError: Float = 0
  let tolerance: Float = 2e-2

  for n in 0..<dimensions.batchSize {
    for oz in 0..<dimensions.outputDepth {
      for oh in 0..<dimensions.outputHeight {
        for ow in 0..<dimensions.outputWidth {
          for oc in 0..<dimensions.outputChannels {
            let index = outputIndex(n: n, z: oz, h: oh, w: ow, oc: oc, dimensions: dimensions)
            let lhsValue = lhs[index]
            let rhsValue = rhs[index]
            let error = abs(lhsValue - rhsValue)
            maxAbsoluteError = max(maxAbsoluteError, error)
            if error > tolerance {
              mismatches += 1
              if mismatches <= 8 {
                log("\(lhsLabel) vs \(rhsLabel) mismatch at n=\(n), z=\(oz), h=\(oh), w=\(ow), o=\(oc): \(lhsValue) vs \(rhsValue)")
              }
            }
          }
        }
      }
    }
  }

  return ValidationResult(maxAbsoluteError: maxAbsoluteError, mismatches: mismatches)
}

func operationsCount(
  dimensions: Conv2DDimensions,
  duplicatedCount: Int
) -> Int {
  2 * dimensions.batchSize * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels *
  dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * duplicatedCount
}

func throughputGFLOPS(
  dimensions: Conv2DDimensions,
  latency: Double,
  duplicatedCount: Int
) -> Double {
  guard latency > 0 else {
    return 0
  }
  return Double(operationsCount(dimensions: dimensions, duplicatedCount: duplicatedCount)) / latency / 1e9
}

func operationsCount(
  dimensions: Conv3DDimensions,
  duplicatedCount: Int
) -> Int {
  2 * dimensions.batchSize * dimensions.outputDepth * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels *
  dimensions.kernelDepth * dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * duplicatedCount
}

func throughputGFLOPS(
  dimensions: Conv3DDimensions,
  latency: Double,
  duplicatedCount: Int
) -> Double {
  guard latency > 0 else {
    return 0
  }
  return Double(operationsCount(dimensions: dimensions, duplicatedCount: duplicatedCount)) / latency / 1e9
}

func permutationBytesCount(
  dimensions: Conv3DDimensions,
  duplicatedCount: Int
) -> Int {
  let elementCount = dimensions.kernelDepth * dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * dimensions.outputChannels
  return 2 * elementCount * MemoryLayout<Float16>.size * duplicatedCount
}

func bandwidthGBPS(
  dimensions: Conv3DDimensions,
  latency: Double,
  duplicatedCount: Int
) -> Double {
  guard latency > 0 else {
    return 0
  }
  return Double(permutationBytesCount(dimensions: dimensions, duplicatedCount: duplicatedCount)) / latency / 1e9
}

func validateFlatOutput(
  actual: [Float],
  expected: [Float16],
  label: String,
  log: (String) -> Void
) -> ValidationResult {
  precondition(actual.count == expected.count)

  var mismatches = 0
  var maxAbsoluteError: Float = 0
  let tolerance: Float = 0

  for index in 0..<actual.count {
    let expectedValue = Float(expected[index])
    let error = abs(actual[index] - expectedValue)
    maxAbsoluteError = max(maxAbsoluteError, error)
    if error > tolerance {
      mismatches += 1
      if mismatches <= 8 {
        log("\(label) mismatch at linear index \(index): expected \(expectedValue), got \(actual[index])")
      }
    }
  }

  return ValidationResult(maxAbsoluteError: maxAbsoluteError, mismatches: mismatches)
}

func printSingleRunThroughput(
  label: String,
  dimensions: Conv2DDimensions,
  result: ExecutionResult,
  duplicatedCount: Int,
  log: (String) -> Void
) {
  if let wallLatency = result.wallLatency, wallLatency > 0 {
    log(String(
      format: "%@ wall latency: %.6f ms, throughput: %.3f GFLOP/s",
      label,
      wallLatency * 1e3,
      throughputGFLOPS(dimensions: dimensions, latency: wallLatency, duplicatedCount: duplicatedCount)
    ))
  }
  if let gpuLatency = result.gpuLatency, gpuLatency > 0 {
    log(String(
      format: "%@ GPU latency: %.6f ms, throughput: %.3f GFLOP/s",
      label,
      gpuLatency * 1e3,
      throughputGFLOPS(dimensions: dimensions, latency: gpuLatency, duplicatedCount: duplicatedCount)
    ))
  }
}
