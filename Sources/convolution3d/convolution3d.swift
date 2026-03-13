import Foundation
import Metal

/*
 Standalone Conv3D tensor-op research scaffold.

 This file is intentionally self-contained, similar in spirit to `matmul.swift`.
 It captures the final working Conv3D approach we validated on the M5 iPad:

 1. There is no native 3D tensor-op convolution API here. The working approach is
    to reuse the 2D `convolution2d` tensor op over spatial slices.
 2. Conv3D is expressed as host-side orchestration over the depth axis:
    - for each output depth slice
    - for each kernel depth slice
    - launch one spatial Conv2D kernel
 3. The first depth slice uses `mode::multiply`.
 4. Later depth slices use `mode::multiply_accumulate`.
 5. On Apple GPU hardware we tested, correctness requires the cooperative
    destination path. Direct destination tensors were not reliable enough on the
    device path we care about.
 6. For `multiply_accumulate`, the cooperative tensor must load the existing
    output tile before running the op, then store it back after accumulation.
 7. Edge handling is kept inside the shader, not split into separate dispatches:
    - interior tiles use static `slice<...>`
    - edge tiles use dynamic `slice(...)`
    - the convolution descriptor itself stays static for the full tile
 8. Bias is implemented as a first-stage variant:
    - the first depth slice uses `multiply_bias`
    - later depth slices use plain `multiply_accumulate`
    This is simpler than bias-on-last-slice because the current host orchestration
    already has a unique "first producer" for every output tile.
    Bias is loaded from a rank-1 tensor of length `outputChannels` into a
    cooperative destination tensor, so the broadcast happens inside the tensor-op
    value layout rather than as a separate pointwise pass.
 9. We also keep a small `OIDHW -> DHWIO` permutation kernel here because the
    tensor-op path consumes `DHWIO`, while some surrounding software may store
    weights as `OIDHW`. The benchmark reports this as explicit overhead.

 Tensor layouts used in this file:
 - activation: NDHWC in host memory
 - tensor-op spatial slice activation: NHWC
 - weights for tensor-op kernel: DHWIO in host memory, sliced per depth into HWIO
 - output: NDHWO in host memory, sliced per depth into NHWO

 Notes for productionization:
 - The shader source builders in this file are the core research artifact.
 - The session types are deliberately small and direct; they can be split into
   reusable library code later.
 - Validation is CPU reference based only on small shapes. Large shapes are
   profile-only so local runs stay practical.
*/

struct Conv3DDimensions: CustomStringConvertible {
  var batchSize: Int
  var inputDepth: Int
  var inputHeight: Int
  var inputWidth: Int
  var inputChannels: Int
  var outputChannels: Int
  var kernelDepth: Int
  var kernelHeight: Int
  var kernelWidth: Int
  var strideZ: Int
  var strideY: Int
  var strideX: Int
  var dilationZ: Int
  var dilationY: Int
  var dilationX: Int

  var outputDepth: Int {
    ((inputDepth - ((kernelDepth - 1) * dilationZ + 1)) / strideZ) + 1
  }

  var outputHeight: Int {
    ((inputHeight - ((kernelHeight - 1) * dilationY + 1)) / strideY) + 1
  }

  var outputWidth: Int {
    ((inputWidth - ((kernelWidth - 1) * dilationX + 1)) / strideX) + 1
  }

  var description: String {
    "N=\(batchSize), D=\(inputDepth), H=\(inputHeight), W=\(inputWidth), C=\(inputChannels), O=\(outputChannels), KD=\(kernelDepth), KH=\(kernelHeight), KW=\(kernelWidth), SZ=\(strideZ), SY=\(strideY), SX=\(strideX), DZ=\(dilationZ), DY=\(dilationY), DX=\(dilationX)"
  }
}

struct Conv2DSpatialSliceDimensions {
  var inputHeight: Int
  var inputWidth: Int
  var inputChannels: Int
  var outputChannels: Int
  var kernelHeight: Int
  var kernelWidth: Int
  var strideY: Int
  var strideX: Int
  var dilationY: Int
  var dilationX: Int

  var outputHeight: Int {
    ((inputHeight - ((kernelHeight - 1) * dilationY + 1)) / strideY) + 1
  }

  var outputWidth: Int {
    ((inputWidth - ((kernelWidth - 1) * dilationX + 1)) / strideX) + 1
  }
}

struct BuildOptions {
  var executionSIMDGroups: Int
  var outputTileWidth: Int
  var outputTileHeight: Int
}

struct ProfileOptions {
  var warmupIterations: Int
  var timedIterations: Int
  var duplicatedCount: Int
}

struct ValidationResult {
  var maxAbsoluteError: Float
  var mismatches: Int
}

struct ExecutionResult {
  var output: [Float]
  var wallLatency: Double?
  var gpuLatency: Double?
}

struct Conv3DProfileResult {
  var averageWallLatencyMS: Double
  var averageGPULatencyMS: Double?
  var wallThroughputGFLOPS: Double
  var gpuThroughputGFLOPS: Double?
}

struct Conv3DBiasProfileResult {
  var averageWallLatencyMS: Double
  var averageGPULatencyMS: Double?
  var wallThroughputGFLOPS: Double
  var gpuThroughputGFLOPS: Double?
}

struct PermuteAndConv3DProfileResult {
  var permutationValidation: ValidationResult
  var combinedValidation: ValidationResult
  var permutationAverageWallLatencyMS: Double
  var permutationAverageGPULatencyMS: Double?
  var permutationWallBandwidthGBPS: Double
  var permutationGPUBandwidthGBPS: Double?
  var combinedAverageWallLatencyMS: Double
  var combinedAverageGPULatencyMS: Double?
  var combinedWallThroughputGFLOPS: Double
  var combinedGPUThroughputGFLOPS: Double?
}

enum Conv3DWeightLayout {
  case dhwio
  case oidhw
}

enum Conv3DMode {
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

func spatialSliceDimensions(_ dimensions: Conv3DDimensions) -> Conv2DSpatialSliceDimensions {
  Conv2DSpatialSliceDimensions(
    inputHeight: dimensions.inputHeight,
    inputWidth: dimensions.inputWidth,
    inputChannels: dimensions.inputChannels,
    outputChannels: dimensions.outputChannels,
    kernelHeight: dimensions.kernelHeight,
    kernelWidth: dimensions.kernelWidth,
    strideY: dimensions.strideY,
    strideX: dimensions.strideX,
    dilationY: dimensions.dilationY,
    dilationX: dimensions.dilationX
  )
}

func makeDevice() -> MTLDevice {
  guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device.")
  }
  guard device.supportsFamily(.metal4) else {
    fatalError("This device does not support the tensor operations used in this file.")
  }
  return device
}

func makeDispatchThreadgroups(
  outputWidth: Int,
  outputHeight: Int,
  tileWidth: Int,
  tileHeight: Int
) -> MTLSize {
  MTLSize(
    width: (outputWidth + tileWidth - 1) / tileWidth,
    height: (outputHeight + tileHeight - 1) / tileHeight,
    depth: 1
  )
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
  layout: Conv3DWeightLayout = .dhwio
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
  layout: Conv3DWeightLayout = .dhwio
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
  weightsDHWIO: [Float16],
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
                let ih = oh * dimensions.strideY + kh * dimensions.dilationY
                for kw in 0..<dimensions.kernelWidth {
                  let iw = ow * dimensions.strideX + kw * dimensions.dilationX
                  for ic in 0..<dimensions.inputChannels {
                    let a = Float(activation[activationIndex(n: n, z: iz, h: ih, w: iw, c: ic, dimensions: dimensions)])
                    let w = Float(weightsDHWIO[weightIndex(kd: kd, kh: kh, kw: kw, ic: ic, oc: oc, dimensions: dimensions, layout: .dhwio)])
                    sum += a * w
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
  weightsDHWIO: [Float16],
  bias: [Float16],
  dimensions: Conv3DDimensions
) -> [Float] {
  precondition(bias.count == dimensions.outputChannels)
  var output = referenceConvolution(
    activation: activation,
    weightsDHWIO: weightsDHWIO,
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

func copyToSharedBuffer(
  device: MTLDevice,
  values: [Float16]
) -> MTLBuffer {
  let size = values.count * MemoryLayout<Float16>.size
  guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
    fatalError("Could not allocate shared buffer.")
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
    fatalError("Could not allocate shared buffer.")
  }
  buffer.contents().initializeMemory(as: Float16.self, repeating: 0, count: count)
  return buffer
}

func readFloat16BufferAsFloat(
  _ buffer: MTLBuffer,
  count: Int
) -> [Float] {
  let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
  return Array(UnsafeBufferPointer(start: pointer, count: count)).map(Float.init)
}

func validateOutput(
  actual: [Float],
  expected: [Float],
  dimensions: Conv3DDimensions,
  label: String
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
            let error = abs(actual[index] - expected[index])
            maxAbsoluteError = max(maxAbsoluteError, error)
            if error > tolerance {
              mismatches += 1
              if mismatches <= 8 {
                print("\(label) mismatch at n=\(n), z=\(oz), h=\(oh), w=\(ow), o=\(oc): expected \(expected[index]), got \(actual[index])")
              }
            }
          }
        }
      }
    }
  }

  return ValidationResult(maxAbsoluteError: maxAbsoluteError, mismatches: mismatches)
}

func validateFlatOutput(
  actual: [Float],
  expected: [Float16],
  label: String
) -> ValidationResult {
  precondition(actual.count == expected.count)

  var mismatches = 0
  var maxAbsoluteError: Float = 0
  for index in 0..<actual.count {
    let expectedValue = Float(expected[index])
    let error = abs(actual[index] - expectedValue)
    maxAbsoluteError = max(maxAbsoluteError, error)
    if error > 0 {
      mismatches += 1
      if mismatches <= 8 {
        print("\(label) mismatch at linear index \(index): expected \(expectedValue), got \(actual[index])")
      }
    }
  }

  return ValidationResult(maxAbsoluteError: maxAbsoluteError, mismatches: mismatches)
}

func createConv3DTensorOpSource(
  dimensions: Conv3DDimensions,
  buildOptions: BuildOptions,
  includeBias: Bool = false
) -> String {
  let slice = spatialSliceDimensions(dimensions)
  let inputTileWidth = (buildOptions.outputTileWidth - 1) * slice.strideX + (slice.kernelWidth - 1) * slice.dilationX + 1
  let inputTileHeight = (buildOptions.outputTileHeight - 1) * slice.strideY + (slice.kernelHeight - 1) * slice.dilationY + 1
  let scopeType = "execution_simdgroups<\(buildOptions.executionSIMDGroups)>"
  let offsetSetup =
    "  conv2d_op.set_offsets(int2(\((slice.kernelWidth - 1) * slice.dilationX / 2), \((slice.kernelHeight - 1) * slice.dilationY / 2)));"

  func destinationSetup(outputTensorSetup: String, mode: Conv3DMode, withBias: Bool) -> String {
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

  func createKernel(mode: Conv3DMode, withBias: Bool) -> String {
    let staticOutputSetup = """
  auto output = output_base.slice<\(slice.outputChannels), \(buildOptions.outputTileWidth), \(buildOptions.outputTileHeight), 1>(
      0,
      output_origin_x,
      output_origin_y,
      0);
"""
    let dynamicOutputSetup = """
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
      dextents<int32_t, 1>(\(slice.outputChannels)));
"""
    } else {
      biasTensorSetup = ""
    }

    return """
kernel void \(mode.functionName(withBias: withBias))(device half *activation_buf [[buffer(0)]],
                                 device half *weights_buf [[buffer(1)]],
                                 device half *output_buf [[buffer(2)]],
                                 constant uint& activation_base [[buffer(3)]],
                                 constant uint& weights_base [[buffer(4)]],
                                 constant uint& output_base_offset [[buffer(5)]]\(biasArguments),
                                 uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]])
{
  activation_buf += activation_base;
  weights_buf += weights_base;
  output_buf += output_base_offset;

  const int output_origin_x = int(threadgroup_position_in_grid.x) * \(buildOptions.outputTileWidth);
  const int output_origin_y = int(threadgroup_position_in_grid.y) * \(buildOptions.outputTileHeight);
  if (output_origin_x >= \(slice.outputWidth) || output_origin_y >= \(slice.outputHeight)) {
    return;
  }

  const int input_origin_x = output_origin_x * \(slice.strideX);
  const int input_origin_y = output_origin_y * \(slice.strideY);

  auto activation_base_tensor = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(\(slice.inputChannels), \(slice.inputWidth), \(slice.inputHeight), 1));
  auto output_base = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      output_buf,
      dextents<int32_t, 4>(\(slice.outputChannels), \(slice.outputWidth), \(slice.outputHeight), 1));
\(biasTensorSetup)
  auto weights = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      weights_buf,
      dextents<int32_t, 4>(\(slice.outputChannels), \(slice.inputChannels), \(slice.kernelWidth), \(slice.kernelHeight)));
  constexpr auto descriptor = convolution2d_descriptor(
      int4(\(slice.outputChannels), \(buildOptions.outputTileWidth), \(buildOptions.outputTileHeight), 1),
      int4(\(slice.inputChannels), \(inputTileWidth), \(inputTileHeight), 1),
      int2(\(slice.kernelWidth), \(slice.kernelHeight)),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(\(slice.strideX), \(slice.strideY)),
      int2(\(slice.dilationX), \(slice.dilationY)),
      1,
      false,
      convolution2d_descriptor::mode::\(mode.descriptorMode));
  convolution2d<descriptor, \(scopeType)> conv2d_op;
\(offsetSetup)

  if (output_origin_x + \(buildOptions.outputTileWidth) <= \(slice.outputWidth) &&
      output_origin_y + \(buildOptions.outputTileHeight) <= \(slice.outputHeight)) {
    auto activation = activation_base_tensor.slice<\(slice.inputChannels), \(inputTileWidth), \(inputTileHeight), 1>(
        0,
        input_origin_x,
        input_origin_y,
        0);
\(destinationSetup(outputTensorSetup: staticOutputSetup, mode: mode, withBias: withBias))
  } else {
    auto activation = activation_base_tensor.slice(
        0,
        input_origin_x,
        input_origin_y,
        0);
\(destinationSetup(outputTensorSetup: dynamicOutputSetup, mode: mode, withBias: withBias))
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

  // source is OIDHW, destination is DHWIO
  const uint sourceIndex = ((((oc * inputChannels + ic) * kernelDepth + kd) * kernelHeight + kh) * kernelWidth) + kw;
  destination[gid] = source[sourceIndex];
}
"""
}

final class TensorOpSession3D {
  private let dimensions: Conv3DDimensions
  private let slice: Conv2DSpatialSliceDimensions
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
    weightsDHWIO: [Float16]
  ) throws {
    self.dimensions = dimensions
    self.slice = spatialSliceDimensions(dimensions)
    self.buildOptions = buildOptions

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(domain: "convolution3d", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not create command queue."])
    }
    self.commandQueue = commandQueue

    let library = try device.makeLibrary(
      source: createConv3DTensorOpSource(dimensions: dimensions, buildOptions: buildOptions),
      options: nil
    )
    guard let multiplyFunction = library.makeFunction(name: Conv3DMode.multiply.functionName()),
          let accumulateFunction = library.makeFunction(name: Conv3DMode.multiplyAccumulate.functionName()) else {
      throw NSError(domain: "convolution3d", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not create Conv3D tensor-op functions."])
    }
    multiplyPipelineState = try device.makeComputePipelineState(function: multiplyFunction)
    accumulatePipelineState = try device.makeComputePipelineState(function: accumulateFunction)

    outputCount = dimensions.batchSize * dimensions.outputDepth * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    activationBuffer = copyToSharedBuffer(device: device, values: activation)
    weightBuffer = copyToSharedBuffer(device: device, values: weightsDHWIO)
    outputBuffer = makeZeroedSharedBuffer(device: device, count: outputCount)
  }

  private func execute(duplicatedCount: Int, readOutput: Bool) -> ExecutionResult {
    precondition(duplicatedCount > 0)

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create Conv3D tensor-op command buffer or encoder.")
    }

    computeCommandEncoder.useResource(activationBuffer, usage: .read)
    computeCommandEncoder.useResource(weightBuffer, usage: .read)
    computeCommandEncoder.useResource(outputBuffer, usage: [.read, .write])

    let threadgroups = makeDispatchThreadgroups(
      outputWidth: slice.outputWidth,
      outputHeight: slice.outputHeight,
      tileWidth: buildOptions.outputTileWidth,
      tileHeight: buildOptions.outputTileHeight
    )
    let threadsPerThreadgroup = MTLSize(
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
      fatalError("Conv3D tensor-op command buffer failed. status=\(commandBuffer.status.rawValue), error=\(errorDescription)")
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

final class TensorOpBiasSession3D {
  private let dimensions: Conv3DDimensions
  private let slice: Conv2DSpatialSliceDimensions
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
    weightsDHWIO: [Float16],
    bias: [Float16]
  ) throws {
    precondition(bias.count == dimensions.outputChannels)

    self.dimensions = dimensions
    self.slice = spatialSliceDimensions(dimensions)
    self.buildOptions = buildOptions

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(domain: "convolution3d", code: 30, userInfo: [NSLocalizedDescriptionKey: "Could not create bias command queue."])
    }
    self.commandQueue = commandQueue

    let library = try device.makeLibrary(
      source: createConv3DTensorOpSource(dimensions: dimensions, buildOptions: buildOptions, includeBias: true),
      options: nil
    )
    guard let accumulateFunction = library.makeFunction(name: Conv3DMode.multiplyAccumulate.functionName()),
          let multiplyBiasFunction = library.makeFunction(name: Conv3DMode.multiply.functionName(withBias: true)) else {
      throw NSError(domain: "convolution3d", code: 31, userInfo: [NSLocalizedDescriptionKey: "Could not create Conv3D bias functions."])
    }
    accumulatePipelineState = try device.makeComputePipelineState(function: accumulateFunction)
    multiplyBiasPipelineState = try device.makeComputePipelineState(function: multiplyBiasFunction)

    outputCount = dimensions.batchSize * dimensions.outputDepth * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    activationBuffer = copyToSharedBuffer(device: device, values: activation)
    weightBuffer = copyToSharedBuffer(device: device, values: weightsDHWIO)
    outputBuffer = makeZeroedSharedBuffer(device: device, count: outputCount)
    biasBuffer = copyToSharedBuffer(device: device, values: bias)
  }

  private func execute(duplicatedCount: Int, readOutput: Bool) -> ExecutionResult {
    precondition(duplicatedCount > 0)

    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create Conv3D bias command buffer or encoder.")
    }

    computeCommandEncoder.useResource(activationBuffer, usage: .read)
    computeCommandEncoder.useResource(weightBuffer, usage: .read)
    computeCommandEncoder.useResource(outputBuffer, usage: [.read, .write])
    computeCommandEncoder.useResource(biasBuffer, usage: .read)

    let threadgroups = makeDispatchThreadgroups(
      outputWidth: slice.outputWidth,
      outputHeight: slice.outputHeight,
      tileWidth: buildOptions.outputTileWidth,
      tileHeight: buildOptions.outputTileHeight
    )
    let threadsPerThreadgroup = MTLSize(
      width: max(multiplyBiasPipelineState.threadExecutionWidth, accumulatePipelineState.threadExecutionWidth) * buildOptions.executionSIMDGroups,
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
            let pipelineState: MTLComputePipelineState
            if kd == 0 {
              pipelineState = multiplyBiasPipelineState
            } else {
              pipelineState = accumulatePipelineState
            }

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
      fatalError("Conv3D bias command buffer failed. status=\(commandBuffer.status.rawValue), error=\(errorDescription)")
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
      throw NSError(domain: "convolution3d", code: 3, userInfo: [NSLocalizedDescriptionKey: "Could not create permutation command queue."])
    }
    self.commandQueue = commandQueue

    let library = try device.makeLibrary(source: createConv3DWeightPermutationSource(), options: nil)
    guard let function = library.makeFunction(name: "permute_oidhw_to_dhwio") else {
      throw NSError(domain: "convolution3d", code: 4, userInfo: [NSLocalizedDescriptionKey: "Could not create permutation function."])
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
      fatalError("Could not create permutation command buffer or encoder.")
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

    let threadsPerThreadgroup = MTLSize(
      width: min(max(pipelineState.threadExecutionWidth, 1), pipelineState.maxTotalThreadsPerThreadgroup),
      height: 1,
      depth: 1
    )
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
      fatalError("Permutation command buffer failed. status=\(commandBuffer.status.rawValue), error=\(errorDescription)")
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
  private let slice: Conv2DSpatialSliceDimensions
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
    self.dimensions = dimensions
    self.slice = spatialSliceDimensions(dimensions)
    self.buildOptions = buildOptions

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(domain: "convolution3d", code: 5, userInfo: [NSLocalizedDescriptionKey: "Could not create combined command queue."])
    }
    self.commandQueue = commandQueue

    let permutationLibrary = try device.makeLibrary(source: createConv3DWeightPermutationSource(), options: nil)
    guard let permutationFunction = permutationLibrary.makeFunction(name: "permute_oidhw_to_dhwio") else {
      throw NSError(domain: "convolution3d", code: 6, userInfo: [NSLocalizedDescriptionKey: "Could not create combined permutation function."])
    }
    permutationPipelineState = try device.makeComputePipelineState(function: permutationFunction)

    let convolutionLibrary = try device.makeLibrary(
      source: createConv3DTensorOpSource(dimensions: dimensions, buildOptions: buildOptions),
      options: nil
    )
    guard let multiplyFunction = convolutionLibrary.makeFunction(name: Conv3DMode.multiply.functionName()),
          let accumulateFunction = convolutionLibrary.makeFunction(name: Conv3DMode.multiplyAccumulate.functionName()) else {
      throw NSError(domain: "convolution3d", code: 7, userInfo: [NSLocalizedDescriptionKey: "Could not create combined Conv3D functions."])
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
      fatalError("Could not create combined command buffer or encoder.")
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

    let convolutionThreadgroups = makeDispatchThreadgroups(
      outputWidth: slice.outputWidth,
      outputHeight: slice.outputHeight,
      tileWidth: buildOptions.outputTileWidth,
      tileHeight: buildOptions.outputTileHeight
    )
    let convolutionThreadsPerThreadgroup = MTLSize(
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
            computeCommandEncoder.dispatchThreadgroups(convolutionThreadgroups, threadsPerThreadgroup: convolutionThreadsPerThreadgroup)
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
      fatalError("Combined command buffer failed. status=\(commandBuffer.status.rawValue), error=\(errorDescription)")
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

func validationCases() -> [Conv3DDimensions] {
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

func defaultBuildOptions() -> BuildOptions {
  BuildOptions(executionSIMDGroups: 4, outputTileWidth: 8, outputTileHeight: 8)
}

func validationPermutationDimensions() -> Conv3DDimensions {
  validationCases()[1]
}

func largeProfileDimensions() -> Conv3DDimensions {
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

func largeProfileOptions() -> ProfileOptions {
  ProfileOptions(warmupIterations: 1, timedIterations: 3, duplicatedCount: 1)
}

func validateTensorOpConv3D(
  device: MTLDevice,
  buildOptions: BuildOptions
) {
  print("Conv3D tensor-op validation")
  print("Build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(buildOptions.outputTileWidth)x\(buildOptions.outputTileHeight)")

  var allPassed = true
  for (index, dimensions) in validationCases().enumerated() {
    let activation = createActivationData(dimensions: dimensions)
    let weightsDHWIO = createWeightData(dimensions: dimensions, layout: .dhwio)
    let expected = referenceConvolution(activation: activation, weightsDHWIO: weightsDHWIO, dimensions: dimensions)

    let session: TensorOpSession3D
    do {
      session = try TensorOpSession3D(
        device: device,
        dimensions: dimensions,
        buildOptions: buildOptions,
        activation: activation,
        weightsDHWIO: weightsDHWIO
      )
    } catch {
      fatalError("Could not create Conv3D tensor-op session: \(error)")
    }

    let result = session.run(duplicatedCount: 1)
    let validation = validateOutput(
      actual: result.output,
      expected: expected,
      dimensions: dimensions,
      label: "Tensor-op Conv3D"
    )

    print("Case \(index + 1)/\(validationCases().count): \(dimensions)")
    print("Max absolute error: \(validation.maxAbsoluteError)")
    print("Mismatches above tolerance: \(validation.mismatches)")
    if validation.mismatches > 0 {
      allPassed = false
    }
  }
  print("All Conv3D tensor-op validation cases passed: \(allPassed)")
}

func validateTensorOpConv3DBias(
  device: MTLDevice,
  buildOptions: BuildOptions
) {
  print("Conv3D tensor-op bias validation")
  print("Build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(buildOptions.outputTileWidth)x\(buildOptions.outputTileHeight)")

  var allPassed = true
  for (index, dimensions) in validationCases().enumerated() {
    let activation = createActivationData(dimensions: dimensions)
    let weightsDHWIO = createWeightData(dimensions: dimensions, layout: .dhwio)
    let bias = createBiasData(dimensions: dimensions)
    let expected = referenceConvolutionWithBias(
      activation: activation,
      weightsDHWIO: weightsDHWIO,
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
        weightsDHWIO: weightsDHWIO,
        bias: bias
      )
    } catch {
      fatalError("Could not create Conv3D bias session: \(error)")
    }

    let result = session.run(duplicatedCount: 1)
    let validation = validateOutput(
      actual: result.output,
      expected: expected,
      dimensions: dimensions,
      label: "Tensor-op Conv3D Bias"
    )

    print("Case \(index + 1)/\(validationCases().count): \(dimensions)")
    print("Max absolute error: \(validation.maxAbsoluteError)")
    print("Mismatches above tolerance: \(validation.mismatches)")
    if validation.mismatches > 0 {
      allPassed = false
    }
  }
  print("All Conv3D tensor-op bias validation cases passed: \(allPassed)")
}

func profileTensorOpConv3D(
  device: MTLDevice,
  dimensions: Conv3DDimensions,
  buildOptions: BuildOptions,
  options: ProfileOptions
) -> Conv3DProfileResult {
  let activation = createActivationData(dimensions: dimensions)
  let weightsDHWIO = createWeightData(dimensions: dimensions, layout: .dhwio)

  let session: TensorOpSession3D
  do {
    session = try TensorOpSession3D(
      device: device,
      dimensions: dimensions,
      buildOptions: buildOptions,
      activation: activation,
      weightsDHWIO: weightsDHWIO
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
  let wallThroughput = throughputGFLOPS(dimensions: dimensions, latency: averageWallLatency, duplicatedCount: options.duplicatedCount)
  let gpuThroughput = throughputGFLOPS(dimensions: dimensions, latency: averageGPULatency, duplicatedCount: options.duplicatedCount)

  let slice = spatialSliceDimensions(dimensions)
  let dispatchGrid = makeDispatchThreadgroups(
    outputWidth: slice.outputWidth,
    outputHeight: slice.outputHeight,
    tileWidth: buildOptions.outputTileWidth,
    tileHeight: buildOptions.outputTileHeight
  )

  print("Conv3D tensor-op build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(buildOptions.outputTileWidth)x\(buildOptions.outputTileHeight)")
  print("Conv3D dispatch threadgroups: \(dispatchGrid.width)x\(dispatchGrid.height)x\(dispatchGrid.depth), threadsPerThreadgroup: \(buildOptions.executionSIMDGroups * 32)")
  print("Conv3D profile options: warmup=\(options.warmupIterations), timed=\(options.timedIterations), duplicatedCount=\(options.duplicatedCount)")
  print(String(format: "Conv3D tensor op average wall latency: %.6f ms, throughput: %.3f GFLOP/s", averageWallLatency * 1e3, wallThroughput))
  print(String(format: "Conv3D tensor op average GPU latency: %.6f ms, throughput: %.3f GFLOP/s", averageGPULatency * 1e3, gpuThroughput))

  return Conv3DProfileResult(
    averageWallLatencyMS: averageWallLatency * 1e3,
    averageGPULatencyMS: averageGPULatency * 1e3,
    wallThroughputGFLOPS: wallThroughput,
    gpuThroughputGFLOPS: gpuThroughput
  )
}

func profileTensorOpConv3DBias(
  device: MTLDevice,
  dimensions: Conv3DDimensions,
  buildOptions: BuildOptions,
  options: ProfileOptions
) -> Conv3DBiasProfileResult {
  let activation = createActivationData(dimensions: dimensions)
  let weightsDHWIO = createWeightData(dimensions: dimensions, layout: .dhwio)
  let bias = createBiasData(dimensions: dimensions)

  let session: TensorOpBiasSession3D
  do {
    session = try TensorOpBiasSession3D(
      device: device,
      dimensions: dimensions,
      buildOptions: buildOptions,
      activation: activation,
      weightsDHWIO: weightsDHWIO,
      bias: bias
    )
  } catch {
    fatalError("Could not create Conv3D bias session: \(error)")
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
  // Throughput is still reported against convolution MAC count only.
  let wallThroughput = throughputGFLOPS(dimensions: dimensions, latency: averageWallLatency, duplicatedCount: options.duplicatedCount)
  let gpuThroughput = throughputGFLOPS(dimensions: dimensions, latency: averageGPULatency, duplicatedCount: options.duplicatedCount)

  print("Conv3D tensor-op bias build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(buildOptions.outputTileWidth)x\(buildOptions.outputTileHeight)")
  print("Conv3D tensor-op bias profile options: warmup=\(options.warmupIterations), timed=\(options.timedIterations), duplicatedCount=\(options.duplicatedCount)")
  print(String(format: "Conv3D tensor op bias average wall latency: %.6f ms, throughput: %.3f GFLOP/s", averageWallLatency * 1e3, wallThroughput))
  print(String(format: "Conv3D tensor op bias average GPU latency: %.6f ms, throughput: %.3f GFLOP/s", averageGPULatency * 1e3, gpuThroughput))

  return Conv3DBiasProfileResult(
    averageWallLatencyMS: averageWallLatency * 1e3,
    averageGPULatencyMS: averageGPULatency * 1e3,
    wallThroughputGFLOPS: wallThroughput,
    gpuThroughputGFLOPS: gpuThroughput
  )
}

func profilePermutationAndConv3D(
  device: MTLDevice,
  validationDimensions: Conv3DDimensions,
  profileDimensions: Conv3DDimensions,
  buildOptions: BuildOptions,
  options: ProfileOptions
) -> PermuteAndConv3DProfileResult {
  let validationWeightsOIDHW = createWeightData(dimensions: validationDimensions, layout: .oidhw)
  let validationWeightsDHWIO = createWeightData(dimensions: validationDimensions, layout: .dhwio)
  let validationActivation = createActivationData(dimensions: validationDimensions)
  let validationExpected = referenceConvolution(
    activation: validationActivation,
    weightsDHWIO: validationWeightsDHWIO,
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
    fatalError("Could not create validation permutation sessions: \(error)")
  }

  let permutationValidation = validateFlatOutput(
    actual: permutationValidationSession.run(duplicatedCount: 1).output,
    expected: validationWeightsDHWIO,
    label: "OIDHW->DHWIO permutation"
  )
  let combinedValidation = validateOutput(
    actual: combinedValidationSession.run(duplicatedCount: 1).output,
    expected: validationExpected,
    dimensions: validationDimensions,
    label: "Permute+Tensor-op Conv3D"
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
    fatalError("Could not create profile permutation sessions: \(error)")
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

  let permutationWallBandwidth = bandwidthGBPS(dimensions: profileDimensions, latency: averagePermutationWallLatency, duplicatedCount: options.duplicatedCount)
  let permutationGPUBandwidth = bandwidthGBPS(dimensions: profileDimensions, latency: averagePermutationGPULatency, duplicatedCount: options.duplicatedCount)
  let combinedWallThroughput = throughputGFLOPS(dimensions: profileDimensions, latency: averageCombinedWallLatency, duplicatedCount: options.duplicatedCount)
  let combinedGPUThroughput = throughputGFLOPS(dimensions: profileDimensions, latency: averageCombinedGPULatency, duplicatedCount: options.duplicatedCount)

  print("Validated Conv3D weight permutation on \(validationDimensions)")
  print("Permutation max absolute error: \(permutationValidation.maxAbsoluteError)")
  print("Permutation mismatches above tolerance: \(permutationValidation.mismatches)")
  print("Permute+Conv3D max absolute error: \(combinedValidation.maxAbsoluteError)")
  print("Permute+Conv3D mismatches above tolerance: \(combinedValidation.mismatches)")
  print("Permute+Conv3D build options: executionSIMDGroups=\(buildOptions.executionSIMDGroups), tile=\(buildOptions.outputTileWidth)x\(buildOptions.outputTileHeight)")
  print("Permute+Conv3D profile dimensions: \(profileDimensions)")
  print("Permute+Conv3D profile options: warmup=\(options.warmupIterations), timed=\(options.timedIterations), duplicatedCount=\(options.duplicatedCount)")
  print(String(format: "Permutation average wall latency: %.6f ms, bandwidth: %.3f GB/s", averagePermutationWallLatency * 1e3, permutationWallBandwidth))
  print(String(format: "Permutation average GPU latency: %.6f ms, bandwidth: %.3f GB/s", averagePermutationGPULatency * 1e3, permutationGPUBandwidth))
  print(String(format: "Permute+Conv3D average wall latency: %.6f ms, throughput: %.3f GFLOP/s", averageCombinedWallLatency * 1e3, combinedWallThroughput))
  print(String(format: "Permute+Conv3D average GPU latency: %.6f ms, throughput: %.3f GFLOP/s", averageCombinedGPULatency * 1e3, combinedGPUThroughput))

  return PermuteAndConv3DProfileResult(
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

@main
struct convolution3d {
  static func main() {
    let device = makeDevice()
    let buildOptions = defaultBuildOptions()
    let profileDimensions = largeProfileDimensions()
    let profileOptions = largeProfileOptions()

    print("Running standalone Conv3D tensor-op research scaffold on \(device.name)")
    print("")

    validateTensorOpConv3D(device: device, buildOptions: buildOptions)
    print("")
    validateTensorOpConv3DBias(device: device, buildOptions: buildOptions)

    print("")
    let directProfile = profileTensorOpConv3D(
      device: device,
      dimensions: profileDimensions,
      buildOptions: buildOptions,
      options: profileOptions
    )
    if let gpuThroughput = directProfile.gpuThroughputGFLOPS {
      print(String(format: "Large Conv3D tensor-op GPU throughput: %.3f GFLOP/s", gpuThroughput))
    }

    print("")
    let biasProfile = profileTensorOpConv3DBias(
      device: device,
      dimensions: profileDimensions,
      buildOptions: buildOptions,
      options: profileOptions
    )
    if let gpuThroughput = biasProfile.gpuThroughputGFLOPS {
      print(String(format: "Large Conv3D tensor-op bias GPU throughput: %.3f GFLOP/s", gpuThroughput))
    }

    print("")
    let combinedProfile = profilePermutationAndConv3D(
      device: device,
      validationDimensions: validationPermutationDimensions(),
      profileDimensions: profileDimensions,
      buildOptions: buildOptions,
      options: profileOptions
    )
    if let combinedGPUThroughput = combinedProfile.combinedGPUThroughputGFLOPS {
      print(String(format: "Large permute+Conv3D GPU throughput: %.3f GFLOP/s", combinedGPUThroughput))
      if let directGPUThroughput = directProfile.gpuThroughputGFLOPS, directGPUThroughput > 0 {
        let overhead = 100 * (directGPUThroughput - combinedGPUThroughput) / directGPUThroughput
        print(String(format: "Permutation overhead on effective GPU throughput: %.3f%%", overhead))
      }
    }
  }
}
