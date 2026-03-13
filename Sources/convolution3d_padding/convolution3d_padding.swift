import Foundation
import Metal

/*
 Readable local Conv3D padding probe.

 This file exists to answer one narrow question cleanly:

 Can we express Conv3D horizontal padding through `convolution2d::set_offsets(...)`
 when Conv3D itself is decomposed into repeated spatial Conv2D tensor-op launches?

 Scope of this probe:
 - Conv3D is still implemented as host-side loops over kernel depth.
 - Spatial work is done by one readable inline Metal shader, not the composable
   source builders used by the shared iPad path.
 - Padding under test is:
   - padLeft = 1
   - padRight = 1
   - padTop = 0
   - padBottom = 0
 - The shader uses explicit cooperative-tensor load/store so edge handling stays
   readable and does not depend on `cOutput.load/store(...)` edge semantics.

 If this local probe validates, it gives us a clean baseline before touching the
 iPad path again.
*/

struct PaddedConv3DDimensions: CustomStringConvertible {
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
  var paddingTop: Int
  var paddingBottom: Int
  var paddingLeft: Int
  var paddingRight: Int

  var effectiveKernelDepth: Int {
    (kernelDepth - 1) * dilationZ + 1
  }

  var effectiveKernelHeight: Int {
    (kernelHeight - 1) * dilationY + 1
  }

  var effectiveKernelWidth: Int {
    (kernelWidth - 1) * dilationX + 1
  }

  var outputDepth: Int {
    ((inputDepth - effectiveKernelDepth) / strideZ) + 1
  }

  var outputHeight: Int {
    ((inputHeight + paddingTop + paddingBottom - effectiveKernelHeight) / strideY) + 1
  }

  var outputWidth: Int {
    ((inputWidth + paddingLeft + paddingRight - effectiveKernelWidth) / strideX) + 1
  }

  var description: String {
    "N=\(batchSize), D=\(inputDepth), H=\(inputHeight), W=\(inputWidth), C=\(inputChannels), O=\(outputChannels), KD=\(kernelDepth), KH=\(kernelHeight), KW=\(kernelWidth), SZ=\(strideZ), SY=\(strideY), SX=\(strideX), DZ=\(dilationZ), DY=\(dilationY), DX=\(dilationX), PT=\(paddingTop), PB=\(paddingBottom), PL=\(paddingLeft), PR=\(paddingRight)"
  }
}

struct SpatialSlice {
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
  var paddingTop: Int
  var paddingLeft: Int
  var outputHeight: Int
  var outputWidth: Int
}

struct BuildOptions {
  var executionSIMDGroups: Int
  var outputTileWidth: Int
  var outputTileHeight: Int
}

struct ValidationResult {
  var maxAbsoluteError: Float
  var mismatches: Int
}

struct ExecutionResult {
  var output: [Float]
}

enum Conv3DMode {
  case multiply
  case multiplyAccumulate

  var functionName: String {
    switch self {
    case .multiply:
      return "conv3d_padded_multiply"
    case .multiplyAccumulate:
      return "conv3d_padded_macc"
    }
  }
}

func spatialSlice(_ dimensions: PaddedConv3DDimensions) -> SpatialSlice {
  SpatialSlice(
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
    paddingLeft: dimensions.paddingLeft,
    outputHeight: dimensions.outputHeight,
    outputWidth: dimensions.outputWidth
  )
}

func makeDevice() -> MTLDevice {
  guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this machine.")
  }
  guard device.supportsFamily(.metal4) else {
    fatalError("This device does not support the tensor ops used by this probe.")
  }
  return device
}

func activationIndex(
  n: Int,
  z: Int,
  h: Int,
  w: Int,
  c: Int,
  dimensions: PaddedConv3DDimensions
) -> Int {
  ((((n * dimensions.inputDepth + z) * dimensions.inputHeight + h) * dimensions.inputWidth + w) * dimensions.inputChannels) + c
}

func weightIndex(
  kd: Int,
  kh: Int,
  kw: Int,
  ic: Int,
  oc: Int,
  dimensions: PaddedConv3DDimensions
) -> Int {
  ((((kd * dimensions.kernelHeight + kh) * dimensions.kernelWidth + kw) * dimensions.inputChannels + ic) * dimensions.outputChannels) + oc
}

func outputIndex(
  n: Int,
  z: Int,
  h: Int,
  w: Int,
  oc: Int,
  dimensions: PaddedConv3DDimensions
) -> Int {
  ((((n * dimensions.outputDepth + z) * dimensions.outputHeight + h) * dimensions.outputWidth + w) * dimensions.outputChannels) + oc
}

func createActivationData(dimensions: PaddedConv3DDimensions) -> [Float16] {
  let count = dimensions.batchSize * dimensions.inputDepth * dimensions.inputHeight * dimensions.inputWidth * dimensions.inputChannels
  return (0..<count).map {
    let value = Float((($0 * 7) + 3) % 29 - 14) * 0.0625
    return Float16(value)
  }
}

func createWeightData(dimensions: PaddedConv3DDimensions) -> [Float16] {
  let count = dimensions.kernelDepth * dimensions.kernelHeight * dimensions.kernelWidth * dimensions.inputChannels * dimensions.outputChannels
  return (0..<count).map {
    let value = Float((($0 * 5) + 1) % 23 - 11) * 0.03125
    return Float16(value)
  }
}

func referenceConvolution(
  activation: [Float16],
  weights: [Float16],
  dimensions: PaddedConv3DDimensions
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
                  if ih >= 0 && ih < dimensions.inputHeight && iw >= 0 && iw < dimensions.inputWidth {
                    for ic in 0..<dimensions.inputChannels {
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

func copyToSharedBuffer(device: MTLDevice, values: [Float16]) -> MTLBuffer {
  let size = values.count * MemoryLayout<Float16>.size
  guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
    fatalError("Could not allocate buffer.")
  }
  buffer.contents().copyMemory(from: values, byteCount: size)
  return buffer
}

func makeZeroedSharedBuffer(device: MTLDevice, count: Int) -> MTLBuffer {
  let size = count * MemoryLayout<Float16>.size
  guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
    fatalError("Could not allocate buffer.")
  }
  buffer.contents().initializeMemory(as: Float16.self, repeating: 0, count: count)
  return buffer
}

func readFloat16BufferAsFloat(_ buffer: MTLBuffer, count: Int) -> [Float] {
  let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
  return Array(UnsafeBufferPointer(start: pointer, count: count)).map(Float.init)
}

func validateOutput(
  actual: [Float],
  expected: [Float],
  dimensions: PaddedConv3DDimensions,
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

func createReadablePaddedConv3DSource(
  dimensions: PaddedConv3DDimensions,
  buildOptions: BuildOptions
) -> String {
  let slice = spatialSlice(dimensions)
  let inputTileWidth =
    (buildOptions.outputTileWidth - 1) * slice.strideX + (slice.kernelWidth - 1) * slice.dilationX + 1
  let inputTileHeight =
    (buildOptions.outputTileHeight - 1) * slice.strideY + (slice.kernelHeight - 1) * slice.dilationY + 1
  let baseOffsetX = ((slice.kernelWidth - 1) * slice.dilationX / 2) - slice.paddingLeft
  let baseOffsetY = ((slice.kernelHeight - 1) * slice.dilationY / 2) - slice.paddingTop

  return """

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// This probe keeps the shader intentionally direct:
// - one spatial Conv2D tensor-op kernel for the first depth slice
// - one spatial Conv2D tensor-op kernel for later depth slices
// - spatial padding is encoded through set_offsets(...)
// - edge tiles clamp the activation slice and compensate that shift in set_offsets(...)
// - accumulation uses explicit load/store so edge correctness does not depend on
//   cOutput.load/store(...) behavior

kernel void conv3d_padded_multiply(device half *activation_buf [[buffer(0)]],
                                   device half *weights_buf [[buffer(1)]],
                                   device half *output_buf [[buffer(2)]],
                                   constant uint& activation_base [[buffer(3)]],
                                   constant uint& weights_base [[buffer(4)]],
                                   constant uint& output_base_offset [[buffer(5)]],
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

  const int unclamped_input_origin_x = output_origin_x * \(slice.strideX);
  const int unclamped_input_origin_y = output_origin_y * \(slice.strideY);
  const int clamped_input_origin_x = min(unclamped_input_origin_x, max(0, \(slice.inputWidth - inputTileWidth)));
  const int clamped_input_origin_y = min(unclamped_input_origin_y, max(0, \(slice.inputHeight - inputTileHeight)));
  const int adjusted_offset_x = \(baseOffsetX) + (unclamped_input_origin_x - clamped_input_origin_x);
  const int adjusted_offset_y = \(baseOffsetY) + (unclamped_input_origin_y - clamped_input_origin_y);

  auto activation_base_tensor = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(\(slice.inputChannels), \(slice.inputWidth), \(slice.inputHeight), 1));
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
      convolution2d_descriptor::mode::multiply);
  convolution2d<descriptor, execution_simdgroups<\(buildOptions.executionSIMDGroups)>> conv2d_op;

  conv2d_op.set_offsets(int2(adjusted_offset_x, adjusted_offset_y));

  if (output_origin_x + \(buildOptions.outputTileWidth) <= \(slice.outputWidth) &&
      output_origin_y + \(buildOptions.outputTileHeight) <= \(slice.outputHeight)) {
    auto activation = activation_base_tensor.slice<\(slice.inputChannels), \(inputTileWidth), \(inputTileHeight), 1>(
        0,
        clamped_input_origin_x,
        clamped_input_origin_y,
        0);
    auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
      if (cOutput.is_valid_element(i)) {
        cOutput[i] = 0;
      }
    }
    conv2d_op.run(activation, weights, cOutput);
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
      if (cOutput.is_valid_element(i)) {
        auto idx = cOutput.get_multidimensional_index(i);
        const int ox = output_origin_x + int(idx[1]);
        const int oy = output_origin_y + int(idx[2]);
        if (ox < \(slice.outputWidth) && oy < \(slice.outputHeight)) {
          output_buf[((oy * \(slice.outputWidth) + ox) * \(slice.outputChannels)) + idx[0]] = cOutput[i];
        }
      }
    }
  } else {
    auto activation = activation_base_tensor.slice(
        0,
        clamped_input_origin_x,
        clamped_input_origin_y,
        0);
    auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
      if (cOutput.is_valid_element(i)) {
        cOutput[i] = 0;
      }
    }
    conv2d_op.run(activation, weights, cOutput);
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
      if (cOutput.is_valid_element(i)) {
        auto idx = cOutput.get_multidimensional_index(i);
        const int ox = output_origin_x + int(idx[1]);
        const int oy = output_origin_y + int(idx[2]);
        if (ox < \(slice.outputWidth) && oy < \(slice.outputHeight)) {
          output_buf[((oy * \(slice.outputWidth) + ox) * \(slice.outputChannels)) + idx[0]] = cOutput[i];
        }
      }
    }
  }
}

kernel void conv3d_padded_macc(device half *activation_buf [[buffer(0)]],
                               device half *weights_buf [[buffer(1)]],
                               device half *output_buf [[buffer(2)]],
                               constant uint& activation_base [[buffer(3)]],
                               constant uint& weights_base [[buffer(4)]],
                               constant uint& output_base_offset [[buffer(5)]],
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

  const int unclamped_input_origin_x = output_origin_x * \(slice.strideX);
  const int unclamped_input_origin_y = output_origin_y * \(slice.strideY);
  const int clamped_input_origin_x = min(unclamped_input_origin_x, max(0, \(slice.inputWidth - inputTileWidth)));
  const int clamped_input_origin_y = min(unclamped_input_origin_y, max(0, \(slice.inputHeight - inputTileHeight)));
  const int adjusted_offset_x = \(baseOffsetX) + (unclamped_input_origin_x - clamped_input_origin_x);
  const int adjusted_offset_y = \(baseOffsetY) + (unclamped_input_origin_y - clamped_input_origin_y);

  auto activation_base_tensor = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(\(slice.inputChannels), \(slice.inputWidth), \(slice.inputHeight), 1));
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
      convolution2d_descriptor::mode::multiply_accumulate);
  convolution2d<descriptor, execution_simdgroups<\(buildOptions.executionSIMDGroups)>> conv2d_op;
  conv2d_op.set_offsets(int2(adjusted_offset_x, adjusted_offset_y));

  if (output_origin_x + \(buildOptions.outputTileWidth) <= \(slice.outputWidth) &&
      output_origin_y + \(buildOptions.outputTileHeight) <= \(slice.outputHeight)) {
    auto activation = activation_base_tensor.slice<\(slice.inputChannels), \(inputTileWidth), \(inputTileHeight), 1>(
        0,
        clamped_input_origin_x,
        clamped_input_origin_y,
        0);
    auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
      if (cOutput.is_valid_element(i)) {
        auto idx = cOutput.get_multidimensional_index(i);
        const int ox = output_origin_x + int(idx[1]);
        const int oy = output_origin_y + int(idx[2]);
        if (ox < \(slice.outputWidth) && oy < \(slice.outputHeight)) {
          cOutput[i] = output_buf[((oy * \(slice.outputWidth) + ox) * \(slice.outputChannels)) + idx[0]];
        } else {
          cOutput[i] = 0;
        }
      }
    }
    conv2d_op.run(activation, weights, cOutput);
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
      if (cOutput.is_valid_element(i)) {
        auto idx = cOutput.get_multidimensional_index(i);
        const int ox = output_origin_x + int(idx[1]);
        const int oy = output_origin_y + int(idx[2]);
        if (ox < \(slice.outputWidth) && oy < \(slice.outputHeight)) {
          output_buf[((oy * \(slice.outputWidth) + ox) * \(slice.outputChannels)) + idx[0]] = cOutput[i];
        }
      }
    }
  } else {
    auto activation = activation_base_tensor.slice(
        0,
        clamped_input_origin_x,
        clamped_input_origin_y,
        0);
    auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
      if (cOutput.is_valid_element(i)) {
        auto idx = cOutput.get_multidimensional_index(i);
        const int ox = output_origin_x + int(idx[1]);
        const int oy = output_origin_y + int(idx[2]);
        if (ox < \(slice.outputWidth) && oy < \(slice.outputHeight)) {
          cOutput[i] = output_buf[((oy * \(slice.outputWidth) + ox) * \(slice.outputChannels)) + idx[0]];
        } else {
          cOutput[i] = 0;
        }
      }
    }
    conv2d_op.run(activation, weights, cOutput);
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
      if (cOutput.is_valid_element(i)) {
        auto idx = cOutput.get_multidimensional_index(i);
        const int ox = output_origin_x + int(idx[1]);
        const int oy = output_origin_y + int(idx[2]);
        if (ox < \(slice.outputWidth) && oy < \(slice.outputHeight)) {
          output_buf[((oy * \(slice.outputWidth) + ox) * \(slice.outputChannels)) + idx[0]] = cOutput[i];
        }
      }
    }
  }
}

"""
}

final class ReadablePaddedConv3DSession {
  private let dimensions: PaddedConv3DDimensions
  private let slice: SpatialSlice
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
    dimensions: PaddedConv3DDimensions,
    buildOptions: BuildOptions,
    activation: [Float16],
    weights: [Float16]
  ) throws {
    self.dimensions = dimensions
    self.slice = spatialSlice(dimensions)
    self.buildOptions = buildOptions

    guard let commandQueue = device.makeCommandQueue() else {
      throw NSError(domain: "convolution3d_padding", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not create command queue."])
    }
    self.commandQueue = commandQueue

    let library = try device.makeLibrary(
      source: createReadablePaddedConv3DSource(dimensions: dimensions, buildOptions: buildOptions),
      options: nil
    )
    guard let multiplyFunction = library.makeFunction(name: Conv3DMode.multiply.functionName),
          let accumulateFunction = library.makeFunction(name: Conv3DMode.multiplyAccumulate.functionName) else {
      throw NSError(domain: "convolution3d_padding", code: 2, userInfo: [NSLocalizedDescriptionKey: "Could not create padded Conv3D functions."])
    }

    multiplyPipelineState = try device.makeComputePipelineState(function: multiplyFunction)
    accumulatePipelineState = try device.makeComputePipelineState(function: accumulateFunction)

    outputCount = dimensions.batchSize * dimensions.outputDepth * dimensions.outputHeight * dimensions.outputWidth * dimensions.outputChannels
    activationBuffer = copyToSharedBuffer(device: device, values: activation)
    weightBuffer = copyToSharedBuffer(device: device, values: weights)
    outputBuffer = makeZeroedSharedBuffer(device: device, count: outputCount)
  }

  func run() -> ExecutionResult {
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create command buffer.")
    }

    let threadgroups = MTLSize(
      width: (slice.outputWidth + buildOptions.outputTileWidth - 1) / buildOptions.outputTileWidth,
      height: (slice.outputHeight + buildOptions.outputTileHeight - 1) / buildOptions.outputTileHeight,
      depth: 1
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

    encoder.useResource(activationBuffer, usage: .read)
    encoder.useResource(weightBuffer, usage: .read)
    encoder.useResource(outputBuffer, usage: [.read, .write])

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

          encoder.setComputePipelineState(pipelineState)
          encoder.setBuffer(activationBuffer, offset: 0, index: 0)
          encoder.setBuffer(weightBuffer, offset: 0, index: 1)
          encoder.setBuffer(outputBuffer, offset: 0, index: 2)
          var activationBaseVar = activationBase
          var weightsBaseVar = weightsBase
          var outputBaseVar = outputBase
          encoder.setBytes(&activationBaseVar, length: MemoryLayout<UInt32>.size, index: 3)
          encoder.setBytes(&weightsBaseVar, length: MemoryLayout<UInt32>.size, index: 4)
          encoder.setBytes(&outputBaseVar, length: MemoryLayout<UInt32>.size, index: 5)
          encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        }
      }
    }

    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if commandBuffer.status != .completed {
      let errorDescription = commandBuffer.error.map { "\($0)" } ?? "none"
      fatalError("Padded Conv3D probe failed. error=\(errorDescription)")
    }

    return ExecutionResult(output: readFloat16BufferAsFloat(outputBuffer, count: outputCount))
  }
}

func probeBuildOptions() -> [(label: String, options: BuildOptions)] {
  [
    (
      label: "single oversized tile",
      options: BuildOptions(executionSIMDGroups: 1, outputTileWidth: 8, outputTileHeight: 8)
    ),
    (
      label: "tiled 4x4",
      options: BuildOptions(executionSIMDGroups: 1, outputTileWidth: 4, outputTileHeight: 4)
    ),
  ]
}

func defaultValidationCases() -> [PaddedConv3DDimensions] {
  [
    PaddedConv3DDimensions(
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
      paddingTop: 0,
      paddingBottom: 0,
      paddingLeft: 1,
      paddingRight: 1
    ),
    PaddedConv3DDimensions(
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
      paddingTop: 0,
      paddingBottom: 0,
      paddingLeft: 1,
      paddingRight: 1
    ),
    PaddedConv3DDimensions(
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
      paddingTop: 0,
      paddingBottom: 0,
      paddingLeft: 1,
      paddingRight: 1
    ),
  ]
}

@main
struct convolution3d_padding {
  static func main() {
    let device = makeDevice()
    print("Device: \(device.name)")
    print("Readable local Conv3D padding probe")
    print("Shader strategy: single readable inline shader, clamped edge slice + adjusted set_offsets, explicit cooperative-tensor load/store")
    print("")

    for build in probeBuildOptions() {
      print("Build: \(build.label)")
      print("executionSIMDGroups=\(build.options.executionSIMDGroups), tile=\(build.options.outputTileWidth)x\(build.options.outputTileHeight)")
      print("")

      var allPassed = true
      for (index, dimensions) in defaultValidationCases().enumerated() {
        let activation = createActivationData(dimensions: dimensions)
        let weights = createWeightData(dimensions: dimensions)
        let expected = referenceConvolution(
          activation: activation,
          weights: weights,
          dimensions: dimensions
        )

        let session: ReadablePaddedConv3DSession
        do {
          session = try ReadablePaddedConv3DSession(
            device: device,
            dimensions: dimensions,
            buildOptions: build.options,
            activation: activation,
            weights: weights
          )
        } catch {
          fatalError("Could not create readable padded Conv3D session: \(error)")
        }

        let result = session.run()
        let validation = validateOutput(
          actual: result.output,
          expected: expected,
          dimensions: dimensions,
          label: "Readable padded Conv3D"
        )
        if validation.mismatches > 0 {
          allPassed = false
        }

        print("Case \(index + 1)/\(defaultValidationCases().count): \(dimensions)")
        print("Max absolute error: \(validation.maxAbsoluteError)")
        print("Mismatches above tolerance: \(validation.mismatches)")
        print("")
      }

      print("All cases passed for \(build.label): \(allPassed)")
      print("")
    }
  }
}
