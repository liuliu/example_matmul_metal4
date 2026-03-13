import Foundation
import Metal

private struct ProbeCase {
  var label: String
  var offsetX: Int
  var offsetY: Int
}

private struct ShiftMatch {
  var sampleDeltaX: Int
  var sampleDeltaY: Int
  var mismatches: Int
  var maxAbsoluteError: Float
}

private let inputWidth = 5
private let inputHeight = 5
private let kernelWidth = 3
private let kernelHeight = 3
private let outputWidth = 5
private let outputHeight = 5

private func createProbeSource(offsetX: Int, offsetY: Int) -> String {
  """

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void conv2d_offset_probe(device half *activation_buf [[buffer(0)]],
                                device half *weights_buf [[buffer(1)]],
                                device half *output_buf [[buffer(2)]])
{
  auto activation = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(1, \(inputWidth), \(inputHeight), 1));
  auto weights = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      weights_buf,
      dextents<int32_t, 4>(1, 1, \(kernelWidth), \(kernelHeight)));
  auto output = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      output_buf,
      dextents<int32_t, 4>(1, \(outputWidth), \(outputHeight), 1));

  constexpr auto descriptor = convolution2d_descriptor(
      int4(1, \(outputWidth), \(outputHeight), 1),
      int4(1, \(inputWidth), \(inputHeight), 1),
      int2(\(kernelWidth), \(kernelHeight)),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(1, 1),
      int2(1, 1),
      1,
      false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<descriptor, execution_simdgroups<1>> conv2d_op;
  conv2d_op.set_offsets(int2(\(offsetX), \(offsetY)));

  auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
  #pragma clang loop unroll(full)
  for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
    if (cOutput.is_valid_element(i)) {
      cOutput[i] = 0;
    }
  }
  conv2d_op.run(activation, weights, cOutput);
  cOutput.store(output);
}

"""
}

private func createTiledProbeSource(
  offsetX: Int,
  offsetY: Int,
  tileWidth: Int,
  tileHeight: Int,
  subtractPaddingFromInputOrigin: Bool
) -> String {
  let inputOriginAdjustmentX = subtractPaddingFromInputOrigin ? "- 1" : ""
  let inputOriginAdjustmentY = subtractPaddingFromInputOrigin ? "- 1" : ""
  let inputTileWidth = (tileWidth - 1) + (kernelWidth - 1) + 1
  let inputTileHeight = (tileHeight - 1) + (kernelHeight - 1) + 1
  return """

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void conv2d_tiled_offset_probe(device half *activation_buf [[buffer(0)]],
                                      device half *weights_buf [[buffer(1)]],
                                      device half *output_buf [[buffer(2)]],
                                      uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]])
{
  const int output_origin_x = int(threadgroup_position_in_grid.x) * \(tileWidth);
  const int output_origin_y = int(threadgroup_position_in_grid.y) * \(tileHeight);
  if (output_origin_x >= \(outputWidth) || output_origin_y >= \(outputHeight)) {
    return;
  }

  const int input_origin_x = output_origin_x \(inputOriginAdjustmentX);
  const int input_origin_y = output_origin_y \(inputOriginAdjustmentY);

  auto activation_base = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(1, \(inputWidth), \(inputHeight), 1));
  auto weights = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      weights_buf,
      dextents<int32_t, 4>(1, 1, \(kernelWidth), \(kernelHeight)));
  auto output_base = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      output_buf,
      dextents<int32_t, 4>(1, \(outputWidth), \(outputHeight), 1));

  constexpr auto descriptor = convolution2d_descriptor(
      int4(1, \(tileWidth), \(tileHeight), 1),
      int4(1, \(inputTileWidth), \(inputTileHeight), 1),
      int2(\(kernelWidth), \(kernelHeight)),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(1, 1),
      int2(1, 1),
      1,
      false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<descriptor, execution_simdgroups<1>> conv2d_op;
  conv2d_op.set_offsets(int2(\(offsetX), \(offsetY)));

  if (output_origin_x + \(tileWidth) <= \(outputWidth) &&
      output_origin_y + \(tileHeight) <= \(outputHeight)) {
    auto activation = activation_base.slice<1, \(inputTileWidth), \(inputTileHeight), 1>(
        0,
        input_origin_x,
        input_origin_y,
        0);
    auto output = output_base.slice<1, \(tileWidth), \(tileHeight), 1>(
        0,
        output_origin_x,
        output_origin_y,
        0);
    auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
      if (cOutput.is_valid_element(i)) {
        cOutput[i] = 0;
      }
    }
    conv2d_op.run(activation, weights, cOutput);
    cOutput.store(output);
  } else {
    auto activation = activation_base.slice(
        0,
        input_origin_x,
        input_origin_y,
        0);
    auto output = output_base.slice(
        0,
        output_origin_x,
        output_origin_y,
        0);
    auto cOutput = conv2d_op.get_destination_cooperative_tensor<decltype(activation), decltype(weights), half>();
    #pragma clang loop unroll(full)
    for (unsigned short i = 0; i < cOutput.get_capacity(); ++i) {
      if (cOutput.is_valid_element(i)) {
        cOutput[i] = 0;
      }
    }
    conv2d_op.run(activation, weights, cOutput);
    cOutput.store(output);
  }
}

"""
}

private func makeBuffer(device: MTLDevice, values: [Float16]) -> MTLBuffer {
  let size = values.count * MemoryLayout<Float16>.size
  guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
    fatalError("Could not allocate buffer.")
  }
  buffer.contents().copyMemory(from: values, byteCount: size)
  return buffer
}

private func makeZeroedBuffer(device: MTLDevice, count: Int) -> MTLBuffer {
  let size = count * MemoryLayout<Float16>.size
  guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
    fatalError("Could not allocate output buffer.")
  }
  buffer.contents().initializeMemory(as: Float16.self, repeating: 0, count: count)
  return buffer
}

private func readBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
  let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
  return Array(UnsafeBufferPointer(start: pointer, count: count)).map(Float.init)
}

private func activationData() -> [Float16] {
  (0..<(inputHeight * inputWidth)).map { Float16(Float($0 + 1)) }
}

private func centerTapWeights() -> [Float16] {
  var weights = [Float16](repeating: 0, count: kernelWidth * kernelHeight)
  weights[kernelWidth + 1] = 1
  return weights
}

private func denseWeights() -> [Float16] {
  (0..<(kernelWidth * kernelHeight)).map { Float16(Float($0 + 1)) }
}

private func runProbe(device: MTLDevice, probeCase: ProbeCase, activation: [Float16], weights: [Float16]) -> [Float] {
  guard let commandQueue = device.makeCommandQueue() else {
    fatalError("Could not create command queue.")
  }
  let library: MTLLibrary
  do {
    library = try device.makeLibrary(source: createProbeSource(offsetX: probeCase.offsetX, offsetY: probeCase.offsetY), options: nil)
  } catch {
    fatalError("Could not compile probe source for \(probeCase.label): \(error)")
  }
  guard let function = library.makeFunction(name: "conv2d_offset_probe") else {
    fatalError("Could not create probe function.")
  }
  let pipelineState: MTLComputePipelineState
  do {
    pipelineState = try device.makeComputePipelineState(function: function)
  } catch {
    fatalError("Could not create probe pipeline state: \(error)")
  }

  let activationBuffer = makeBuffer(device: device, values: activation)
  let weightBuffer = makeBuffer(device: device, values: weights)
  let outputBuffer = makeZeroedBuffer(device: device, count: outputWidth * outputHeight)

  guard let commandBuffer = commandQueue.makeCommandBuffer(),
        let encoder = commandBuffer.makeComputeCommandEncoder() else {
    fatalError("Could not create command buffer.")
  }
  encoder.setComputePipelineState(pipelineState)
  encoder.setBuffer(activationBuffer, offset: 0, index: 0)
  encoder.setBuffer(weightBuffer, offset: 0, index: 1)
  encoder.setBuffer(outputBuffer, offset: 0, index: 2)
  encoder.dispatchThreadgroups(
    MTLSize(width: 1, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: pipelineState.threadExecutionWidth, height: 1, depth: 1)
  )
  encoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if commandBuffer.status != .completed {
    let errorDescription = commandBuffer.error.map { "\($0)" } ?? "none"
    fatalError("Probe did not complete successfully. error=\(errorDescription)")
  }

  return readBuffer(outputBuffer, count: outputWidth * outputHeight)
}

private func runTiledProbe(
  device: MTLDevice,
  label: String,
  offsetX: Int,
  offsetY: Int,
  tileWidth: Int,
  tileHeight: Int,
  subtractPaddingFromInputOrigin: Bool,
  activation: [Float16],
  weights: [Float16]
) -> [Float] {
  guard let commandQueue = device.makeCommandQueue() else {
    fatalError("Could not create command queue.")
  }
  let library: MTLLibrary
  do {
    library = try device.makeLibrary(
      source: createTiledProbeSource(
        offsetX: offsetX,
        offsetY: offsetY,
        tileWidth: tileWidth,
        tileHeight: tileHeight,
        subtractPaddingFromInputOrigin: subtractPaddingFromInputOrigin
      ),
      options: nil
    )
  } catch {
    fatalError("Could not compile tiled probe source for \(label): \(error)")
  }
  guard let function = library.makeFunction(name: "conv2d_tiled_offset_probe") else {
    fatalError("Could not create tiled probe function.")
  }
  let pipelineState: MTLComputePipelineState
  do {
    pipelineState = try device.makeComputePipelineState(function: function)
  } catch {
    fatalError("Could not create tiled probe pipeline state: \(error)")
  }

  let activationBuffer = makeBuffer(device: device, values: activation)
  let weightBuffer = makeBuffer(device: device, values: weights)
  let outputBuffer = makeZeroedBuffer(device: device, count: outputWidth * outputHeight)

  guard let commandBuffer = commandQueue.makeCommandBuffer(),
        let encoder = commandBuffer.makeComputeCommandEncoder() else {
    fatalError("Could not create tiled command buffer.")
  }
  encoder.setComputePipelineState(pipelineState)
  encoder.setBuffer(activationBuffer, offset: 0, index: 0)
  encoder.setBuffer(weightBuffer, offset: 0, index: 1)
  encoder.setBuffer(outputBuffer, offset: 0, index: 2)
  encoder.dispatchThreadgroups(
    MTLSize(width: (outputWidth + tileWidth - 1) / tileWidth, height: (outputHeight + tileHeight - 1) / tileHeight, depth: 1),
    threadsPerThreadgroup: MTLSize(width: pipelineState.threadExecutionWidth, height: 1, depth: 1)
  )
  encoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if commandBuffer.status != .completed {
    let errorDescription = commandBuffer.error.map { "\($0)" } ?? "none"
    fatalError("Tiled probe did not complete successfully for \(label). error=\(errorDescription)")
  }

  return readBuffer(outputBuffer, count: outputWidth * outputHeight)
}

private func inferShift(output: [Float], activation: [Float]) -> ShiftMatch {
  var best = ShiftMatch(sampleDeltaX: 0, sampleDeltaY: 0, mismatches: Int.max, maxAbsoluteError: .greatestFiniteMagnitude)
  for dy in -2...2 {
    for dx in -2...2 {
      var mismatches = 0
      var maxAbsoluteError: Float = 0
      for oh in 0..<outputHeight {
        for ow in 0..<outputWidth {
          let ih = oh + dy
          let iw = ow + dx
          let expected: Float
          if ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth {
            expected = activation[ih * inputWidth + iw]
          } else {
            expected = 0
          }
          let actual = output[oh * outputWidth + ow]
          let error = abs(actual - expected)
          maxAbsoluteError = max(maxAbsoluteError, error)
          if error > 0.001 {
            mismatches += 1
          }
        }
      }
      if mismatches < best.mismatches || (mismatches == best.mismatches && maxAbsoluteError < best.maxAbsoluteError) {
        best = ShiftMatch(sampleDeltaX: dx, sampleDeltaY: dy, mismatches: mismatches, maxAbsoluteError: maxAbsoluteError)
      }
    }
  }
  return best
}

private func printMatrix(_ values: [Float], width: Int, height: Int) {
  for y in 0..<height {
    let row = (0..<width).map { String(format: "%5.0f", values[y * width + $0]) }.joined(separator: " ")
    print(row)
  }
}

private func referenceForOffset(activation: [Float], weights: [Float], offsetX: Int, offsetY: Int) -> [Float] {
  let centerX = kernelWidth / 2
  let centerY = kernelHeight / 2
  var output = [Float](repeating: 0, count: outputWidth * outputHeight)
  for oh in 0..<outputHeight {
    for ow in 0..<outputWidth {
      var sum: Float = 0
      for kh in 0..<kernelHeight {
        for kw in 0..<kernelWidth {
          let ih = oh + kh + offsetY - centerY
          let iw = ow + kw + offsetX - centerX
          if ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth {
            sum += activation[ih * inputWidth + iw] * weights[kh * kernelWidth + kw]
          }
        }
      }
      output[oh * outputWidth + ow] = sum
    }
  }
  return output
}

private func compare(_ lhs: [Float], _ rhs: [Float]) -> (mismatches: Int, maxAbsoluteError: Float) {
  var mismatches = 0
  var maxAbsoluteError: Float = 0
  for i in 0..<lhs.count {
    let error = abs(lhs[i] - rhs[i])
    maxAbsoluteError = max(maxAbsoluteError, error)
    if error > 0.001 {
      mismatches += 1
    }
  }
  return (mismatches, maxAbsoluteError)
}

@main
struct convolution {
  static func main() {
    guard let device = MTLCreateSystemDefaultDevice() else {
      fatalError("Metal is not supported on this machine.")
    }

    let activationHalf = activationData()
    let activation = activationHalf.map(Float.init)
    let weights = centerTapWeights()
    let cases: [ProbeCase] = [
      ProbeCase(label: "offset(0,0)", offsetX: 0, offsetY: 0),
      ProbeCase(label: "offset(1,1)", offsetX: 1, offsetY: 1),
      ProbeCase(label: "offset(2,1)", offsetX: 2, offsetY: 1),
      ProbeCase(label: "offset(1,2)", offsetX: 1, offsetY: 2),
      ProbeCase(label: "offset(2,2)", offsetX: 2, offsetY: 2),
    ]

    print("Device: \(device.name)")
    print("Conv2D set_offsets probe")
    print("Activation: 5x5, values 1...25")
    print("Weights: 3x3 center tap = 1, all others = 0")
    print("Descriptor: output 5x5, input 5x5, stride 1, dilation 1, no explicit output-store offset")
    print("")
    print("Activation matrix")
    printMatrix(activation, width: inputWidth, height: inputHeight)

    for probeCase in cases {
      let output = runProbe(device: device, probeCase: probeCase, activation: activationHalf, weights: weights)
      let shift = inferShift(output: output, activation: activation)
      print("")
      print("=== \(probeCase.label) ===")
      printMatrix(output, width: outputWidth, height: outputHeight)
      print(
        "Best fit: output[h,w] = input[h + \(shift.sampleDeltaY), w + \(shift.sampleDeltaX)] with zero fill; mismatches=\(shift.mismatches), maxError=\(shift.maxAbsoluteError)"
      )
    }

    let denseWeightHalf = denseWeights()
    let denseWeight = denseWeightHalf.map(Float.init)
    let denseOffsetCase = ProbeCase(label: "dense-kernel offset(0,0)", offsetX: 0, offsetY: 0)
    let denseOutput = runProbe(device: device, probeCase: denseOffsetCase, activation: activationHalf, weights: denseWeightHalf)
    let denseReference = referenceForOffset(activation: activation, weights: denseWeight, offsetX: 0, offsetY: 0)
    let denseComparison = compare(denseOutput, denseReference)
    print("")
    print("=== dense-kernel offset(0,0) vs CPU reference ===")
    print("Kernel:")
    printMatrix(denseWeight, width: kernelWidth, height: kernelHeight)
    print("GPU output:")
    printMatrix(denseOutput, width: outputWidth, height: outputHeight)
    print("CPU reference:")
    printMatrix(denseReference, width: outputWidth, height: outputHeight)
    print(
      "Comparison: mismatches=\(denseComparison.mismatches), maxError=\(denseComparison.maxAbsoluteError)"
    )

    let tiledReference = referenceForOffset(activation: activation, weights: denseWeight, offsetX: 0, offsetY: 0)
    let tiledCases: [(String, Int, Int, Bool)] = [
      ("tiled offset(0,0), inputOrigin=outputOrigin", 0, 0, false),
      ("tiled offset(1,1), inputOrigin=outputOrigin", 1, 1, false),
      ("tiled offset(0,0), inputOrigin=outputOrigin-pad", 0, 0, true),
      ("tiled offset(1,1), inputOrigin=outputOrigin-pad", 1, 1, true),
      ("tiled offset(2,2), inputOrigin=outputOrigin-pad", 2, 2, true),
    ]
    for (label, offsetX, offsetY, subtractPaddingFromInputOrigin) in tiledCases {
      let tiledOutput = runTiledProbe(
        device: device,
        label: label,
        offsetX: offsetX,
        offsetY: offsetY,
        tileWidth: 2,
        tileHeight: 2,
        subtractPaddingFromInputOrigin: subtractPaddingFromInputOrigin,
        activation: activationHalf,
        weights: denseWeightHalf
      )
      let tiledComparison = compare(tiledOutput, tiledReference)
      print("")
      print("=== \(label) ===")
      printMatrix(tiledOutput, width: outputWidth, height: outputHeight)
      print(
        "Comparison to CPU same-padding reference: mismatches=\(tiledComparison.mismatches), maxError=\(tiledComparison.maxAbsoluteError)"
      )
    }
  }
}
