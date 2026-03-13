import Foundation
import Metal

struct Conv2DCase: CustomStringConvertible {
  let inputHeight: Int
  let inputWidth: Int
  let inputChannels: Int
  let outputChannels: Int
  let kernelHeight: Int
  let kernelWidth: Int
  let strideY: Int
  let strideX: Int
  let dilationY: Int
  let dilationX: Int
  let paddingTop: Int
  let paddingBottom: Int
  let paddingLeft: Int
  let paddingRight: Int
  let tileHeight: Int
  let tileWidth: Int

  var effectiveKernelHeight: Int {
    (kernelHeight - 1) * dilationY + 1
  }

  var effectiveKernelWidth: Int {
    (kernelWidth - 1) * dilationX + 1
  }

  var outputHeight: Int {
    ((inputHeight + paddingTop + paddingBottom - effectiveKernelHeight) / strideY) + 1
  }

  var outputWidth: Int {
    ((inputWidth + paddingLeft + paddingRight - effectiveKernelWidth) / strideX) + 1
  }

  var description: String {
    "H=\(inputHeight), W=\(inputWidth), C=\(inputChannels), O=\(outputChannels), KH=\(kernelHeight), KW=\(kernelWidth), SY=\(strideY), SX=\(strideX), DY=\(dilationY), DX=\(dilationX), PT=\(paddingTop), PB=\(paddingBottom), PL=\(paddingLeft), PR=\(paddingRight), tile=\(tileHeight)x\(tileWidth)"
  }
}

func makeDevice() -> MTLDevice {
  guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this machine.")
  }
  guard device.supportsFamily(.metal4) else {
    fatalError("This machine does not support Metal 4 tensor ops.")
  }
  return device
}

func activationIndex(h: Int, w: Int, c: Int, dims: Conv2DCase) -> Int {
  ((h * dims.inputWidth + w) * dims.inputChannels) + c
}

func weightIndex(kh: Int, kw: Int, ic: Int, oc: Int, dims: Conv2DCase) -> Int {
  (((kh * dims.kernelWidth + kw) * dims.inputChannels + ic) * dims.outputChannels) + oc
}

func outputIndex(h: Int, w: Int, oc: Int, dims: Conv2DCase) -> Int {
  ((h * dims.outputWidth + w) * dims.outputChannels) + oc
}

func activationData(dims: Conv2DCase) -> [Float16] {
  let count = dims.inputHeight * dims.inputWidth * dims.inputChannels
  return (0..<count).map { index in
    Float16(Float(((index * 7) + 3) % 29 - 14) * 0.0625)
  }
}

func weightData(dims: Conv2DCase) -> [Float16] {
  let count = dims.kernelHeight * dims.kernelWidth * dims.inputChannels * dims.outputChannels
  return (0..<count).map { index in
    Float16(Float(((index * 5) + 1) % 23 - 11) * 0.03125)
  }
}

func referenceConvolution(
  activation: [Float16],
  weights: [Float16],
  dims: Conv2DCase
) -> [Float] {
  var output = [Float](repeating: 0, count: dims.outputHeight * dims.outputWidth * dims.outputChannels)
  for oh in 0..<dims.outputHeight {
    for ow in 0..<dims.outputWidth {
      for oc in 0..<dims.outputChannels {
        var sum: Float = 0
        for kh in 0..<dims.kernelHeight {
          let ih = oh * dims.strideY + kh * dims.dilationY - dims.paddingTop
          for kw in 0..<dims.kernelWidth {
            let iw = ow * dims.strideX + kw * dims.dilationX - dims.paddingLeft
            if ih >= 0 && ih < dims.inputHeight && iw >= 0 && iw < dims.inputWidth {
              for ic in 0..<dims.inputChannels {
                let a = Float(activation[activationIndex(h: ih, w: iw, c: ic, dims: dims)])
                let w = Float(weights[weightIndex(kh: kh, kw: kw, ic: ic, oc: oc, dims: dims)])
                sum += a * w
              }
            }
          }
        }
        output[outputIndex(h: oh, w: ow, oc: oc, dims: dims)] = sum
      }
    }
  }
  return output
}

func createSource(dims: Conv2DCase) -> String {
  let inputTileHeight = (dims.tileHeight - 1) * dims.strideY + (dims.kernelHeight - 1) * dims.dilationY + 1
  let inputTileWidth = (dims.tileWidth - 1) * dims.strideX + (dims.kernelWidth - 1) * dims.dilationX + 1
  let baseOffsetY = (dims.kernelHeight - 1) * dims.dilationY / 2
  let baseOffsetX = (dims.kernelWidth - 1) * dims.dilationX / 2

  return """
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void conv2d_padding_probe(
    device half *activation_buf [[buffer(0)]],
    device half *weights_buf [[buffer(1)]],
    device half *output_buf [[buffer(2)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
  const int output_origin_x = int(tgid.x) * \(dims.tileWidth);
  const int output_origin_y = int(tgid.y) * \(dims.tileHeight);
  if (output_origin_x >= \(dims.outputWidth) || output_origin_y >= \(dims.outputHeight)) {
    return;
  }

  const int unclamped_input_origin_x = output_origin_x * \(dims.strideX) - \(dims.paddingLeft);
  const int unclamped_input_origin_y = output_origin_y * \(dims.strideY) - \(dims.paddingTop);
  const int clamped_input_origin_x = max(0, min(unclamped_input_origin_x, max(0, \(dims.inputWidth - inputTileWidth))));
  const int clamped_input_origin_y = max(0, min(unclamped_input_origin_y, max(0, \(dims.inputHeight - inputTileHeight))));
  const int adjusted_offset_x = \(baseOffsetX) + (unclamped_input_origin_x - clamped_input_origin_x);
  const int adjusted_offset_y = \(baseOffsetY) + (unclamped_input_origin_y - clamped_input_origin_y);

  auto activation_base = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      activation_buf,
      dextents<int32_t, 4>(\(dims.inputChannels), \(dims.inputWidth), \(dims.inputHeight), 1));
  auto output_base = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      output_buf,
      dextents<int32_t, 4>(\(dims.outputChannels), \(dims.outputWidth), \(dims.outputHeight), 1));
  auto weights = tensor<device half, dextents<int32_t, 4>, tensor_inline>(
      weights_buf,
      dextents<int32_t, 4>(\(dims.outputChannels), \(dims.inputChannels), \(dims.kernelWidth), \(dims.kernelHeight)));

  constexpr auto descriptor = convolution2d_descriptor(
      int4(\(dims.outputChannels), \(dims.tileWidth), \(dims.tileHeight), 1),
      int4(\(dims.inputChannels), \(inputTileWidth), \(inputTileHeight), 1),
      int2(\(dims.kernelWidth), \(dims.kernelHeight)),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(\(dims.strideX), \(dims.strideY)),
      int2(\(dims.dilationX), \(dims.dilationY)),
      1,
      false,
      convolution2d_descriptor::mode::multiply);
  convolution2d<descriptor, execution_simdgroups<1>> conv2d_op;
  conv2d_op.set_offsets(int2(adjusted_offset_x, adjusted_offset_y));

  if (output_origin_x + \(dims.tileWidth) <= \(dims.outputWidth) &&
      output_origin_y + \(dims.tileHeight) <= \(dims.outputHeight)) {
    auto activation = activation_base.slice<\(dims.inputChannels), \(inputTileWidth), \(inputTileHeight), 1>(
        0,
        clamped_input_origin_x,
        clamped_input_origin_y,
        0);
    auto output = output_base.slice<\(dims.outputChannels), \(dims.tileWidth), \(dims.tileHeight), 1>(
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
        clamped_input_origin_x,
        clamped_input_origin_y,
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

func makeSharedBuffer(device: MTLDevice, values: [Float16]) -> MTLBuffer {
  let size = values.count * MemoryLayout<Float16>.size
  guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
    fatalError("Could not allocate shared buffer.")
  }
  buffer.contents().copyMemory(from: values, byteCount: size)
  return buffer
}

func makeZeroedBuffer(device: MTLDevice, count: Int) -> MTLBuffer {
  let size = count * MemoryLayout<Float16>.size
  guard let buffer = device.makeBuffer(length: size, options: .storageModeShared) else {
    fatalError("Could not allocate output buffer.")
  }
  buffer.contents().initializeMemory(as: Float16.self, repeating: 0, count: count)
  return buffer
}

func readBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
  let pointer = buffer.contents().bindMemory(to: Float16.self, capacity: count)
  return Array(UnsafeBufferPointer(start: pointer, count: count)).map(Float.init)
}

func runCase(device: MTLDevice, dims: Conv2DCase) -> [Float] {
  let activation = activationData(dims: dims)
  let weights = weightData(dims: dims)

  let library: MTLLibrary
  do {
    library = try device.makeLibrary(source: createSource(dims: dims), options: nil)
  } catch {
    fatalError("Could not compile shader for \(dims): \(error)")
  }

  guard let function = library.makeFunction(name: "conv2d_padding_probe") else {
    fatalError("Could not create conv2d_padding_probe.")
  }

  let pipelineState: MTLComputePipelineState
  do {
    pipelineState = try device.makeComputePipelineState(function: function)
  } catch {
    fatalError("Could not create pipeline state for \(dims): \(error)")
  }

  guard let commandQueue = device.makeCommandQueue(),
        let commandBuffer = commandQueue.makeCommandBuffer(),
        let encoder = commandBuffer.makeComputeCommandEncoder() else {
    fatalError("Could not create command buffer objects.")
  }

  let outputCount = dims.outputHeight * dims.outputWidth * dims.outputChannels
  let activationBuffer = makeSharedBuffer(device: device, values: activation)
  let weightBuffer = makeSharedBuffer(device: device, values: weights)
  let outputBuffer = makeZeroedBuffer(device: device, count: outputCount)

  encoder.setComputePipelineState(pipelineState)
  encoder.setBuffer(activationBuffer, offset: 0, index: 0)
  encoder.setBuffer(weightBuffer, offset: 0, index: 1)
  encoder.setBuffer(outputBuffer, offset: 0, index: 2)
  encoder.dispatchThreadgroups(
    MTLSize(
      width: (dims.outputWidth + dims.tileWidth - 1) / dims.tileWidth,
      height: (dims.outputHeight + dims.tileHeight - 1) / dims.tileHeight,
      depth: 1
    ),
    threadsPerThreadgroup: MTLSize(width: pipelineState.threadExecutionWidth, height: 1, depth: 1)
  )
  encoder.endEncoding()
  commandBuffer.commit()
  commandBuffer.waitUntilCompleted()

  if commandBuffer.status != .completed {
    let message = commandBuffer.error.map { "\($0)" } ?? "none"
    fatalError("Probe failed for \(dims). error=\(message)")
  }

  return readBuffer(outputBuffer, count: outputCount)
}

func validate(actual: [Float], expected: [Float], dims: Conv2DCase) -> (maxError: Float, mismatches: Int) {
  var maxError: Float = 0
  var mismatches = 0
  for oh in 0..<dims.outputHeight {
    for ow in 0..<dims.outputWidth {
      for oc in 0..<dims.outputChannels {
        let index = outputIndex(h: oh, w: ow, oc: oc, dims: dims)
        let error = abs(actual[index] - expected[index])
        maxError = max(maxError, error)
        if error > 2e-2 {
          mismatches += 1
          if mismatches <= 8 {
            print("mismatch at h=\(oh), w=\(ow), o=\(oc): expected \(expected[index]), got \(actual[index])")
          }
        }
      }
    }
  }
  return (maxError, mismatches)
}

func cases() -> [Conv2DCase] {
  [
    Conv2DCase(
      inputHeight: 5,
      inputWidth: 5,
      inputChannels: 1,
      outputChannels: 1,
      kernelHeight: 3,
      kernelWidth: 3,
      strideY: 1,
      strideX: 1,
      dilationY: 1,
      dilationX: 1,
      paddingTop: 1,
      paddingBottom: 1,
      paddingLeft: 1,
      paddingRight: 1,
      tileHeight: 8,
      tileWidth: 8
    ),
    Conv2DCase(
      inputHeight: 5,
      inputWidth: 5,
      inputChannels: 4,
      outputChannels: 8,
      kernelHeight: 3,
      kernelWidth: 3,
      strideY: 1,
      strideX: 1,
      dilationY: 1,
      dilationX: 1,
      paddingTop: 1,
      paddingBottom: 1,
      paddingLeft: 1,
      paddingRight: 1,
      tileHeight: 8,
      tileWidth: 8
    ),
    Conv2DCase(
      inputHeight: 17,
      inputWidth: 19,
      inputChannels: 8,
      outputChannels: 12,
      kernelHeight: 3,
      kernelWidth: 3,
      strideY: 1,
      strideX: 1,
      dilationY: 1,
      dilationX: 1,
      paddingTop: 1,
      paddingBottom: 1,
      paddingLeft: 1,
      paddingRight: 1,
      tileHeight: 8,
      tileWidth: 8
    ),
    Conv2DCase(
      inputHeight: 33,
      inputWidth: 35,
      inputChannels: 4,
      outputChannels: 8,
      kernelHeight: 3,
      kernelWidth: 3,
      strideY: 1,
      strideX: 1,
      dilationY: 1,
      dilationX: 1,
      paddingTop: 1,
      paddingBottom: 1,
      paddingLeft: 1,
      paddingRight: 1,
      tileHeight: 8,
      tileWidth: 8
    ),
  ]
}

@main
struct convolution2d_padding {
  static func main() {
    let device = makeDevice()
    print("Device: \(device.name)")
    print("Readable Conv2D padding probe")
    print("Padding is expressed with set_offsets(...).")
    print("Cases use full 1,1,1,1 same padding and 8x8 tiles.")
    print("Every tile uses the same padded-input-space coordinate rule and cOutput.store(output).")
    print("")

    var allPassed = true
    for (index, dims) in cases().enumerated() {
      let activation = activationData(dims: dims)
      let weights = weightData(dims: dims)
      let expected = referenceConvolution(activation: activation, weights: weights, dims: dims)
      let actual = runCase(device: device, dims: dims)
      let result = validate(actual: actual, expected: expected, dims: dims)
      if result.mismatches > 0 {
        allPassed = false
      }

      print("Case \(index + 1)/\(cases().count): \(dims)")
      print("Max absolute error: \(result.maxError)")
      print("Mismatches above tolerance: \(result.mismatches)")
      print("")
    }

    print("All cases passed: \(allPassed)")
  }
}
