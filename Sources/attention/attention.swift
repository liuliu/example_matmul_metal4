import Metal
import Foundation

struct BuildOptions {
  var executionSIMDGroups: Int
}

struct BlockDimenions {
  var R: Int
  var C: Int
  var K: Int
}

struct AttentionDimensions {
  var R: Int
  var C: Int
  var K: Int
}

func createSource(blockDimensions: BlockDimenions, attentionDimensions: AttentionDimensions, buildOptions: BuildOptions) -> String {
  return """

#include <metal_stdlib>
#include <metal_tensor>
// Doesn't seem this header is provided at runtime, hence JIT shader won't work unless we package all the headers too.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];


kernel void attention(device half *Q_buf [[buffer(0)]],
                      device half *K_buf [[buffer(1)]],
                      device half *V_buf [[buffer(2)]],
                      device float *O_buf [[buffer(3)]],
                      threadgroup uchar *threadgroup_block [[threadgroup(0)]],
                      uint2 tgid [[threadgroup_position_in_grid]])
{
  auto Q = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(\(attentionDimensions.K), \(attentionDimensions.R)));
  auto K = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(\(attentionDimensions.K), \(attentionDimensions.C)));
  auto V = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(\(attentionDimensions.K), \(attentionDimensions.C)));
  auto O = tensor<device float,  dextents<int32_t, 2>, tensor_inline>(O_buf, dextents<int32_t, 2>(\(attentionDimensions.K), \(attentionDimensions.R)));
  threadgroup half *P_buf = (threadgroup half*)threadgroup_block;
  auto P = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(P_buf, dextents<int32_t, 2>(\(blockDimensions.C), \(blockDimensions.R)));
  constexpr auto qk_desc = matmul2d_descriptor(\(blockDimensions.R), \(blockDimensions.C), \(blockDimensions.K), false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;

  auto mQ = Q.slice<\(blockDimensions.K), \(blockDimensions.R)>(0, tgid.x * \(blockDimensions.R));
  auto mK = K.slice<\(blockDimensions.K), \(blockDimensions.C)>(0, 0);
  auto cS = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cM = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cL = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto correction = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
    if (cM.is_valid_element(k)) {
      cM[k] = numeric_limits<float>::lowest();
      cL[k] = numeric_limits<float>::denorm_min();
    }
  }
  auto mP = P.slice<\(blockDimensions.C), \(blockDimensions.R)>(0, 0);
  auto mV = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(0, 0);
  constexpr auto pv_desc = matmul2d_descriptor(\(blockDimensions.R), \(blockDimensions.K), \(blockDimensions.C), false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
  auto cO_0 = matmul_pv_op.get_destination_cooperative_tensor<decltype(mP), decltype(mV), float>();
  auto cO_1 = matmul_pv_op.get_destination_cooperative_tensor<decltype(mP), decltype(mV), float>();
  for (uint c = 0; c < \(attentionDimensions.C); c += \(blockDimensions.C)) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if (cS.is_valid_element(k)) {
        cS[k] = 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < \(attentionDimensions.K); k += \(blockDimensions.K)) {
      auto mQ = Q.slice<\(blockDimensions.K), \(blockDimensions.R)>(k, tgid.x * \(blockDimensions.R));
      auto mK = K.slice<\(blockDimensions.K), \(blockDimensions.C)>(k, c);
      matmul_qk_op.run(mQ, mK, cS);
    }
    // Online reduce maximum.
    auto cM_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS, cM_new, reduction_operation::max, numeric_limits<float>::lowest());
    // Online correct O
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = cM_new[k] * (1.442695041 * 0.08838834764);
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (auto it = cS.begin(); it != cS.end(); ++it) {
      auto dst_it = cM.map_iterator(it);
      *it = fast::exp2(*it * (1.442695041 * 0.08838834764) - *dst_it);
    }
    // Online reduce sum.
    auto cL_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS, cL_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if(cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_new[k];
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS.get_capacity(); ++k) {
      if(cS.is_valid_element(k)) {
        auto idx = cS.get_multidimensional_index(k);
        P_buf[idx[0] + idx[1] * \(blockDimensions.C)] = (half)cS[k];
      }
    }
    if (c == 0) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
          cO_0[k] = 0;
          cO_1[k] = 0;
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = correction.map_iterator(it);
          cO_0[k] *= *dst_it;
          cO_1[k] *= *dst_it;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto mP = P.slice<\(blockDimensions.C), \(blockDimensions.R)>(0, 0);
    auto mV_0 = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(0, c);
    auto mV_1 = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(\(blockDimensions.K), c);
    matmul_pv_op.run(mP, mV_0, cO_0);
    matmul_pv_op.run(mP, mV_1, cO_1);
  }
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
    if (cO_0.is_valid_element(k)) {
      auto it = cO_0.get_iterator(k);
      auto dst_it = cL.map_iterator(it);
      auto L_reciprocal = fast::divide(1, *dst_it);
      cO_0[k] *= L_reciprocal;
      cO_1[k] *= L_reciprocal;
    }
  }
  auto mO_0 = O.slice<\(blockDimensions.K), \(blockDimensions.R)>(0, tgid.x * \(blockDimensions.R));
  cO_0.store(mO_0);
  auto mO_1 = O.slice<\(blockDimensions.K), \(blockDimensions.R)>(\(blockDimensions.K), tgid.x * \(blockDimensions.R));
  cO_1.store(mO_1);
}
"""
}

@main
struct attention {
  static func main() {
    run(sequenceDimension: 8192, headDimension: 128, buildOptions: BuildOptions(executionSIMDGroups: 4), duplicatedCount: 20)
  }

  static func run(sequenceDimension: Int, headDimension: Int, buildOptions: BuildOptions, duplicatedCount: Int) {
    // 1. Create a reference to the GPU
    guard let device = MTLCreateSystemDefaultDevice() else {
      fatalError("Metal is not supported on this device")
    }

    // This API is only supported on Apple7 and later
    guard device.supportsFamily(.metal4) else {
      fatalError("This device does not support the tensor operations used in the shader.")
    }

    // 2. Create a command queue
    guard let commandQueue = device.makeCommandQueue() else {
      fatalError("Could not create command queue")
    }

    let blockDimensions = BlockDimenions(R: 16, C: 32, K: 64)
    let library: MTLLibrary
    do {
      let source = createSource(blockDimensions: blockDimensions, attentionDimensions: AttentionDimensions(R: sequenceDimension, C: sequenceDimension, K: 128), buildOptions: buildOptions)
      library = try device.makeLibrary(source: source, options: nil)
    } catch {
      fatalError("Could not create library: \(error).")
    }

    let constants = MTLFunctionConstantValues()
    // 4. Create a function object
    guard let attentionFunction = try? library.makeFunction(name: "attention", constantValues: constants) else {
      fatalError("Could not create function")
    }

    var networkDesc = NetworkDescriptor()
    networkDesc.rowDimension = sequenceDimension
    networkDesc.columnDimension = sequenceDimension
    networkDesc.headDimension = headDimension
    let network = Network(descriptor: networkDesc)

    // 5. Create a compute pipeline state
    let pipelineState: MTLComputePipelineState
    do {
      pipelineState = try device.makeComputePipelineState(function: attentionFunction)
    } catch {
      fatalError("Could not create pipeline state: \(error)")
    }

    let sizeQ = sequenceDimension * headDimension * MemoryLayout<Float16>.size
    let sizeK = sequenceDimension * headDimension * MemoryLayout<Float16>.size
    let sizeV = sequenceDimension * headDimension * MemoryLayout<Float16>.size

    let bufferQ = device.makeBuffer(length: sizeQ, options: .storageModeShared)
    let bufferK = device.makeBuffer(length: sizeK, options: .storageModeShared)
    let bufferV = device.makeBuffer(length: sizeV, options: .storageModeShared)
    let bufferO = device.makeBuffer(length: sizeQ * 2, options: .storageModeShared)

    let Q: [Float16] = network.Q.map { Float16($0) }
    let K: [Float16] = network.K.map { Float16($0) }
    let V: [Float16] = network.V.map { Float16($0) }
    bufferQ?.contents().copyMemory(from: Q, byteCount: sizeQ)
    bufferK?.contents().copyMemory(from: K, byteCount: sizeK)
    bufferV?.contents().copyMemory(from: V, byteCount: sizeV)

    // 7. Create a command buffer and encoder
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create command buffer or encoder")
    }

    computeCommandEncoder.setComputePipelineState(pipelineState)
    computeCommandEncoder.setBuffer(bufferQ, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(bufferK, offset: 0, index: 1)
    computeCommandEncoder.setBuffer(bufferV, offset: 0, index: 2)
    computeCommandEncoder.setBuffer(bufferO, offset: 0, index: 3)
    computeCommandEncoder.useResource(bufferQ!, usage: .read)
    computeCommandEncoder.useResource(bufferK!, usage: .read)
    computeCommandEncoder.useResource(bufferV!, usage: .read)
    computeCommandEncoder.useResource(bufferO!, usage: .write)
    computeCommandEncoder.setThreadgroupMemoryLength(blockDimensions.R * blockDimensions.C * MemoryLayout<Float16>.size, index: 0)

      // 8. Dispatch threads
    let threadgroups = MTLSize(width: sequenceDimension / blockDimensions.R, height: 1, depth: 1)
    let simdgroupWidth = pipelineState.threadExecutionWidth
    let threadsPerThreadgroup = MTLSize(width: simdgroupWidth, height: 1, depth: 1)

    for _ in 0..<duplicatedCount {
      computeCommandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
    }
    computeCommandEncoder.endEncoding()

    // 9. Commit the command buffer and wait for completion
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
  
    // Determine the time taken.
    let start = commandBuffer.gpuStartTime
    let end = commandBuffer.gpuEndTime
    let latency = end - start
    
    // Determine the amount of work done.
    var operations = (2 * headDimension + 5) * sequenceDimension * sequenceDimension
    operations = operations * duplicatedCount
    let gflops = Int(Double(operations) / Double(latency) / 1e9)
    print("GFlops: \(gflops)")
    // 10. Read the results
    var resultO = [Float](repeating: 0, count: sequenceDimension * headDimension)
    let resultBufferPointer = bufferO?.contents().bindMemory(to: Float.self, capacity: sequenceDimension * headDimension)

    if let ptr = resultBufferPointer {
      resultO = Array(UnsafeBufferPointer(start: ptr, count: sequenceDimension * headDimension))
    }
    let expectedO = network.inferenceAttention()

    // Optional: Verify the result on the CPU
    for i in 0..<(sequenceDimension * headDimension) {
      if abs(expectedO[i] - Float(resultO[i])) > 5e-3 {
        print("CPU calculated O[\(i)]: \(expectedO[i])")
        print("GPU calculated O[\(i)]: \(resultO[i])")
      }
    }
  }
}
