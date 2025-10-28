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
  var Hq: Int
  var Hk: Int
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
                      device half *O_buf [[buffer(3)]],
                      threadgroup uchar *threadgroup_block [[threadgroup(0)]],
                      ushort sgid [[simdgroup_index_in_threadgroup]],
                      uint2 tgid [[threadgroup_position_in_grid]])
{
  tgid = { tgid.x / \(attentionDimensions.Hq), tgid.x % \(attentionDimensions.Hq) };
  tgid.x = tgid.x * \(buildOptions.executionSIMDGroups) + sgid;
  if (tgid.x * \(blockDimensions.R) >= \(attentionDimensions.R)) {
    return;
  }
  auto Q = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(\(attentionDimensions.K * attentionDimensions.Hq), \(attentionDimensions.R)));
  auto K = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(\(attentionDimensions.K * attentionDimensions.Hk), \(attentionDimensions.C)));
  auto V = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(\(attentionDimensions.K * attentionDimensions.Hk), \(attentionDimensions.C)));
  threadgroup half *P_buf = (threadgroup half*)threadgroup_block + \(blockDimensions.C) * \(blockDimensions.R) * sgid;
  auto P = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(P_buf, extents<int32_t, \(blockDimensions.C), \(blockDimensions.R)>());
  constexpr auto qk_desc = matmul2d_descriptor(\(blockDimensions.R), \(blockDimensions.C), \(blockDimensions.K), false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;
  auto mQ = Q.slice<\(blockDimensions.K), \(blockDimensions.R)>(tgid.y * \(attentionDimensions.K), tgid.x * \(blockDimensions.R));
  auto mK = K.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K), 0);
  auto cS_0 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
  auto cS_1 = matmul_qk_op.get_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
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
  auto mV = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(0, 0);
  constexpr auto pv_desc = matmul2d_descriptor(\(blockDimensions.R), \(blockDimensions.K), \(blockDimensions.C), false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<pv_desc, execution_simdgroups<1>> matmul_pv_op;
  auto cO_0 = matmul_pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
  auto cO_1 = matmul_pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();
  for (uint c = 0; c < \(attentionDimensions.C); c += \(blockDimensions.C) * 2) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        cS_0[k] = 0;
        cS_1[k] = 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < \(attentionDimensions.K); k += \(blockDimensions.K)) {
      auto mQ = Q.slice<\(blockDimensions.K), \(blockDimensions.R)>(tgid.y * \(attentionDimensions.K) + k, tgid.x * \(blockDimensions.R));
      auto mK_0 = K.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + k, c);
      auto mK_1 = K.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + k, c + \(blockDimensions.C));
      matmul_qk_op.run(mQ, mK_0, cS_0);
      matmul_qk_op.run(mQ, mK_1, cS_1);
    }
    // Online reduce maximum.
    auto cM_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cM_0_new, reduction_operation::max, numeric_limits<float>::lowest());
    auto cM_1_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_1, cM_1_new, reduction_operation::max, numeric_limits<float>::lowest());
    // Online correct O
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cM.get_capacity(); ++k) {
      if (cM.is_valid_element(k)) {
        correction[k] = 1;
        const float M_new = max(cM_0_new[k], cM_1_new[k]) * (1.442695041 * 0.08838834764);
        if (M_new > cM[k]) {
          correction[k] = fast::exp2(cM[k] - M_new);
          cM[k] = M_new;
        }
      }
    }
    // Softmax. cS becomes cP.
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto it = cS_0.get_iterator(k);
        auto dst_it = cM.map_iterator(it);
        cS_0[k] = fast::exp2(cS_0[k] * (1.442695041 * 0.08838834764) - *dst_it);
        cS_1[k] = fast::exp2(cS_1[k] * (1.442695041 * 0.08838834764) - *dst_it);
      }
    }
    // Online reduce sum.
    auto cL_0_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_0, cL_0_new, reduction_operation::sum, (float)0);
    auto cL_1_new = matmul_qk_op.get_row_reduction_destination_cooperative_tensor<decltype(mQ), decltype(mK), float>();
    reduce_rows(cS_1, cL_1_new, reduction_operation::sum, (float)0);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cL.get_capacity(); ++k) {
      if(cL.is_valid_element(k)) {
        cL[k] = cL[k] * correction[k] + cL_0_new[k] + cL_1_new[k];
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
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if(cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        P_buf[idx[0] + idx[1] * \(blockDimensions.C)] = (half)cS_0[k];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto mV_0_0 = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K), c);
    auto mV_0_1 = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + \(blockDimensions.K), c);
    matmul_pv_op.run(P, mV_0_0, cO_0);
    matmul_pv_op.run(P, mV_0_1, cO_1);
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
      if(cS_1.is_valid_element(k)) {
        auto idx = cS_1.get_multidimensional_index(k);
        P_buf[idx[0] + idx[1] * \(blockDimensions.C)] = (half)cS_1[k];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto mV_1_0 = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K), c + \(blockDimensions.C));
    auto mV_1_1 = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + \(blockDimensions.K), c + \(blockDimensions.C));
    matmul_pv_op.run(P, mV_1_0, cO_0);
    matmul_pv_op.run(P, mV_1_1, cO_1);
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
  auto O = O_buf + tgid.x * \(blockDimensions.R) * \(attentionDimensions.K) + tgid.y * \(attentionDimensions.K);
  #pragma clang loop unroll(full)
  for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
    if (cO_0.is_valid_element(k)) {
      auto idx = cO_0.get_multidimensional_index(k);
      O[idx[0] + idx[1] * \(attentionDimensions.K)] = (half)cO_0[k];
      O[idx[0] + idx[1] * \(attentionDimensions.K) + \(blockDimensions.K)] = (half)cO_1[k];
    }
  }
}
"""
}

@main
struct attention {
  static func main() {
    run(sequenceDimension: 1024, headDimension: 128, Hq: 1, Hk: 1, buildOptions: BuildOptions(executionSIMDGroups: 16), duplicatedCount: 20)
  }

  static func run(sequenceDimension: Int, headDimension: Int, Hq: Int, Hk: Int, buildOptions: BuildOptions, duplicatedCount: Int) {
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

    let blockDimensions = BlockDimenions(R: 16, C: 64, K: 64)
    let library: MTLLibrary
    do {
      let source = createSource(blockDimensions: blockDimensions, attentionDimensions: AttentionDimensions(R: sequenceDimension, C: sequenceDimension, K: 128, Hq: Hq, Hk: Hk), buildOptions: buildOptions)
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
    networkDesc.rowDimension = sequenceDimension * Hq
    networkDesc.columnDimension = sequenceDimension * Hk
    networkDesc.headDimension = headDimension
    let network = Network(descriptor: networkDesc)

    // 5. Create a compute pipeline state
    let pipelineState: MTLComputePipelineState
    do {
      pipelineState = try device.makeComputePipelineState(function: attentionFunction)
    } catch {
      fatalError("Could not create pipeline state: \(error)")
    }

    let sizeQ = sequenceDimension * headDimension * Hq * MemoryLayout<Float16>.size
    let sizeK = sequenceDimension * headDimension * Hk * MemoryLayout<Float16>.size
    let sizeV = sequenceDimension * headDimension * Hk * MemoryLayout<Float16>.size

    let bufferQ = device.makeBuffer(length: sizeQ, options: .storageModeShared)
    let bufferK = device.makeBuffer(length: sizeK, options: .storageModeShared)
    let bufferV = device.makeBuffer(length: sizeV, options: .storageModeShared)
    let bufferO = device.makeBuffer(length: sizeQ, options: .storageModeShared)

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
    computeCommandEncoder.setThreadgroupMemoryLength(blockDimensions.R * blockDimensions.C * buildOptions.executionSIMDGroups * MemoryLayout<Float16>.size, index: 0)

      // 8. Dispatch threads
    let threadgroups = MTLSize(width: sequenceDimension / (blockDimensions.R * buildOptions.executionSIMDGroups), height: 1, depth: 1)
    let simdgroupWidth = pipelineState.threadExecutionWidth
    let threadsPerThreadgroup = MTLSize(width: simdgroupWidth * buildOptions.executionSIMDGroups, height: 1, depth: 1)

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
    var resultO = [Float16](repeating: 0, count: sequenceDimension * headDimension)
    let resultBufferPointer = bufferO?.contents().bindMemory(to: Float16.self, capacity: sequenceDimension * headDimension)

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
