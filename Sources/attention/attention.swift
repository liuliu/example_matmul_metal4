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
  let dotScale = 1.442695041 * 1 / Double(attentionDimensions.K).squareRoot()
  let kBlocks = (attentionDimensions.K + blockDimensions.K - 1) / blockDimensions.K
  let allocateO = ((0..<kBlocks).map {
    "  auto cO_\($0) = matmul_pv_op.get_destination_cooperative_tensor<decltype(P), decltype(mV), float>();"
  }).joined(separator: "\n")
  let initializeO = ((0..<kBlocks).map {
    "          cO_\($0)[k] = 0;"
  }).joined(separator: "\n")
  let correctO = ((0..<kBlocks).map {
    "          cO_\($0)[k] *= *dst_it;"
  }).joined(separator: "\n")
  let accumulateO0 = ((0..<kBlocks).map {
"""
    auto mV_0_\($0) = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + \($0 * blockDimensions.K), c);
    matmul_pv_op.run(P, mV_0_\($0), cO_\($0));
"""
  }).joined(separator: "\n")
  let accumulateO1 = ((0..<kBlocks).map {
"""
    auto mV_1_\($0) = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + \($0 * blockDimensions.K), c + \(blockDimensions.C));
    matmul_pv_op.run(P, mV_1_\($0), cO_\($0));
"""
  }).joined(separator: "\n")
  let accumulateO0Last = ((0..<kBlocks).map {
"""
    auto mV_0_\($0) = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + \($0 * blockDimensions.K), C - C_remainder);
    matmul_pv_op.run(P, mV_0_\($0), cO_\($0));
"""
  }).joined(separator: "\n")
  let accumulateO1Last = ((0..<kBlocks).map {
"""
    auto mV_1_\($0) = V.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + \($0 * blockDimensions.K), C + \(blockDimensions.C) - C_remainder);
    matmul_pv_op.run(P, mV_1_\($0), cO_\($0));
"""
  }).joined(separator: "\n")
  let writeO = ((0..<kBlocks).map {
    if ($0 < kBlocks - 1) || (attentionDimensions.K % blockDimensions.K == 0) {
      return """
        O[idx[0] + \($0 * blockDimensions.K) + idx[1] * K_Hq] = (half)(cO_\($0)[k] * L_reciprocal);
"""
    } else {
      return """
        if (idx[0] + \($0 * blockDimensions.K) < \(attentionDimensions.K)) {
          O[idx[0] + \($0 * blockDimensions.K) + idx[1] * K_Hq] = (half)(cO_\($0)[k] * L_reciprocal);
        }
"""
    }
  }).joined(separator: "\n")
  return """

#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant uint R [[function_constant(0)]];
constant uint C [[function_constant(1)]];
constant uint Hq [[function_constant(2)]];
constant uint Hk [[function_constant(3)]];

constant uint C_edge = C - \(blockDimensions.C * 2) + 1;
constant uint C_remainder = C % \(blockDimensions.C * 2);
constant uint R_edge = R - \(blockDimensions.R) + 1;
constant uint R_remainder = R % \(blockDimensions.R);
constant uint K_edge = \(attentionDimensions.K) - \(blockDimensions.K) + 1;
constant uint K_Hq = \(attentionDimensions.K) * Hq;
constant uint K_Hk = \(attentionDimensions.K) * Hk;

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
  auto Q = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(Q_buf, dextents<int32_t, 2>(K_Hq, R));
  auto K = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(K_buf, dextents<int32_t, 2>(K_Hk, C));
  auto V = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(V_buf, dextents<int32_t, 2>(K_Hk, C));
  threadgroup half *P_buf = (threadgroup half*)threadgroup_block + \(blockDimensions.C) * \(blockDimensions.R) * sgid;
  auto P = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(P_buf, extents<int32_t, \(blockDimensions.C), \(blockDimensions.R)>());
  constexpr auto qk_desc = matmul2d_descriptor(\(blockDimensions.R), \(blockDimensions.C), \(blockDimensions.K), false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc, execution_simdgroups<1>> matmul_qk_op;
  constexpr auto qk_desc_last = matmul2d_descriptor(\(blockDimensions.R), \(blockDimensions.C), \(attentionDimensions.K % blockDimensions.K), false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_desc_last, execution_simdgroups<1>> matmul_qk_op_last;
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
\(allocateO)
  for (uint c = 0; c < C_edge; c += \(blockDimensions.C * 2)) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        cS_0[k] = 0;
        cS_1[k] = 0;
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += \(blockDimensions.K)) {
      auto mQ = Q.slice<\(blockDimensions.K), \(blockDimensions.R)>(tgid.y * \(attentionDimensions.K) + k, tgid.x * \(blockDimensions.R));
      auto mK_0 = K.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + k, c);
      auto mK_1 = K.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + k, c + \(blockDimensions.C));
      matmul_qk_op.run(mQ, mK_0, cS_0);
      matmul_qk_op.run(mQ, mK_1, cS_1);
    }
    if (\(attentionDimensions.K % blockDimensions.K > 0 ? "true" : "false")) {
      auto mQ = Q.slice<\(attentionDimensions.K % blockDimensions.K), \(blockDimensions.R)>(tgid.y * \(attentionDimensions.K) + \(attentionDimensions.K - (attentionDimensions.K % blockDimensions.K)), tgid.x * \(blockDimensions.R));
      auto mK_0 = K.slice<\(attentionDimensions.K % blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + \(attentionDimensions.K - (attentionDimensions.K % blockDimensions.K)), c);
      auto mK_1 = K.slice<\(attentionDimensions.K % blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + \(attentionDimensions.K - (attentionDimensions.K % blockDimensions.K)), c + \(blockDimensions.C));
      matmul_qk_op_last.run(mQ, mK_0, cS_0);
      matmul_qk_op_last.run(mQ, mK_1, cS_1);
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
        const float M_new = max(cM_0_new[k], cM_1_new[k]) * \(dotScale);
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
        cS_0[k] = fast::exp2(cS_0[k] * \(dotScale) - *dst_it);
        cS_1[k] = fast::exp2(cS_1[k] * \(dotScale) - *dst_it);
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
\(initializeO)
        }
      }
    } else {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
        if (cO_0.is_valid_element(k)) {
          auto it = cO_0.get_iterator(k);
          auto dst_it = correction.map_iterator(it);
\(correctO)
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
\(accumulateO0)
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
      if(cS_1.is_valid_element(k)) {
        auto idx = cS_1.get_multidimensional_index(k);
        P_buf[idx[0] + idx[1] * \(blockDimensions.C)] = (half)cS_1[k];
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
\(accumulateO1)
  }
  if (C_remainder > 0) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cS_0.get_capacity(); ++k) {
      if (cS_0.is_valid_element(k)) {
        auto idx = cS_0.get_multidimensional_index(k);
        if (idx[0] >= (int)C_remainder) {
          cS_0[k] = numeric_limits<float>::lowest();
        } else {
          cS_0[k] = 0;
        }
        if (idx[0] + \(blockDimensions.C) >= (int)C_remainder) {
          cS_1[k] = numeric_limits<float>::lowest();
        } else {
          cS_1[k] = 0;
        }
      }
    }
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < K_edge; k += \(blockDimensions.K)) {
      auto mQ = Q.slice<\(blockDimensions.K), \(blockDimensions.R)>(tgid.y * \(attentionDimensions.K) + k, tgid.x * \(blockDimensions.R));
      auto mK_0 = K.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + k, C - C_remainder);
      matmul_qk_op.run(mQ, mK_0, cS_0);
      if (C_remainder > \(blockDimensions.C)) {
        auto mK_1 = K.slice<\(blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + k, C + \(blockDimensions.C) - C_remainder);
        matmul_qk_op.run(mQ, mK_1, cS_1);
      }
    }
    if (\(attentionDimensions.K % blockDimensions.K > 0 ? "true" : "false")) {
      auto mQ = Q.slice<\(attentionDimensions.K % blockDimensions.K), \(blockDimensions.R)>(tgid.y * \(attentionDimensions.K) + \(attentionDimensions.K - (attentionDimensions.K % blockDimensions.K)), tgid.x * \(blockDimensions.R));
      auto mK_0 = K.slice<\(attentionDimensions.K % blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + \(attentionDimensions.K - (attentionDimensions.K % blockDimensions.K)), C - C_remainder);
      matmul_qk_op_last.run(mQ, mK_0, cS_0);
      if (C_remainder > \(blockDimensions.C)) {
        auto mK_1 = K.slice<\(attentionDimensions.K % blockDimensions.K), \(blockDimensions.C)>(tgid.y * \(attentionDimensions.K) + \(attentionDimensions.K - (attentionDimensions.K % blockDimensions.K)), C + \(blockDimensions.C) - C_remainder);
        matmul_qk_op_last.run(mQ, mK_1, cS_1);
      }
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
        const float M_new = max(cM_0_new[k], cM_1_new[k]) * \(dotScale);
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
        cS_0[k] = fast::exp2(cS_0[k] * \(dotScale) - *dst_it);
        cS_1[k] = fast::exp2(cS_1[k] * \(dotScale) - *dst_it);
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
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = correction.map_iterator(it);
\(correctO)
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
\(accumulateO0Last)
    if (C_remainder > \(blockDimensions.C)) {
      #pragma clang loop unroll(full)
      for (unsigned short k = 0; k < cS_1.get_capacity(); ++k) {
        if(cS_1.is_valid_element(k)) {
          auto idx = cS_1.get_multidimensional_index(k);
          P_buf[idx[0] + idx[1] * \(blockDimensions.C)] = (half)cS_1[k];
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
\(accumulateO1Last)
    }
  }
  auto O = O_buf + tgid.x * (\(blockDimensions.R) * K_Hq) + tgid.y * \(attentionDimensions.K);
  if (R_remainder > 0 && tgid.x * \(blockDimensions.R) >= R_edge) {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = cL.map_iterator(it);
        auto L_reciprocal = fast::divide(1, *dst_it);
        auto idx = cO_0.get_multidimensional_index(k);
        if (idx[1] < (int)R_remainder) {
\(writeO)
        }
      }
    }
  } else {
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cO_0.get_capacity(); ++k) {
      if (cO_0.is_valid_element(k)) {
        auto it = cO_0.get_iterator(k);
        auto dst_it = cL.map_iterator(it);
        auto L_reciprocal = fast::divide(1, *dst_it);
        auto idx = cO_0.get_multidimensional_index(k);
\(writeO)
      }
    }
  }
}
"""
}

@main
struct attention {
  static func main() {
    run(sequenceDimension: 520, headDimension: 120, Hq: 1, Hk: 1, buildOptions: BuildOptions(executionSIMDGroups: 16), duplicatedCount: 20)
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
      let source = createSource(blockDimensions: blockDimensions, attentionDimensions: AttentionDimensions(R: sequenceDimension, C: sequenceDimension, K: headDimension, Hq: Hq, Hk: Hk), buildOptions: buildOptions)
      library = try device.makeLibrary(source: source, options: nil)
    } catch {
      fatalError("Could not create library: \(error).")
    }

    let constants = MTLFunctionConstantValues()
    var constantR: UInt32 = UInt32(sequenceDimension)
    var constantC: UInt32 = UInt32(sequenceDimension)
    var constantHq: UInt32 = UInt32(Hq)
    var constantHk: UInt32 = UInt32(Hk)
    constants.setConstantValue(&constantR, type: .uint, index: 0)
    constants.setConstantValue(&constantC, type: .uint, index: 1)
    constants.setConstantValue(&constantHq, type: .uint, index: 2)
    constants.setConstantValue(&constantHk, type: .uint, index: 3)
    // 4. Create a function object
    guard let attentionFunction = try? library.makeFunction(name: "attention", constantValues: constants) else {
      fatalError("Could not create function")
    }

    var networkDesc = NetworkDescriptor()
    networkDesc.rowDimension = sequenceDimension
    networkDesc.columnDimension = sequenceDimension
    networkDesc.headDimension = headDimension
    let networks = (0..<Hq).map { _ in Network(descriptor: networkDesc) }

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

    var Q = [Float16](repeating: 0, count: sequenceDimension * headDimension * Hq)
    var K = [Float16](repeating: 0, count: sequenceDimension * headDimension * Hk)
    var V = [Float16](repeating: 0, count: sequenceDimension * headDimension * Hk)
    for i in 0..<Hq {
      for j in 0..<sequenceDimension {
        for k in 0..<headDimension {
          Q[j * headDimension * Hq + i * headDimension + k] = Float16(networks[i].Q[j * headDimension + k])
          K[j * headDimension * Hq + i * headDimension + k] = Float16(networks[i].K[j * headDimension + k])
          V[j * headDimension * Hq + i * headDimension + k] = Float16(networks[i].V[j * headDimension + k])
        }
      }
    }
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
    let threadgroups = MTLSize(width: (sequenceDimension + blockDimensions.R * buildOptions.executionSIMDGroups - 1) / (blockDimensions.R * buildOptions.executionSIMDGroups) * Hq, height: 1, depth: 1)
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
    var operations = (2 * headDimension + 5) * sequenceDimension * sequenceDimension * Hq
    operations = operations * duplicatedCount
    let gflops = Int(Double(operations) / Double(latency) / 1e9)
    print("GFlops: \(gflops)")
    // 10. Read the results
    var resultO = [Float16](repeating: 0, count: sequenceDimension * headDimension * Hq)
    let resultBufferPointer = bufferO?.contents().bindMemory(to: Float16.self, capacity: sequenceDimension * headDimension * Hq)

    if let ptr = resultBufferPointer {
      resultO = Array(UnsafeBufferPointer(start: ptr, count: sequenceDimension * headDimension * Hq))
    }
    var expectedO = [Float](repeating: 0, count: sequenceDimension * headDimension * Hq)
    for i in 0..<Hq {
      let O = networks[i].inferenceAttention()
      for j in 0..<sequenceDimension {
        for k in 0..<headDimension {
          expectedO[j * headDimension * Hq + i * headDimension + k] = O[j * headDimension + k]
        }
      }
    }

    // Optional: Verify the result on the CPU
    for i in 0..<(sequenceDimension * headDimension * Hq) {
      if abs(expectedO[i] - Float(resultO[i])) > 5e-3 {
        print("CPU calculated O[\(i)]: \(expectedO[i])")
        print("GPU calculated O[\(i)]: \(resultO[i])")
      }
    }
  }
}
