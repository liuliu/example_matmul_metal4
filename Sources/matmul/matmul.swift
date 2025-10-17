import Metal
import Foundation


struct GEMMDimensions {
  var M: Int
  var N: Int
  var K: Int
}

func createSource(matrixDimensions: GEMMDimensions, blockDimensions: GEMMDimensions, transpose: (left: Bool, right: Bool), bias: Bool, executionSIMDGroups: Int, swapMN: Bool) -> String {
  let biasInputTerm: String
  let initializeC: String
  if bias {
    biasInputTerm = "device half *bias_buf [[buffer(3)]],"
    initializeC = """
        #pragma clang loop unroll(full)
        for (unsigned short k = 0; k < cT.get_capacity(); ++k) {
          if(cT.is_valid_element(k)) {
            auto idx = cT.get_multidimensional_index(k);
            cT[k] = bias_buf[idx[0] + tgid.x * \(blockDimensions.N)];
          }
        }
    """
  } else {
    biasInputTerm = ""
    initializeC = """
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cT.get_capacity(); ++k) {
      if(cT.is_valid_element(k)) {
        cT[k] = 0;
      }
    }
  """
  }
  let aSlice: String
  let aMatrixSize: String
  let aTile0Size: String
  let aTileK1Size: String
  let aTileK2Size: String
  let aTileLastK2Size: String
  let aResidualSlice: String
  let aTileLastKSize: String
  if transpose.left {
    aSlice = "\(blockDimensions.M), \(blockDimensions.K)"
    aMatrixSize = "\(matrixDimensions.M), \(matrixDimensions.K)"
    aTile0Size = "tgid.y * \(blockDimensions.M), 0"
    aTileK1Size = "tgid.y * \(blockDimensions.M), k"
    aTileK2Size = "tgid.y * \(blockDimensions.M), k + \(blockDimensions.K)"
    aTileLastK2Size = "tgid.y * \(blockDimensions.M), \((matrixDimensions.K / blockDimensions.K - 1) * blockDimensions.K)"
    aTileLastKSize = "tgid.y * \(blockDimensions.M), \(matrixDimensions.K / (blockDimensions.K * 2) * (blockDimensions.K * 2))"
    aResidualSlice = "\(blockDimensions.M), \(matrixDimensions.K % (blockDimensions.K * 2))"
  } else {
    aSlice = "\(blockDimensions.K), \(blockDimensions.M)"
    aMatrixSize = "\(matrixDimensions.K), \(matrixDimensions.M)"
    aTile0Size = "0, tgid.y * \(blockDimensions.M)"
    aTileK1Size = "k, tgid.y * \(blockDimensions.M)"
    aTileK2Size = "k + \(blockDimensions.K), tgid.y * \(blockDimensions.M)"
    aTileLastK2Size = "\((matrixDimensions.K / blockDimensions.K - 1) * blockDimensions.K), tgid.y * \(blockDimensions.M)"
    aTileLastKSize = "\(matrixDimensions.K / (blockDimensions.K * 2) * (blockDimensions.K * 2)), tgid.y * \(blockDimensions.M)"
    aResidualSlice = "\(matrixDimensions.K % (blockDimensions.K * 2)), \(blockDimensions.M)"
  }
  let bSlice: String
  let bMatrixSize: String
  let bTile0Size: String
  let bTileK1Size: String
  let bTileK2Size: String
  let bTileLastK2Size: String
  let bResidualSlice: String
  let bTileLastKSize: String
  if transpose.right {
    bSlice = "\(blockDimensions.K), \(blockDimensions.N)"
    bMatrixSize = "\(matrixDimensions.K), \(matrixDimensions.N)"
    bTile0Size = "0, tgid.x * \(blockDimensions.N)"
    bTileK1Size = "k, tgid.x * \(blockDimensions.N)"
    bTileK2Size = "k + \(blockDimensions.K), tgid.x * \(blockDimensions.N)"
    bTileLastK2Size = "\((matrixDimensions.K / blockDimensions.K - 1) * blockDimensions.K), tgid.x * \(blockDimensions.N)"
    bTileLastKSize = "\(matrixDimensions.K / (blockDimensions.K * 2) * (blockDimensions.K * 2)), tgid.x * \(blockDimensions.N)"
    bResidualSlice = "\(matrixDimensions.K % (blockDimensions.K * 2)), \(blockDimensions.N)"
  } else {
    bSlice = "\(blockDimensions.N), \(blockDimensions.K)"
    bMatrixSize = "\(matrixDimensions.N), \(matrixDimensions.K)"
    bTile0Size = "tgid.x * \(blockDimensions.N), 0"
    bTileK1Size = "tgid.x * \(blockDimensions.N), k"
    bTileK2Size = "tgid.x * \(blockDimensions.N), k + \(blockDimensions.K)"
    bTileLastK2Size = "tgid.x * \(blockDimensions.N), \((matrixDimensions.K / blockDimensions.K - 1) * blockDimensions.K)"
    bTileLastKSize = "tgid.x * \(blockDimensions.N), \(matrixDimensions.K / (blockDimensions.K * 2) * (blockDimensions.K * 2))"
    bResidualSlice = "\(blockDimensions.N), \(matrixDimensions.K % (blockDimensions.K * 2))"
  }
  let swapXY: String
  if swapMN {
    swapXY = "tgid.xy = tgid.yx;"
  } else {
    swapXY = ""
  }
  return """

#include <metal_stdlib>
#include <metal_tensor>
// Doesn't seem this header is provided at runtime, hence JIT shader won't work unless we package all the headers too.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul_static_slice_dynamic_extents(device half *A_buf [[buffer(0)]],
                         device half *B_buf [[buffer(1)]],
                         device half *C_buf [[buffer(2)]], \(biasInputTerm)
                         uint2 tgid [[threadgroup_position_in_grid]])
{
  // Construct shader allocated tensors. This is easier since we can just bind buffer directly with Metal 3 APIs.
  auto A = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(\(aMatrixSize)));
  auto B = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(\(bMatrixSize)));
  auto C = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(\(matrixDimensions.N), \(matrixDimensions.M)));
  \(swapXY)

  if (tgid.x * \(blockDimensions.N) + \(blockDimensions.N - 1) < \(matrixDimensions.N) && tgid.y * \(blockDimensions.M) + \(blockDimensions.M - 1) < \(matrixDimensions.M)) {
    // Use static slice.
    // descriptor to create matmul operation that does \(blockDimensions.K)x\(blockDimensions.M) times \(blockDimensions.N)x\(blockDimensions.K) producing \(blockDimensions.N)x\(blockDimensions.M)
    constexpr auto matmulDescriptor = matmul2d_descriptor(\(blockDimensions.M), \(blockDimensions.N), \(blockDimensions.K), \(transpose.left ? "true" : "false"), \(transpose.right ? "true" : "false"), false, matmul2d_descriptor::mode::multiply_accumulate);

    // create matmul op from above descriptor with \(executionSIMDGroups) SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<\(executionSIMDGroups)>> matmul_op;

    auto mA = A.slice<\(aSlice)>(\(aTile0Size));
    auto mB = B.slice<\(bSlice)>(\(bTile0Size));
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
\(initializeC)
    #pragma clang loop unroll(full)
    for (ushort k = 0; k < \(matrixDimensions.K - (blockDimensions.K * 2) + 1); k += \(blockDimensions.K * 2)) {
      // Create appropriate slice for this thread group to work on.
      auto mA0 = A.slice<\(aSlice)>(\(aTileK1Size));
      auto mB0 = B.slice<\(bSlice)>(\(bTileK1Size));
      auto mA1 = A.slice<\(aSlice)>(\(aTileK2Size));
      auto mB1 = B.slice<\(bSlice)>(\(bTileK2Size));
      matmul_op.run(mA0, mB0, cT);
      matmul_op.run(mA1, mB1, cT);
    }
    if (\((matrixDimensions.K % (blockDimensions.K * 2) != 0) && (matrixDimensions.K % (blockDimensions.K) == 0) ? "true" : "false")) {
      auto mA = A.slice<\(aSlice)>(\(aTileLastK2Size));
      auto mB = B.slice<\(bSlice)>(\(bTileLastK2Size));
      matmul_op.run(mA, mB, cT);
    }
    if (\((matrixDimensions.K % blockDimensions.K != 0) ? "true" : "false")) {
      constexpr auto matmulDescriptor = matmul2d_descriptor(\(blockDimensions.M), \(blockDimensions.N), \(matrixDimensions.K % (blockDimensions.K * 2)), \(transpose.left ? "true" : "false"), \(transpose.right ? "true" : "false"), false, matmul2d_descriptor::mode::multiply_accumulate);
      // create matmul op from above descriptor with \(executionSIMDGroups) SIMD-Groups.
      matmul2d<matmulDescriptor, execution_simdgroups<\(executionSIMDGroups)>> matmul_op;
      auto mA = A.slice<\(aResidualSlice)>(\(aTileLastKSize));
      auto mB = B.slice<\(bResidualSlice)>(\(bTileLastKSize));
      matmul_op.run(mA, mB, cT);
    }
    auto mC = C.slice<\(blockDimensions.N), \(blockDimensions.M)>(tgid.x * \(blockDimensions.N), tgid.y * \(blockDimensions.M));
    cT.store(mC);
  } else {
    // Use dynamic slice for this edge case.
    // descriptor to create matmul operation that does \(blockDimensions.K)x\(blockDimensions.M) times \(blockDimensions.N)x\(blockDimensions.K) producing \(blockDimensions.N)x\(blockDimensions.M)
    constexpr auto matmulDescriptor = matmul2d_descriptor(\(blockDimensions.M), \(blockDimensions.N), dynamic_length_v<int>, \(transpose.left ? "true" : "false"), \(transpose.right ? "true" : "false"), false, matmul2d_descriptor::mode::multiply);

    // create matmul op from above descriptor with \(executionSIMDGroups) SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<\(executionSIMDGroups)>> matmul_op;

    auto mA = A.slice(\(aTile0Size));
    auto mB = B.slice(\(bTile0Size));
    auto cT = matmul_op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
\(initializeC)
    matmul_op.run(mA, mB, cT);
    auto mC = C.slice(tgid.x * \(blockDimensions.N), tgid.y * \(blockDimensions.M));
    cT.store(mC);
  }
}

"""
}

@main
struct matmul {
  static func main() {
    run(M: 3072 * 4, N: 3072, K: 3072 + 64, blockDimensions: GEMMDimensions(M: 128, N: 64, K: 64))
  }

  static func run(M: Int, N: Int, K: Int, blockDimensions: GEMMDimensions) {

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

    let library: MTLLibrary
    do {
      let source = createSource(matrixDimensions: GEMMDimensions(M: M, N: N, K: K), blockDimensions: blockDimensions, transpose: (left: false, right: true), bias: true, executionSIMDGroups: 4, swapMN: M > N)
      print(source)
      library = try device.makeLibrary(source: source, options: nil)
    } catch {
      fatalError("Could not create library: \(error).")
    }


    // 4. Create a function object
    guard let matmulFunction = library.makeFunction(name: "matmul_static_slice_dynamic_extents") else {
      fatalError("Could not create function")
    }

    // 5. Create a compute pipeline state
    let pipelineState: MTLComputePipelineState
    do {
      pipelineState = try device.makeComputePipelineState(function: matmulFunction)
    } catch {
      fatalError("Could not create pipeline state: \(error)")
    }

    let sizeA = M * K * MemoryLayout<Float16>.size
    let sizeB = K * N * MemoryLayout<Float16>.size
    let sizeC = M * N * MemoryLayout<Float16>.size
    let sizeBias = N * MemoryLayout<Float16>.size

    let bufferA = device.makeBuffer(length: sizeA, options: .storageModeShared)
    let bufferB = device.makeBuffer(length: sizeB, options: .storageModeShared)
    let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared)
    let bufferBias = device.makeBuffer(length: sizeBias, options: .storageModeShared)

    // Initialize matrices A and B with some values
    var matrixA = [Float16](repeating: 0, count: M * K)
    var matrixB = [Float16](repeating: 0, count: K * N)
    var bias = [Float16](repeating: 0, count: N)
    let normalizationFactor = 1 / Float(K).squareRoot()
    for i in 0..<(M * K) {
      matrixA[i] = Float16.random(in: 0...1) * Float16(normalizationFactor)
    }
    for i in 0..<(K * N) {
      matrixB[i] = Float16.random(in: 0...1) * Float16(normalizationFactor)
    }
    for i in 0..<N {
      bias[i] = Float16.random(in: 0...1) * Float16(normalizationFactor)
    }

    bufferA?.contents().copyMemory(from: matrixA, byteCount: sizeA)
    bufferB?.contents().copyMemory(from: matrixB, byteCount: sizeB)
    bufferBias?.contents().copyMemory(from: bias, byteCount: sizeBias)
    bufferC?.contents().initializeMemory(as: UInt8.self, repeating: 0, count: sizeC)

    // 7. Create a command buffer and encoder
    guard let commandBuffer = commandQueue.makeCommandBuffer(),
          let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
      fatalError("Could not create command buffer or encoder")
    }

    computeCommandEncoder.setComputePipelineState(pipelineState)
    computeCommandEncoder.setBuffer(bufferA, offset: 0, index: 0)
    computeCommandEncoder.setBuffer(bufferB, offset: 0, index: 1)
    computeCommandEncoder.setBuffer(bufferC, offset: 0, index: 2)
    computeCommandEncoder.setBuffer(bufferBias, offset: 0, index: 3)
    computeCommandEncoder.useResource(bufferA!, usage: .read)
    computeCommandEncoder.useResource(bufferB!, usage: .read)
    computeCommandEncoder.useResource(bufferBias!, usage: .read)
    computeCommandEncoder.useResource(bufferC!, usage: .write)

      // 8. Dispatch threads
    let threadgroups: MTLSize
    if M > N {
      threadgroups = MTLSize(width: (M + blockDimensions.M - 1) / blockDimensions.M, height: (N + blockDimensions.N - 1) / blockDimensions.N, depth: 1)
    } else {
      threadgroups = MTLSize(width: (N + blockDimensions.N - 1) / blockDimensions.N, height: (M + blockDimensions.M - 1) / blockDimensions.M, depth: 1)
    }
    let simdgroupWidth = pipelineState.threadExecutionWidth
    let threadsPerThreadgroup = MTLSize(width: simdgroupWidth * 4, height: 1, depth: 1)

    for _ in 0..<20 {
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
    var operations = 2 * M * N * K
    operations = operations * 20
    let gflops = Int(Double(operations) / Double(latency) / 1e9)
    print("GFlops: \(gflops)")
    // 10. Read the results
    var resultMatrix = [Float16](repeating: 0, count: M * N)
    let resultBufferPointer = bufferC?.contents().bindMemory(to: Float16.self, capacity: M * N)

    if let ptr = resultBufferPointer {
      resultMatrix = Array(UnsafeBufferPointer(start: ptr, count: M * N))
    }

    // Optional: Verify the result on the CPU
    for m in 0..<M {
      for n in 0..<N {
        var expected: Float = Float(bias[n])
        for i in 0..<K {
          expected += Float(matrixA[i + m * K]) * Float(matrixB[i + n * K])
        }
        if abs(expected - Float(resultMatrix[m * N + n])) > 5e-3 {
          print("CPU calculated C[\(m), \(n)]: \(expected)")
          print("GPU calculated C[\(m), \(n)]: \(resultMatrix[m * N + n])")
        }
      }
    }
  }
}
