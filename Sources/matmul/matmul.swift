import Metal
import Foundation


struct GEMMDimensions {
  var M: Int
  var N: Int
  var K: Int
}

func createSource(matrixDimensions: GEMMDimensions, blockDimensions: GEMMDimensions) -> String {
  return """

#include <metal_stdlib>
#include <metal_tensor>
// Doesn't seem this header is provided at runtime, hence JIT shader won't work unless we package all the headers too.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul_static_slice_dynamic_extents(device half *A_buf [[buffer(0)]],
                         device half *B_buf [[buffer(1)]],
                         device half *C_buf [[buffer(2)]],
                         uint2 tgid [[threadgroup_position_in_grid]])
{
    // Construct shader allocated tensors. This is easier since we can just bind buffer directly with Metal 3 APIs.
    auto A = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(\(matrixDimensions.K), \(matrixDimensions.M)));
    auto B = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(\(matrixDimensions.K), \(matrixDimensions.N)));
    auto C = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(\(matrixDimensions.N), \(matrixDimensions.M)));
    // descriptor to create matmul operation that does 64x128 times 64x64 producing 64x128
    constexpr auto matmulDescriptor = matmul2d_descriptor(\(blockDimensions.M), \(blockDimensions.N), \(blockDimensions.K), false, true, true, matmul2d_descriptor::mode::multiply_accumulate);

    // create matmul op from above descriptor with 4 SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;
        
    auto mA = A.slice<\(blockDimensions.K), \(blockDimensions.M)>(0, tgid.y * \(blockDimensions.M));
    auto mB = B.slice<\(blockDimensions.K), \(blockDimensions.N)>(0, tgid.x * \(blockDimensions.N));
    auto cT = matmulOp.get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cT.get_capacity(); ++k) {
      if(cT.is_valid_element(k)) {
        // auto idx = cT.get_multidimensional_index(k);
        cT[k] = 0;
      }
    }

    for (ushort k = 0; k < \(matrixDimensions.K); k += \(blockDimensions.K)) {
        // Create appropriate slice for this thread group to work on.
        auto mA = A.slice<\(blockDimensions.K), \(blockDimensions.M)>(k, tgid.y * \(blockDimensions.M));
        auto mB = B.slice<\(blockDimensions.K), \(blockDimensions.N)>(k, tgid.x * \(blockDimensions.N));

        // execute the operation. Assumes C is is initialized to zero.
        matmulOp.run(mA, mB, cT);
    }
    auto mC = C.slice<\(blockDimensions.N), \(blockDimensions.M)>(tgid.x * \(blockDimensions.N), tgid.y * \(blockDimensions.M));
    cT.store(mC);
}

kernel void matmul_static_slice_static_extents(device half *A_buf [[buffer(0)]],
                         device half *B_buf [[buffer(1)]],
                         device half *C_buf [[buffer(2)]],
                         uint2 tgid [[threadgroup_position_in_grid]])
{
    // Construct shader allocated tensors. This is easier since we can just bind buffer directly with Metal 3 APIs.
  // Use static extents. Note that these shapes are template parameters, it is fixed at compile-time.
    auto A = tensor<device half,  extents<int32_t, 3072, 3072>, tensor_inline>(A_buf, extents<int32_t, 3072, 3072>());
    auto B = tensor<device half,  extents<int32_t, 3072, 3072>, tensor_inline>(B_buf, extents<int32_t, 3072, 3072>());
    auto C = tensor<device half,  extents<int32_t, 3072, 3072>, tensor_inline>(C_buf, extents<int32_t, 3072, 3072>());
    // descriptor to create matmul operation that does 64x32 times 32x32 producing 64x64
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, 64, 64, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);

    // create matmul op from above descriptor with 4 SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;
        
    auto mA = A.slice<64, 64>(0, tgid.y * 64);
    auto mB = B.slice<64, 64>(0, tgid.x * 64);
    auto cT = matmulOp.get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cT.get_capacity(); ++k) {
      if(cT.is_valid_element(k)) {
        // auto idx = cT.get_multidimensional_index(k);
        cT[k] = 0;
      }
    }

    for (ushort k = 0; k < 3072; k += 64) {
        // Create appropriate slice for this thread group to work on.
        auto mA = A.slice<64, 64>(k, tgid.y * 64);
        auto mB = B.slice<64, 64>(k, tgid.x * 64);

        // execute the operation. Assumes C is is initialized to zero.
        matmulOp.run(mA, mB, cT);
    }
    auto mC = C.slice<64, 64>(tgid.x * 64, tgid.y * 64);
    cT.store(mC);
}

kernel void matmul_auto_slice_dynamic_extents_and_bias(device half *A_buf [[buffer(0)]],
                         device half *B_buf [[buffer(1)]],
                         device half *C_buf [[buffer(2)]],
                         device half *bias_buf [[buffer(3)]],
                         uint2 tgid [[threadgroup_position_in_grid]])
{
    // Construct shader allocated tensors. This is easier since we can just bind buffer directly with Metal 3 APIs.
    auto A = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(3072, 3072));
    auto B = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(3072, 3072));
    auto C = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(3072, 3072));
    // descriptor to create matmul operation that does 64x32 times 32x32 producing 64x32
    // Note that for K, we use dynamic_length_v<int> rather than "0" in some examples (these are wrong).
    constexpr auto matmulDescriptor = matmul2d_descriptor(48, 48, dynamic_length_v<int>, false, true, false, matmul2d_descriptor::mode::multiply);

    // create matmul op from above descriptor with 4 SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    // Create appropriate slice for this thread group to work on.
    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(0, tgid.x * 64);
    auto mC = C.slice(tgid.x * 64, tgid.y * 64);
    auto cT = matmulOp.get_destination_cooperative_tensor<decltype(mA), decltype(mB), half>();
    
    #pragma clang loop unroll(full)
    for (unsigned short k = 0; k < cT.get_capacity(); ++k) {
      if(cT.is_valid_element(k)) {
        // auto idx = cT.get_multidimensional_index(k);
        cT[k] = 0; // bias_buf[idx[0] + tgid.x * 64];
      }
    }
    // execute the operation. Assumes C is is initialized to zero.
    matmulOp.run(mA, mB, cT);
    cT.store(mC);
}

"""
}

@main
struct matmul {
    static func main() {
        run()
    }

    static func run() {
      
      // 6. Prepare data
      let M = 3072
      let N = 3072 * 4
      let K = 3072
      let blockDimensions = GEMMDimensions(M: 128, N: 64, K: 64)

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
            library = try device.makeLibrary(source: createSource(matrixDimensions: GEMMDimensions(M: M, N: N, K: K), blockDimensions: blockDimensions), options: nil)
            // library = try device.makeLibrary(URL: URL(fileURLWithPath: "Sources/matmul/default.metallib"))
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
            bias[i] = Float16.random(in: 1...2)
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
      let threadgroups = MTLSize(width: (N + blockDimensions.N - 1) / blockDimensions.N, height: (M + blockDimensions.M - 1) / blockDimensions.M, depth: 1)
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
          var expected: Float = 0 // Float(bias[n])
          for i in 0..<K {
              // C[0,0] = sum(A[0,k] * B[k,0])
              // A[0,k] is matrixA[i]
              // B[k,0] is matrixB[i * N]
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
