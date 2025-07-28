import Metal
import Foundation

@main
struct matmul {
    static func main() {
        run()
    }

    static func run() {
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
            library = try device.makeLibrary(URL: URL(fileURLWithPath: "Sources/matmul/default.metallib"))
        } catch {
            fatalError("Could not create library: \(error).")
        }


        // 4. Create a function object
        guard let matmulFunction = library.makeFunction(name: "simpleMatMul") else {
            fatalError("Could not create function")
        }

        // 5. Create a compute pipeline state
        let pipelineState: MTLComputePipelineState
        do {
            pipelineState = try device.makeComputePipelineState(function: matmulFunction)
        } catch {
            fatalError("Could not create pipeline state: \(error)")
        }

        // 6. Prepare data
        let M = 128
        let N = 64
        let K = 256

        let sizeA = M * K * MemoryLayout<Float16>.size
        let sizeB = K * N * MemoryLayout<Float16>.size
        let sizeC = M * N * MemoryLayout<Float16>.size

        let bufferA = device.makeBuffer(length: sizeA, options: .storageModeShared)
        let bufferB = device.makeBuffer(length: sizeB, options: .storageModeShared)
        let bufferC = device.makeBuffer(length: sizeC, options: .storageModeShared)

        // Initialize matrices A and B with some values
        var matrixA = [Float16](repeating: 0, count: M * K)
        var matrixB = [Float16](repeating: 0, count: K * N)
        for i in 0..<(M * K) {
            matrixA[i] = Float16.random(in: 0...1)
        }
        for i in 0..<(K * N) {
            matrixB[i] = Float16.random(in: 0...1)
        }

        bufferA?.contents().copyMemory(from: matrixA, byteCount: sizeA)
        bufferB?.contents().copyMemory(from: matrixB, byteCount: sizeB)
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

        // 8. Dispatch threads
        let threadgroups = MTLSize(width: (N + 31) / 32, height: (M + 63) / 64, depth: 1)
        let simdgroupWidth = pipelineState.threadExecutionWidth
        let threadsPerThreadgroup = MTLSize(width: simdgroupWidth * 4, height: 1, depth: 1)

        computeCommandEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        computeCommandEncoder.endEncoding()

        // 9. Commit the command buffer and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // 10. Read the results
        var resultMatrix = [Float16](repeating: 0, count: M * N)
        let resultBufferPointer = bufferC?.contents().bindMemory(to: Float16.self, capacity: M * N)

        if let ptr = resultBufferPointer {
            resultMatrix = Array(UnsafeBufferPointer(start: ptr, count: M * N))
        }

        print("Result matrix (first 10 elements): \(resultMatrix.prefix(10))")

        // Optional: Verify the result on the CPU
        var expected: Float = 0
        for i in 0..<K {
            // C[0,0] = sum(A[0,k] * B[k,0])
            // A[0,k] is matrixA[i]
            // B[k,0] is matrixB[i * N]
            expected += Float(matrixA[i]) * Float(matrixB[i * N])
        }

        print("CPU calculated C[0]: \(expected)")
        print("GPU calculated C[0]: \(resultMatrix[0])")
    }
}
