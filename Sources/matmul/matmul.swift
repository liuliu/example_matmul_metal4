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

        // 2. Create a command queue
        guard let commandQueue = device.makeCommandQueue() else {
            fatalError("Could not create command queue")
        }

        // 3. Create a library from inline source
        let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        kernel void add_arrays(device const float* inA,
                               device const float* inB,
                               device float* result,
                               uint index [[thread_position_in_grid]]) {
            result[index] = inA[index] + inB[index];
        }
        """

        let library: MTLLibrary
        do {
            library = try device.makeLibrary(source: shaderSource, options: nil)
        } catch {
            fatalError("Could not create library: \(error)")
        }


        // 4. Create a function object
        guard let addFunction = library.makeFunction(name: "add_arrays") else {
            fatalError("Could not create function")
        }

        // 5. Create a compute pipeline state
        let pipelineState: MTLComputePipelineState
        do {
            pipelineState = try device.makeComputePipelineState(function: addFunction)
        } catch {
            fatalError("Could not create pipeline state: \(error)")
        }

        // 6. Prepare data
        let arrayLength = 10
        let dataSize = arrayLength * MemoryLayout<Float>.size

        let bufferA = device.makeBuffer(length: dataSize, options: .storageModeShared)
        let bufferB = device.makeBuffer(length: dataSize, options: .storageModeShared)
        let bufferResult = device.makeBuffer(length: dataSize, options: .storageModeShared)

        var vectorA = [Float](repeating: 0, count: arrayLength)
        var vectorB = [Float](repeating: 0, count: arrayLength)

        for i in 0..<arrayLength {
            vectorA[i] = Float(i)
            vectorB[i] = Float(i)
        }

        bufferA?.contents().copyMemory(from: vectorA, byteCount: dataSize)
        bufferB?.contents().copyMemory(from: vectorB, byteCount: dataSize)

        // 7. Create a command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Could not create command buffer or encoder")
        }

        computeCommandEncoder.setComputePipelineState(pipelineState)
        computeCommandEncoder.setBuffer(bufferA, offset: 0, index: 0)
        computeCommandEncoder.setBuffer(bufferB, offset: 0, index: 1)
        computeCommandEncoder.setBuffer(bufferResult, offset: 0, index: 2)

        // 8. Dispatch threads
        let gridSize = MTLSize(width: arrayLength, height: 1, depth: 1)
        var threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup
        if threadGroupSize > arrayLength {
            threadGroupSize = arrayLength
        }
        let threadgroupSize = MTLSize(width: threadGroupSize, height: 1, depth: 1)

        computeCommandEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
        computeCommandEncoder.endEncoding()

        // 9. Commit the command buffer and wait for completion
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // 10. Read the results
        var resultVector = [Float](repeating: 0, count: arrayLength)
        let resultBufferPointer = bufferResult?.contents().bindMemory(to: Float.self, capacity: arrayLength)

        for i in 0..<arrayLength {
            resultVector[i] = resultBufferPointer![i]
        }

        print("Result: \(resultVector)")
    }
}
