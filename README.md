# MatMul with Metal 4

This repository provides a working example of matrix multiplication using the new tensor APIs introduced in Metal 4.

## Background

Metal 4 introduced a set of tensor APIs for performing matrix multiplication directly from compute shaders. However, the documentation on how to use these APIs is sparse and contains some errors. For example, the `MPPTensorOpsMatMul2d.h` header (from Xcode 26 Beta 4) contains obsolete references to `.offset` and `.static_slice` APIs that no longer exist. The [Metal 4 Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) is a better resource, but it still has some issues. For instance, the following code will not work as expected because `0 != dynamic_length_v<int>`, so the automatic slicing won't be triggered:

```cpp
constexpr auto matmulDescriptor = tensor_ops::matmul2d_descriptor(64, 32, 0);
```

This repository provides a complete, working example of the new tensor APIs to help developers explore their capabilities. To keep things familiar, the host-side code uses Metal 3 APIs.

## Getting Started

To run the program, simply use the following command:

```bash
swift run
```

You can change the kernels to call from the code. The matrix shapes are hard-coded in the shader, so if you change them, you will need to update the shader code and re-generate the `metallib` file by running the following commands:

```bash
cd ./Sources/matmul/
bash package.sh
```

## Q & A

**Q: Does `metal_tensor` require Metal 4?**

**A:** It appears not. While Metal 4 provides methods to bind host-bound tensor objects to shaders through `MTLArgumentTable`, you can still use Metal 3 APIs to work with tensors. This was easier for this example because I couldn't figure out how to call `waitUntilCompleted()` with Metal 4.

**Q: What about `MachineLearningCommandEncoder`?**

**A:** To the best of my knowledge, `MachineLearningCommandEncoder` is used for running packaged CoreML models. Calling tensor operations from a shader program just uses a regular compute shader, so Metal 3 is sufficient.

**Q: What exactly is the example doing under the hood?**

**A:** Apple has heavily abstracted the tensor-related APIs. A quick disassembly of the `.air` file suggests that it's calling into Apple's own packaged APIs. The good news is that you don't need to load data into specific memory regions like CUDA's tensor memory. You only need to worry about device memory and threadgroup memory (I believe this corresponds to `dv` vs. `tg`).

## Benchmark

TBD.