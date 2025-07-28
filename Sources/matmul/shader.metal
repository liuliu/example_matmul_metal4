#include <metal_stdlib>
#include <metal_tensor>
// Doesn't seem this header is provided at runtime, hence JIT shader won't work unless we package all the headers too.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void matmul_auto_slice_dynamic_extents(device half *A_buf [[buffer(0)]],
                         device half *B_buf [[buffer(1)]],
                         device half *C_buf [[buffer(2)]],
                         uint2 tgid [[threadgroup_position_in_grid]])
{
    // Construct shader allocated tensors. This is easier since we can just bind buffer directly with Metal 3 APIs.
    auto A = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(256, 128));
    auto B = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(64, 256));
    auto C = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(128, 64));
    // descriptor to create matmul operation that does 64x32 times 32x32 producing 64x32
	// Note that for K, we use dynamic_length_v<int> rather than "0" in some examples (these are wrong).
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, false, false, matmul2d_descriptor::mode::multiply);

    // create matmul op from above descriptor with 4 SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    // Create appropriate slice for this thread group to work on.
    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);

    // execute the operation. Assumes C is is initialized to zero.
    matmulOp.run(mA, mB, mC);
}

kernel void matmul_auto_slice_static_extents(device half *A_buf [[buffer(0)]],
                         device half *B_buf [[buffer(1)]],
                         device half *C_buf [[buffer(2)]],
                         uint2 tgid [[threadgroup_position_in_grid]])
{
    // Construct shader allocated tensors. This is easier since we can just bind buffer directly with Metal 3 APIs.
	// Use static extents. Note that these shapes are template parameters, it is fixed at compile-time.
    auto A = tensor<device half,  extents<int32_t, 256, 128>, tensor_inline>(A_buf, extents<int32_t, 256, 128>());
    auto B = tensor<device half,  extents<int32_t, 64, 256>, tensor_inline>(B_buf, extents<int32_t, 64, 256>());
    auto C = tensor<device half,  extents<int32_t, 128, 64>, tensor_inline>(C_buf, extents<int32_t, 128, 64>());
    // descriptor to create matmul operation that does 64x32 times 32x32 producing 64x32
	// Note that for K, we use dynamic_length_v<int> rather than "0" in some examples (these are wrong).
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, 32, dynamic_length_v<int>, false, false, false, matmul2d_descriptor::mode::multiply);

    // create matmul op from above descriptor with 4 SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    // Create appropriate slice for this thread group to work on.
    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);

    // execute the operation. Assumes C is is initialized to zero.
    matmulOp.run(mA, mB, mC);
}

kernel void matmul_static_slice_dynamic_extents(device half *A_buf [[buffer(0)]],
                         device half *B_buf [[buffer(1)]],
                         device half *C_buf [[buffer(2)]],
                         uint2 tgid [[threadgroup_position_in_grid]])
{
    // Construct shader allocated tensors. This is easier since we can just bind buffer directly with Metal 3 APIs.
    auto A = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(256, 128));
    auto B = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(64, 256));
    auto C = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(128, 64));
    // descriptor to create matmul operation that does 64x32 times 32x32 producing 64x32
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);

    // create matmul op from above descriptor with 4 SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    for (int k = 0; k < 256; k += 16) {
        // Create appropriate slice for this thread group to work on.
        auto mA = A.slice<16, 64>(k, tgid.y * 64);
        auto mB = B.slice<32, 16>(tgid.x * 32, k);
        auto mC = C.slice<32, 64>(tgid.x * 32, tgid.y * 64);

        // execute the operation. Assumes C is is initialized to zero.
        matmulOp.run(mA, mB, mC);
    }
}

kernel void matmul_static_slice_static_extents(device half *A_buf [[buffer(0)]],
                         device half *B_buf [[buffer(1)]],
                         device half *C_buf [[buffer(2)]],
                         uint2 tgid [[threadgroup_position_in_grid]])
{
    // Construct shader allocated tensors. This is easier since we can just bind buffer directly with Metal 3 APIs.
	// Use static extents. Note that these shapes are template parameters, it is fixed at compile-time.
    auto A = tensor<device half,  extents<int32_t, 256, 128>, tensor_inline>(A_buf, extents<int32_t, 256, 128>());
    auto B = tensor<device half,  extents<int32_t, 64, 256>, tensor_inline>(B_buf, extents<int32_t, 64, 256>());
    auto C = tensor<device half,  extents<int32_t, 128, 64>, tensor_inline>(C_buf, extents<int32_t, 128, 64>());
    // descriptor to create matmul operation that does 64x32 times 32x32 producing 64x32
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, 32, 16, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);

    // create matmul op from above descriptor with 4 SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    for (int k = 0; k < 256; k += 16) {
        // Create appropriate slice for this thread group to work on.
        auto mA = A.slice<16, 64>(k, tgid.y * 64);
        auto mB = B.slice<32, 16>(tgid.x * 32, k);
        auto mC = C.slice<32, 64>(tgid.x * 32, tgid.y * 64);

        // execute the operation. Assumes C is is initialized to zero.
        matmulOp.run(mA, mB, mC);
    }
}
