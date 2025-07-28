#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void simpleMatMul(device half *A_buf [[buffer(0)]],
                         device half *B_buf [[buffer(1)]],
                         device half *C_buf [[buffer(2)]],
                         uint2 tgid [[threadgroup_position_in_grid]])
{
    auto A = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(A_buf, dextents<int32_t, 2>(128, 128));
    auto B = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(B_buf, dextents<int32_t, 2>(128, 128));
    auto C = tensor<device half,  dextents<int32_t, 2>, tensor_inline>(C_buf, dextents<int32_t, 2>(128, 128));
    // descriptor to create matmul operation that does 64x32 times 32x32 producing 64x32
    constexpr auto matmulDescriptor = matmul2d_descriptor(64, 32, 0, false, false, false);

    // create matmul op from above descriptor with 4 SIMD-Groups.
    matmul2d<matmulDescriptor, execution_simdgroups<4>> matmulOp;

    // Create appropriate slice for this thread group to work on.
    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);

    // execute the operation. Assumes C is is initialized to zero.
    matmulOp.run(mA, mB, mC);
}
