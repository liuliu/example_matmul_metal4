#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

kernel void simpleMatMul(tensor<device half,  dextents<int32_t, 2>, tensor_handle> A,
                         tensor<device half,  dextents<int32_t, 2>, tensor_handle> B,
                         tensor<device float, dextents<int32_t, 2>, tensor_handle> C,
                         uint2 tgid [[threadgroup_position_in_grid]])
{
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
