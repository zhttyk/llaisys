#include "add_musa.muh"

#include "../../../utils.hpp"

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace {

__device__ inline float addValue(float a, float b) {
    return a + b;
}

__device__ inline __half addValue(__half a, __half b) {
    return __float2half(
        __half2float(a) + __half2float(b));
}

__device__ inline __mt_bfloat16 addValue(
    __mt_bfloat16 a,
    __mt_bfloat16 b) {

    return __float2bfloat16(
        __bfloat162float(a) +
        __bfloat162float(b));
}

template <typename T>
__global__ void addKernel(
    T *c,
    const T *a,
    const T *b,
    size_t numel) {

    size_t i =
        static_cast<size_t>(blockIdx.x) *
            blockDim.x +
        threadIdx.x;

    if (i < numel) {
        c[i] = addValue(a[i], b[i]);
    }
}

template <typename T>
void launchAdd(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    size_t numel,
    musaStream_t stream) {

    if (numel == 0) {
        return;
    }

    constexpr unsigned int BLOCK_SIZE = 256;

    unsigned int grid_size =
        static_cast<unsigned int>(
            (numel + BLOCK_SIZE - 1) /
            BLOCK_SIZE);

    addKernel<T>
        <<<grid_size, BLOCK_SIZE, 0, stream>>>(
            reinterpret_cast<T *>(c),
            reinterpret_cast<const T *>(a),
            reinterpret_cast<const T *>(b),
            numel);

    musaError_t err = musaGetLastError();

    ASSERT(
        err == musaSuccess,
        musaGetErrorString(err));
}

} // namespace

namespace llaisys::ops::musa {

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream) {

    auto musa_stream =
        reinterpret_cast<musaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchAdd<float>(
            c,
            a,
            b,
            numel,
            musa_stream);

    case LLAISYS_DTYPE_F16:
        return launchAdd<__half>(
            c,
            a,
            b,
            numel,
            musa_stream);

    case LLAISYS_DTYPE_BF16:
        return launchAdd<__mt_bfloat16>(
            c,
            a,
            b,
            numel,
            musa_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::musa
