#include "swiglu_musa.muh"

#include "../../../utils.hpp"

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace {

constexpr unsigned int BLOCK_SIZE = 256;

__device__ inline float toFloat(float value) {
    return value;
}

__device__ inline float toFloat(__half value) {
    return __half2float(value);
}

__device__ inline float toFloat(__mt_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ inline void storeValue(float *dst, float value) {
    *dst = value;
}

__device__ inline void storeValue(__half *dst, float value) {
    *dst = __float2half_rn(value);
}

__device__ inline void storeValue(__mt_bfloat16 *dst, float value) {
    *dst = __float2bfloat16_rn(value);
}

template <typename T>
__global__ void swigluKernel(
    T *out,
    const T *gate,
    const T *up,
    size_t numel) {

    size_t i =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= numel) {
        return;
    }

    float gate_val = toFloat(gate[i]);
    float up_val = toFloat(up[i]);

    // Match the CPU/PyTorch test semantics:
    // exp is calculated in float32, then rounded to output dtype.
    T exp_t;
    storeValue(&exp_t, expf(-gate_val));

    float exp_val = toFloat(exp_t);

    float silu =
        gate_val / (1.0f + exp_val);

    storeValue(
        &out[i],
        up_val * silu);
}

template <typename T>
void launchSwiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    size_t numel,
    musaStream_t stream) {

    if (numel == 0) {
        return;
    }

    unsigned int grid_size =
        static_cast<unsigned int>(
            (numel + BLOCK_SIZE - 1) / BLOCK_SIZE);

    swigluKernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);

    musaError_t err = musaGetLastError();
    ASSERT(err == musaSuccess, musaGetErrorString(err));
}

} // namespace

namespace llaisys::ops::musa {

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream) {

    auto musa_stream =
        reinterpret_cast<musaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchSwiglu<float>(
            out, gate, up, numel, musa_stream);

    case LLAISYS_DTYPE_F16:
        return launchSwiglu<__half>(
            out, gate, up, numel, musa_stream);

    case LLAISYS_DTYPE_BF16:
        return launchSwiglu<__mt_bfloat16>(
            out, gate, up, numel, musa_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::musa
