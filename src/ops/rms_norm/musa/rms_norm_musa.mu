#include "rms_norm_musa.muh"

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
__global__ void rmsNormKernel(
    T *out,
    const T *in,
    const T *weight,
    size_t hidden_size,
    float eps) {

    __shared__ float shared[BLOCK_SIZE];

    size_t row = static_cast<size_t>(blockIdx.x);
    unsigned int tid = threadIdx.x;

    size_t base = row * hidden_size;

    float sum_sq = 0.0f;

    for (size_t j = tid; j < hidden_size; j += blockDim.x) {
        float x = toFloat(in[base + j]);
        sum_sq += x * x;
    }

    shared[tid] = sum_sq;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        float mean_sq =
            shared[0] / static_cast<float>(hidden_size);

        shared[0] = rsqrtf(mean_sq + eps);
    }

    __syncthreads();

    float inv_rms = shared[0];

    for (size_t j = tid; j < hidden_size; j += blockDim.x) {
        float x = toFloat(in[base + j]);
        float w = toFloat(weight[j]);

        storeValue(
            &out[base + j],
            x * inv_rms * w);
    }
}

template <typename T>
void launchRmsNorm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t rows,
    size_t hidden_size,
    float eps,
    musaStream_t stream) {

    if (rows == 0) {
        return;
    }

    rmsNormKernel<T><<<
        static_cast<unsigned int>(rows),
        BLOCK_SIZE,
        0,
        stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        hidden_size,
        eps);

    musaError_t err = musaGetLastError();
    ASSERT(err == musaSuccess, musaGetErrorString(err));
}

} // namespace

namespace llaisys::ops::musa {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t rows,
    size_t hidden_size,
    float eps,
    llaisysStream_t stream) {

    auto musa_stream =
        reinterpret_cast<musaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float>(
            out, in, weight,
            rows, hidden_size, eps, musa_stream);

    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<__half>(
            out, in, weight,
            rows, hidden_size, eps, musa_stream);

    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<__mt_bfloat16>(
            out, in, weight,
            rows, hidden_size, eps, musa_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::musa
