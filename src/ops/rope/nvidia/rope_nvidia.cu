#include "rope_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr unsigned int BLOCK_SIZE = 256;

__device__ inline float toFloat(float value) {
    return value;
}

__device__ inline float toFloat(__half value) {
    return __half2float(value);
}

__device__ inline float toFloat(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ inline void storeValue(float *dst, float value) {
    *dst = value;
}

__device__ inline void storeValue(__half *dst, float value) {
    *dst = __float2half_rn(value);
}

__device__ inline void storeValue(__nv_bfloat16 *dst, float value) {
    *dst = __float2bfloat16_rn(value);
}

template <typename T>
__global__ void ropeKernel(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta) {

    size_t half_dim = head_dim / 2;
    size_t total = seq_len * half_dim;

    size_t pos =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (pos >= total) {
        return;
    }

    size_t s = pos / half_dim;
    size_t j = pos % half_dim;

    float exponent =
        2.0f * static_cast<float>(j)
        / static_cast<float>(head_dim);

    float denominator = powf(theta, exponent);
    float angle =
        static_cast<float>(pos_ids[s]) / denominator;

    float sin_val;
    float cos_val;
    sincosf(angle, &sin_val, &cos_val);

    for (size_t h = 0; h < n_heads; ++h) {
        size_t base =
            (s * n_heads + h) * head_dim;

        size_t a_idx = base + j;
        size_t b_idx = base + half_dim + j;

        float a = toFloat(in[a_idx]);
        float b = toFloat(in[b_idx]);

        storeValue(
            &out[a_idx],
            a * cos_val - b * sin_val);

        storeValue(
            &out[b_idx],
            b * cos_val + a * sin_val);
    }
}

template <typename T>
void launchRope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta,
    cudaStream_t stream) {

    size_t total =
        seq_len * (head_dim / 2);

    if (total == 0) {
        return;
    }

    unsigned int grid_size =
        static_cast<unsigned int>(
            (total + BLOCK_SIZE - 1) / BLOCK_SIZE);

    ropeKernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(pos_ids),
        seq_len,
        n_heads,
        head_dim,
        theta);

    cudaError_t err = cudaGetLastError();
    ASSERT(err == cudaSuccess, cudaGetErrorString(err));
}

} // namespace

namespace llaisys::ops::nvidia {

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta,
    llaisysStream_t stream) {

    auto cuda_stream =
        reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRope<float>(
            out, in, pos_ids,
            seq_len, n_heads, head_dim,
            theta, cuda_stream);

    case LLAISYS_DTYPE_F16:
        return launchRope<__half>(
            out, in, pos_ids,
            seq_len, n_heads, head_dim,
            theta, cuda_stream);

    case LLAISYS_DTYPE_BF16:
        return launchRope<__nv_bfloat16>(
            out, in, pos_ids,
            seq_len, n_heads, head_dim,
            theta, cuda_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
