#include "linear_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr unsigned int TILE = 16;

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
__global__ void linearKernel(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t m,
    size_t n,
    size_t k) {

    __shared__ float in_tile[TILE][TILE];
    __shared__ float weight_tile[TILE][TILE];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    size_t row =
        static_cast<size_t>(blockIdx.y) * TILE + ty;
    size_t col =
        static_cast<size_t>(blockIdx.x) * TILE + tx;

    float sum = 0.0f;

    if (col < n && bias != nullptr) {
        sum = toFloat(bias[col]);
    }

    for (size_t tile = 0; tile < k; tile += TILE) {
        size_t in_col = tile + tx;
        size_t weight_k = tile + tx;
        size_t weight_row =
            static_cast<size_t>(blockIdx.x) * TILE + ty;

        if (row < m && in_col < k) {
            in_tile[ty][tx] =
                toFloat(in[row * k + in_col]);
        } else {
            in_tile[ty][tx] = 0.0f;
        }

        if (weight_row < n && weight_k < k) {
            weight_tile[ty][tx] =
                toFloat(weight[weight_row * k + weight_k]);
        } else {
            weight_tile[ty][tx] = 0.0f;
        }

        __syncthreads();

        if (row < m && col < n) {
#pragma unroll
            for (unsigned int p = 0; p < TILE; ++p) {
                sum +=
                    in_tile[ty][p]
                    * weight_tile[tx][p];
            }
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        storeValue(&out[row * n + col], sum);
    }
}

template <typename T>
void launchLinear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    size_t m,
    size_t n,
    size_t k,
    cudaStream_t stream) {

    dim3 block(TILE, TILE);

    dim3 grid(
        static_cast<unsigned int>((n + TILE - 1) / TILE),
        static_cast<unsigned int>((m + TILE - 1) / TILE));

    linearKernel<T><<<grid, block, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        bias ? reinterpret_cast<const T *>(bias) : nullptr,
        m,
        n,
        k);

    cudaError_t err = cudaGetLastError();
    ASSERT(err == cudaSuccess, cudaGetErrorString(err));
}

} // namespace

namespace llaisys::ops::nvidia {

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t m,
    size_t n,
    size_t k,
    llaisysStream_t stream) {

    auto cuda_stream =
        reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchLinear<float>(
            out, in, weight, bias,
            m, n, k, cuda_stream);

    case LLAISYS_DTYPE_F16:
        return launchLinear<__half>(
            out, in, weight, bias,
            m, n, k, cuda_stream);

    case LLAISYS_DTYPE_BF16:
        return launchLinear<__nv_bfloat16>(
            out, in, weight, bias,
            m, n, k, cuda_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
