#include "argmax_nvidia.cuh"

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

__device__ inline void writeValue(float *dst, float value) {
    *dst = value;
}

__device__ inline void writeValue(__half *dst, float value) {
    *dst = __float2half_rn(value);
}

__device__ inline void writeValue(__nv_bfloat16 *dst, float value) {
    *dst = __float2bfloat16_rn(value);
}

template <typename T>
__global__ void argmaxKernel(
    int64_t *max_idx,
    T *max_val,
    const T *vals,
    size_t numel) {

    __shared__ float shared_vals[BLOCK_SIZE];
    __shared__ int64_t shared_indices[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;

    float best_val = toFloat(vals[0]);
    int64_t best_idx = 0;

    for (size_t i = tid; i < numel; i += blockDim.x) {
        float val = toFloat(vals[i]);
        int64_t idx = static_cast<int64_t>(i);

        if (val > best_val || (val == best_val && idx < best_idx)) {
            best_val = val;
            best_idx = idx;
        }
    }

    shared_vals[tid] = best_val;
    shared_indices[tid] = best_idx;

    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = shared_vals[tid + stride];
            int64_t other_idx = shared_indices[tid + stride];

            if (other_val > shared_vals[tid]
                || (other_val == shared_vals[tid]
                    && other_idx < shared_indices[tid])) {

                shared_vals[tid] = other_val;
                shared_indices[tid] = other_idx;
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        max_idx[0] = shared_indices[0];
        writeValue(max_val, shared_vals[0]);
    }
}

template <typename T>
void launchArgmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    size_t numel,
    cudaStream_t stream) {

    argmaxKernel<T><<<1, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel);

    cudaError_t err = cudaGetLastError();
    ASSERT(err == cudaSuccess, cudaGetErrorString(err));
}

} // namespace

namespace llaisys::ops::nvidia {

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream) {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(
            max_idx, max_val, vals, numel, cuda_stream);

    case LLAISYS_DTYPE_F16:
        return launchArgmax<__half>(
            max_idx, max_val, vals, numel, cuda_stream);

    case LLAISYS_DTYPE_BF16:
        return launchArgmax<__nv_bfloat16>(
            max_idx, max_val, vals, numel, cuda_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
