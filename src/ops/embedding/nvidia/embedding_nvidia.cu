#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr unsigned int BLOCK_SIZE = 256;

template <typename T>
__global__ void embeddingKernel(
    T *out,
    const int64_t *indices,
    const T *weight,
    size_t num_indices,
    size_t vocab_size,
    size_t embedding_dim) {

    size_t pos =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    size_t total = num_indices * embedding_dim;

    if (pos >= total) {
        return;
    }

    size_t row = pos / embedding_dim;
    size_t col = pos % embedding_dim;

    int64_t idx = indices[row];

    if (idx < 0 || static_cast<size_t>(idx) >= vocab_size) {
        return;
    }

    out[pos] =
        weight[static_cast<size_t>(idx) * embedding_dim + col];
}

template <typename T>
void launchEmbedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    size_t num_indices,
    size_t vocab_size,
    size_t embedding_dim,
    cudaStream_t stream) {

    size_t total = num_indices * embedding_dim;

    if (total == 0) {
        return;
    }

    unsigned int grid_size =
        static_cast<unsigned int>(
            (total + BLOCK_SIZE - 1) / BLOCK_SIZE);

    embeddingKernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const int64_t *>(index),
        reinterpret_cast<const T *>(weight),
        num_indices,
        vocab_size,
        embedding_dim);

    cudaError_t err = cudaGetLastError();
    ASSERT(err == cudaSuccess, cudaGetErrorString(err));
}

} // namespace

namespace llaisys::ops::nvidia {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t num_indices,
    size_t vocab_size,
    size_t embedding_dim,
    llaisysStream_t stream) {

    auto cuda_stream =
        reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchEmbedding<float>(
            out,
            index,
            weight,
            num_indices,
            vocab_size,
            embedding_dim,
            cuda_stream);

    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        return launchEmbedding<uint16_t>(
            out,
            index,
            weight,
            num_indices,
            vocab_size,
            embedding_dim,
            cuda_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
