#include "self_attention_nvidia.cuh"

#include "../../../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>

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
__device__ float attentionScore(
    const T *q,
    const T *k,
    size_t q_base,
    size_t k_base,
    size_t hd,
    float scale) {

    float dot = 0.0f;

    for (size_t d = 0; d < hd; ++d) {
        dot +=
            toFloat(q[q_base + d])
            * toFloat(k[k_base + d]);
    }

    return dot * scale;
}

template <typename T>
__global__ void selfAttentionKernel(
    T *out,
    const T *q,
    const T *k,
    const T *v,
    size_t qlen,
    size_t kvlen,
    size_t nh,
    size_t nkvh,
    size_t hd,
    float scale) {

    __shared__ float shared[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;

    size_t pair = static_cast<size_t>(blockIdx.x);
    size_t qi = pair / nh;
    size_t h = pair % nh;

    if (qi >= qlen) {
        return;
    }

    size_t group_size = nh / nkvh;
    size_t kv_head = h / group_size;

    size_t allowed_keys =
        kvlen - qlen + qi + 1;

    size_t q_base =
        (qi * nh + h) * hd;

    // Pass 1: find maximum score for stable softmax.
    float local_max = -FLT_MAX;

    for (size_t ki = tid;
         ki < allowed_keys;
         ki += blockDim.x) {

        size_t k_base =
            (ki * nkvh + kv_head) * hd;

        float score =
            attentionScore(
                q, k,
                q_base, k_base,
                hd, scale);

        if (score > local_max) {
            local_max = score;
        }
    }

    shared[tid] = local_max;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            shared[tid] =
                fmaxf(shared[tid],
                      shared[tid + stride]);
        }

        __syncthreads();
    }

    float max_score = shared[0];

    // Pass 2: compute softmax denominator.
    float local_sum = 0.0f;

    for (size_t ki = tid;
         ki < allowed_keys;
         ki += blockDim.x) {

        size_t k_base =
            (ki * nkvh + kv_head) * hd;

        float score =
            attentionScore(
                q, k,
                q_base, k_base,
                hd, scale);

        local_sum += expf(score - max_score);
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (unsigned int stride = BLOCK_SIZE / 2;
         stride > 0;
         stride >>= 1) {

        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    float sum_exp = shared[0];

    // Pass 3: compute softmax weights in KV chunks,
    // then accumulate weight @ V.
    for (size_t d_base = 0;
         d_base < hd;
         d_base += BLOCK_SIZE) {

        size_t d = d_base + tid;
        float result = 0.0f;

        for (size_t key_base = 0;
             key_base < allowed_keys;
             key_base += BLOCK_SIZE) {

            size_t ki = key_base + tid;

            if (ki < allowed_keys) {
                size_t k_base =
                    (ki * nkvh + kv_head) * hd;

                float score =
                    attentionScore(
                        q, k,
                        q_base, k_base,
                        hd, scale);

                shared[tid] =
                    expf(score - max_score) / sum_exp;
            } else {
                shared[tid] = 0.0f;
            }

            __syncthreads();

            if (d < hd) {
                size_t chunk =
                    allowed_keys - key_base;

                if (chunk > BLOCK_SIZE) {
                    chunk = BLOCK_SIZE;
                }

                for (size_t j = 0; j < chunk; ++j) {
                    size_t key = key_base + j;

                    size_t v_base =
                        (key * nkvh + kv_head) * hd;

                    result +=
                        shared[j]
                        * toFloat(v[v_base + d]);
                }
            }

            __syncthreads();
        }

        if (d < hd) {
            storeValue(
                &out[q_base + d],
                result);
        }
    }
}

template <typename T>
void launchSelfAttention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    size_t qlen,
    size_t kvlen,
    size_t nh,
    size_t nkvh,
    size_t hd,
    float scale,
    cudaStream_t stream) {

    if (qlen == 0) {
        return;
    }

    size_t blocks = qlen * nh;

    selfAttentionKernel<T><<<
        static_cast<unsigned int>(blocks),
        BLOCK_SIZE,
        0,
        stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        reinterpret_cast<const T *>(v),
        qlen,
        kvlen,
        nh,
        nkvh,
        hd,
        scale);

    cudaError_t err = cudaGetLastError();
    ASSERT(err == cudaSuccess, cudaGetErrorString(err));
}

} // namespace

namespace llaisys::ops::nvidia {

void self_attention(
    std::byte *out,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t qlen,
    size_t kvlen,
    size_t nh,
    size_t nkvh,
    size_t hd,
    float scale,
    llaisysStream_t stream) {

    auto cuda_stream =
        reinterpret_cast<cudaStream_t>(stream);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchSelfAttention<float>(
            out, q, k, v,
            qlen, kvlen, nh, nkvh, hd,
            scale, cuda_stream);

    case LLAISYS_DTYPE_F16:
        return launchSelfAttention<__half>(
            out, q, k, v,
            qlen, kvlen, nh, nkvh, hd,
            scale, cuda_stream);

    case LLAISYS_DTYPE_BF16:
        return launchSelfAttention<__nv_bfloat16>(
            out, q, k, v,
            qlen, kvlen, nh, nkvh, hd,
            scale, cuda_stream);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
