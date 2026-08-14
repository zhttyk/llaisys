#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/self_attention_musa.muh"
#endif

#include <cmath>
#include <limits>
#include <vector>

namespace {

template <typename T>
void self_attention_cpu(
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

    size_t group_size = nh / nkvh;

    // Reused for every (query head, query position).
    std::vector<float> scores(kvlen);

    for (size_t h = 0; h < nh; ++h) {
        // GQA: multiple query heads may share one KV head.
        size_t kv_head = h / group_size;

        for (size_t qi = 0; qi < qlen; ++qi) {
            // Equivalent to tril(diagonal=kvlen-qlen).
            size_t allowed_keys =
                kvlen - qlen + qi + 1;

            size_t q_base =
                (qi * nh + h) * hd;

            float max_score =
                -std::numeric_limits<float>::infinity();

            // 1. Q @ K^T, scale, and causal mask.
            for (size_t ki = 0; ki < allowed_keys; ++ki) {
                size_t k_base =
                    (ki * nkvh + kv_head) * hd;

                float dot = 0.0f;

                for (size_t d = 0; d < hd; ++d) {
                    float q_val =
                        llaisys::utils::cast<float>(
                            q[q_base + d]);

                    float k_val =
                        llaisys::utils::cast<float>(
                            k[k_base + d]);

                    dot += q_val * k_val;
                }

                float score = dot * scale;
                scores[ki] = score;

                if (score > max_score) {
                    max_score = score;
                }
            }

            // 2. Stable softmax.
            float sum_exp = 0.0f;

            for (size_t ki = 0; ki < allowed_keys; ++ki) {
                float exp_val =
                    std::exp(scores[ki] - max_score);

                scores[ki] = exp_val;
                sum_exp += exp_val;
            }

            for (size_t ki = 0; ki < allowed_keys; ++ki) {
                scores[ki] /= sum_exp;
            }

            // 3. attention_weight @ V.
            for (size_t d = 0; d < hd; ++d) {
                float result = 0.0f;

                for (size_t ki = 0; ki < allowed_keys; ++ki) {
                    size_t v_base =
                        (ki * nkvh + kv_head) * hd;

                    float v_val =
                        llaisys::utils::cast<float>(
                            v[v_base + d]);

                    result += scores[ki] * v_val;
                }

                out[q_base + d] =
                    llaisys::utils::cast<T>(result);
            }
        }
    }
}

} // namespace

namespace llaisys::ops {

void self_attention(
    tensor_t attn_val,
    tensor_t q,
    tensor_t k,
    tensor_t v,
    float scale) {

    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(
        attn_val->dtype(),
        q->dtype(),
        k->dtype(),
        v->dtype());

    CHECK_SAME_SHAPE(attn_val->shape(), q->shape());
    CHECK_SAME_SHAPE(k->shape(), v->shape());

    CHECK_ARGUMENT(
        q->ndim() == 3,
        "SelfAttention: query must be a 3D tensor.");

    CHECK_ARGUMENT(
        k->ndim() == 3 && v->ndim() == 3,
        "SelfAttention: key and value must be 3D tensors.");

    CHECK_ARGUMENT(
        q->shape()[2] == k->shape()[2],
        "SelfAttention: query and key head dimensions must match.");

    CHECK_ARGUMENT(
        q->shape()[2] == v->shape()[2],
        "SelfAttention: query and value head dimensions must match.");

    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];
    size_t hd = q->shape()[2];

    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];

    CHECK_ARGUMENT(
        nh > 0 && nkvh > 0 && nh % nkvh == 0,
        "SelfAttention: number of query heads must be divisible by KV heads.");

    CHECK_ARGUMENT(
        kvlen >= qlen,
        "SelfAttention: KV sequence length must be at least query length.");

    CHECK_ARGUMENT(
        hd > 0,
        "SelfAttention: head dimension must be positive.");

    ASSERT(
        attn_val->isContiguous()
            && q->isContiguous()
            && k->isContiguous()
            && v->isContiguous(),
        "SelfAttention: all tensors must be contiguous.");

    if (attn_val->deviceType() != LLAISYS_DEVICE_CPU) {
        llaisys::core::context().setDevice(
            attn_val->deviceType(),
            attn_val->deviceId());

        switch (attn_val->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::self_attention(
                attn_val->data(),
                q->data(),
                k->data(),
                v->data(),
                attn_val->dtype(),
                qlen,
                kvlen,
                nh,
                nkvh,
                hd,
                scale,
                llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_MUSA_API
        case LLAISYS_DEVICE_MUSA:
            return musa::self_attention(
                attn_val->data(),
                q->data(),
                k->data(),
                v->data(),
                attn_val->dtype(),
                qlen,
                kvlen,
                nh,
                nkvh,
                hd,
                scale,
                llaisys::core::context().runtime().stream());
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
    }

    switch (attn_val->dtype()) {
    case LLAISYS_DTYPE_F32:
        return self_attention_cpu(
            reinterpret_cast<float *>(attn_val->data()),
            reinterpret_cast<const float *>(q->data()),
            reinterpret_cast<const float *>(k->data()),
            reinterpret_cast<const float *>(v->data()),
            qlen,
            kvlen,
            nh,
            nkvh,
            hd,
            scale);

    case LLAISYS_DTYPE_F16:
        return self_attention_cpu(
            reinterpret_cast<llaisys::fp16_t *>(attn_val->data()),
            reinterpret_cast<const llaisys::fp16_t *>(q->data()),
            reinterpret_cast<const llaisys::fp16_t *>(k->data()),
            reinterpret_cast<const llaisys::fp16_t *>(v->data()),
            qlen,
            kvlen,
            nh,
            nkvh,
            hd,
            scale);

    case LLAISYS_DTYPE_BF16:
        return self_attention_cpu(
            reinterpret_cast<llaisys::bf16_t *>(attn_val->data()),
            reinterpret_cast<const llaisys::bf16_t *>(q->data()),
            reinterpret_cast<const llaisys::bf16_t *>(k->data()),
            reinterpret_cast<const llaisys::bf16_t *>(v->data()),
            qlen,
            kvlen,
            nh,
            nkvh,
            hd,
            scale);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
    }
}

} // namespace llaisys::ops
