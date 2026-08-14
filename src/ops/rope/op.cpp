#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/rope_musa.muh"
#endif

#include <cmath>
#include <vector>

namespace {

template <typename T>
void rope_cpu(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t n_heads,
    size_t head_dim,
    float theta) {

    size_t half_dim = head_dim / 2;

    // freq_denominator[j] = theta^(2j / head_dim)
    std::vector<float> freq_denominator(half_dim);

    for (size_t j = 0; j < half_dim; ++j) {
        float exponent =
            2.0f * static_cast<float>(j)
            / static_cast<float>(head_dim);

        freq_denominator[j] =
            std::pow(theta, exponent);
    }

    for (size_t s = 0; s < seq_len; ++s) {
        float position =
            static_cast<float>(pos_ids[s]);

        for (size_t j = 0; j < half_dim; ++j) {
            float angle =
                position / freq_denominator[j];

            float sin_val = std::sin(angle);
            float cos_val = std::cos(angle);

            for (size_t h = 0; h < n_heads; ++h) {
                size_t base =
                    (s * n_heads + h) * head_dim;

                size_t a_idx = base + j;
                size_t b_idx = base + half_dim + j;

                float a =
                    llaisys::utils::cast<float>(in[a_idx]);

                float b =
                    llaisys::utils::cast<float>(in[b_idx]);

                out[a_idx] =
                    llaisys::utils::cast<T>(
                        a * cos_val - b * sin_val);

                out[b_idx] =
                    llaisys::utils::cast<T>(
                        b * cos_val + a * sin_val);
            }
        }
    }
}

} // namespace

namespace llaisys::ops {

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    CHECK_ARGUMENT(
        in->ndim() == 3,
        "RoPE: input must be a 3D tensor.");

    CHECK_ARGUMENT(
        out->ndim() == 3,
        "RoPE: output must be a 3D tensor.");

    CHECK_ARGUMENT(
        pos_ids->ndim() == 1,
        "RoPE: pos_ids must be a 1D tensor.");

    CHECK_ARGUMENT(
        pos_ids->dtype() == LLAISYS_DTYPE_I64,
        "RoPE: pos_ids must have int64 dtype.");

    CHECK_ARGUMENT(
        pos_ids->numel() == in->shape()[0],
        "RoPE: number of position ids must match sequence length.");

    CHECK_ARGUMENT(
        in->shape()[2] > 0
            && in->shape()[2] % 2 == 0,
        "RoPE: head dimension must be positive and even.");

    CHECK_ARGUMENT(
        theta > 0.0f,
        "RoPE: theta must be positive.");

    ASSERT(
        out->isContiguous()
            && in->isContiguous()
            && pos_ids->isContiguous(),
        "RoPE: all tensors must be contiguous.");

    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        llaisys::core::context().setDevice(
            out->deviceType(), out->deviceId());

        switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::rope(
                out->data(),
                in->data(),
                pos_ids->data(),
                out->dtype(),
                seq_len,
                n_heads,
                head_dim,
                theta,
                llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_MUSA_API
        case LLAISYS_DEVICE_MUSA:
            return musa::rope(
                out->data(),
                in->data(),
                pos_ids->data(),
                out->dtype(),
                seq_len,
                n_heads,
                head_dim,
                theta,
                llaisys::core::context().runtime().stream());
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
    }

    const auto *positions =
        reinterpret_cast<const int64_t *>(pos_ids->data());

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rope_cpu(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(in->data()),
            positions,
            seq_len,
            n_heads,
            head_dim,
            theta);

    case LLAISYS_DTYPE_F16:
        return rope_cpu(
            reinterpret_cast<llaisys::fp16_t *>(out->data()),
            reinterpret_cast<const llaisys::fp16_t *>(in->data()),
            positions,
            seq_len,
            n_heads,
            head_dim,
            theta);

    case LLAISYS_DTYPE_BF16:
        return rope_cpu(
            reinterpret_cast<llaisys::bf16_t *>(out->data()),
            reinterpret_cast<const llaisys::bf16_t *>(in->data()),
            positions,
            seq_len,
            n_heads,
            head_dim,
            theta);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops
