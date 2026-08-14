#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/rms_norm_musa.muh"
#endif

#include <cmath>

namespace {

template <typename T>
void rms_norm_cpu(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t hidden_size,
    float eps) {

    for (size_t i = 0; i < rows; ++i) {
        float sum_sq = 0.0f;

        // 1. Calculate mean(x^2) for this row.
        for (size_t j = 0; j < hidden_size; ++j) {
            float x =
                llaisys::utils::cast<float>(
                    in[i * hidden_size + j]);

            sum_sq += x * x;
        }

        float mean_sq =
            sum_sq / static_cast<float>(hidden_size);

        // 2. Calculate 1 / sqrt(mean(x^2) + eps).
        float inv_rms =
            1.0f / std::sqrt(mean_sq + eps);

        // 3. Normalize and apply weight.
        for (size_t j = 0; j < hidden_size; ++j) {
            float x =
                llaisys::utils::cast<float>(
                    in[i * hidden_size + j]);

            float w =
                llaisys::utils::cast<float>(
                    weight[j]);

            out[i * hidden_size + j] =
                llaisys::utils::cast<T>(
                    x * inv_rms * w);
        }
    }
}

} // namespace

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_SAME_SHAPE(out->shape(), in->shape());

    CHECK_ARGUMENT(
        in->ndim() == 2,
        "RMSNorm: input must be a 2D tensor.");

    CHECK_ARGUMENT(
        out->ndim() == 2,
        "RMSNorm: output must be a 2D tensor.");

    CHECK_ARGUMENT(
        weight->ndim() == 1,
        "RMSNorm: weight must be a 1D tensor.");

    CHECK_ARGUMENT(
        weight->numel() == in->shape()[1],
        "RMSNorm: weight size must match input hidden dimension.");

    CHECK_ARGUMENT(
        in->shape()[1] > 0,
        "RMSNorm: hidden dimension must be greater than zero.");

    ASSERT(
        out->isContiguous()
            && in->isContiguous()
            && weight->isContiguous(),
        "RMSNorm: all tensors must be contiguous.");

    size_t rows = in->shape()[0];
    size_t hidden_size = in->shape()[1];

    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        llaisys::core::context().setDevice(
            out->deviceType(), out->deviceId());

        switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::rms_norm(
                out->data(),
                in->data(),
                weight->data(),
                out->dtype(),
                rows,
                hidden_size,
                eps,
                llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_MUSA_API
        case LLAISYS_DEVICE_MUSA:
            return musa::rms_norm(
                out->data(),
                in->data(),
                weight->data(),
                out->dtype(),
                rows,
                hidden_size,
                eps,
                llaisys::core::context().runtime().stream());
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
    }

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_cpu(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(in->data()),
            reinterpret_cast<const float *>(weight->data()),
            rows,
            hidden_size,
            eps);

    case LLAISYS_DTYPE_F16:
        return rms_norm_cpu(
            reinterpret_cast<llaisys::fp16_t *>(out->data()),
            reinterpret_cast<const llaisys::fp16_t *>(in->data()),
            reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
            rows,
            hidden_size,
            eps);

    case LLAISYS_DTYPE_BF16:
        return rms_norm_cpu(
            reinterpret_cast<llaisys::bf16_t *>(out->data()),
            reinterpret_cast<const llaisys::bf16_t *>(in->data()),
            reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
            rows,
            hidden_size,
            eps);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops
