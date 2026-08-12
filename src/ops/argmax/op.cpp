#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif

namespace {

template <typename T>
void argmax_cpu(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    size_t best_idx = 0;
    float best_val = llaisys::utils::cast<float>(vals[0]);

    for (size_t i = 1; i < numel; ++i) {
        float val = llaisys::utils::cast<float>(vals[i]);

        if (val > best_val) {
            best_val = val;
            best_idx = i;
        }
    }

    max_idx[0] = static_cast<int64_t>(best_idx);
    max_val[0] = llaisys::utils::cast<T>(best_val);
}

} // namespace

namespace llaisys::ops {

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);

    CHECK_ARGUMENT(vals->ndim() == 1, "Argmax: vals must be a 1D tensor.");
    CHECK_ARGUMENT(vals->numel() > 0, "Argmax: vals must not be empty.");

    CHECK_ARGUMENT(
        max_idx->numel() == 1,
        "Argmax: max_idx must contain exactly one element.");

    CHECK_ARGUMENT(
        max_val->numel() == 1,
        "Argmax: max_val must contain exactly one element.");

    CHECK_ARGUMENT(
        max_idx->dtype() == LLAISYS_DTYPE_I64,
        "Argmax: max_idx must have int64 dtype.");

    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());

    ASSERT(
        max_idx->isContiguous()
            && max_val->isContiguous()
            && vals->isContiguous(),
        "Argmax: all tensors must be contiguous.");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        auto *idx = reinterpret_cast<int64_t *>(max_idx->data());

        switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            return argmax_cpu(
                idx,
                reinterpret_cast<float *>(max_val->data()),
                reinterpret_cast<const float *>(vals->data()),
                vals->numel());

        case LLAISYS_DTYPE_F16:
            return argmax_cpu(
                idx,
                reinterpret_cast<llaisys::fp16_t *>(max_val->data()),
                reinterpret_cast<const llaisys::fp16_t *>(vals->data()),
                vals->numel());

        case LLAISYS_DTYPE_BF16:
            return argmax_cpu(
                idx,
                reinterpret_cast<llaisys::bf16_t *>(max_val->data()),
                reinterpret_cast<const llaisys::bf16_t *>(vals->data()),
                vals->numel());

        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel(),
            llaisys::core::context().runtime().stream());
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
