#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif
#ifdef ENABLE_MUSA_API
#include "musa/swiglu_musa.muh"
#endif

#include <cmath>

namespace {

template <typename T>
void swiglu_cpu(
    T *out,
    const T *gate,
    const T *up,
    size_t numel) {

    for (size_t i = 0; i < numel; ++i) {
        float gate_val =
            llaisys::utils::cast<float>(gate[i]);

        float up_val =
            llaisys::utils::cast<float>(up[i]);

        // PyTorch test computes exp in float32, then casts it
        // back to the output dtype before the remaining operations.
        T exp_t =
            llaisys::utils::cast<T>(
                std::exp(-gate_val));

        float exp_val =
            llaisys::utils::cast<float>(exp_t);

        float silu =
            gate_val / (1.0f + exp_val);

        out[i] =
            llaisys::utils::cast<T>(
                up_val * silu);
    }
}

} // namespace

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());

    CHECK_SAME_SHAPE(
        out->shape(),
        gate->shape(),
        up->shape());

    ASSERT(
        out->isContiguous()
            && gate->isContiguous()
            && up->isContiguous(),
        "SwiGLU: all tensors must be contiguous.");

    size_t numel = out->numel();

    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        llaisys::core::context().setDevice(
            out->deviceType(), out->deviceId());

        switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::swiglu(
                out->data(),
                gate->data(),
                up->data(),
                out->dtype(),
                numel,
                llaisys::core::context().runtime().stream());
#endif
#ifdef ENABLE_MUSA_API
        case LLAISYS_DEVICE_MUSA:
            return musa::swiglu(
                out->data(),
                gate->data(),
                up->data(),
                out->dtype(),
                numel,
                llaisys::core::context().runtime().stream());
#endif
        case LLAISYS_DEVICE_TYPE_COUNT:
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
    }

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return swiglu_cpu(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(gate->data()),
            reinterpret_cast<const float *>(up->data()),
            numel);

    case LLAISYS_DTYPE_F16:
        return swiglu_cpu(
            reinterpret_cast<llaisys::fp16_t *>(out->data()),
            reinterpret_cast<const llaisys::fp16_t *>(gate->data()),
            reinterpret_cast<const llaisys::fp16_t *>(up->data()),
            numel);

    case LLAISYS_DTYPE_BF16:
        return swiglu_cpu(
            reinterpret_cast<llaisys::bf16_t *>(out->data()),
            reinterpret_cast<const llaisys::bf16_t *>(gate->data()),
            reinterpret_cast<const llaisys::bf16_t *>(up->data()),
            numel);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops
