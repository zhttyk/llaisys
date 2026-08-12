#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

namespace {

template <typename T>
void linear_cpu(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t m,
    size_t n,
    size_t k) {

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {

            float sum = 0.0f;

            if (bias != nullptr) {
                sum = llaisys::utils::cast<float>(bias[j]);
            }

            for (size_t p = 0; p < k; ++p) {
                float x =
                    llaisys::utils::cast<float>(in[i * k + p]);

                float w =
                    llaisys::utils::cast<float>(weight[j * k + p]);

                sum += x * w;
            }

            out[i * n + j] =
                llaisys::utils::cast<T>(sum);
        }
    }
}

} // namespace

namespace llaisys::ops {

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());

    CHECK_ARGUMENT(
        out->ndim() == 2,
        "Linear: out must be a 2D tensor.");

    CHECK_ARGUMENT(
        in->ndim() == 2,
        "Linear: input must be a 2D tensor.");

    CHECK_ARGUMENT(
        weight->ndim() == 2,
        "Linear: weight must be a 2D tensor.");

    CHECK_ARGUMENT(
        in->shape()[1] == weight->shape()[1],
        "Linear: input and weight feature dimensions must match.");

    CHECK_ARGUMENT(
        out->shape()[0] == in->shape()[0],
        "Linear: output rows must match input rows.");

    CHECK_ARGUMENT(
        out->shape()[1] == weight->shape()[0],
        "Linear: output columns must match weight rows.");

    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());

        CHECK_ARGUMENT(
            bias->ndim() == 1,
            "Linear: bias must be a 1D tensor.");

        CHECK_ARGUMENT(
            bias->numel() == weight->shape()[0],
            "Linear: bias size must match output features.");

        ASSERT(
            bias->isContiguous(),
            "Linear: bias must be contiguous.");
    }

    ASSERT(
        out->isContiguous()
            && in->isContiguous()
            && weight->isContiguous(),
        "Linear: out, input and weight must be contiguous.");

    size_t m = in->shape()[0];
    size_t k = in->shape()[1];
    size_t n = weight->shape()[0];

    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        llaisys::core::context().setDevice(
            out->deviceType(), out->deviceId());

        switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
        case LLAISYS_DEVICE_NVIDIA:
            return nvidia::linear(
                out->data(),
                in->data(),
                weight->data(),
                bias ? bias->data() : nullptr,
                out->dtype(),
                m,
                n,
                k,
                llaisys::core::context().runtime().stream());
#endif
        default:
            EXCEPTION_UNSUPPORTED_DEVICE;
        }
    }

    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        return linear_cpu(
            reinterpret_cast<float *>(out->data()),
            reinterpret_cast<const float *>(in->data()),
            reinterpret_cast<const float *>(weight->data()),
            bias
                ? reinterpret_cast<const float *>(bias->data())
                : nullptr,
            m,
            n,
            k);

    case LLAISYS_DTYPE_F16:
        return linear_cpu(
            reinterpret_cast<llaisys::fp16_t *>(out->data()),
            reinterpret_cast<const llaisys::fp16_t *>(in->data()),
            reinterpret_cast<const llaisys::fp16_t *>(weight->data()),
            bias
                ? reinterpret_cast<const llaisys::fp16_t *>(bias->data())
                : nullptr,
            m,
            n,
            k);

    case LLAISYS_DTYPE_BF16:
        return linear_cpu(
            reinterpret_cast<llaisys::bf16_t *>(out->data()),
            reinterpret_cast<const llaisys::bf16_t *>(in->data()),
            reinterpret_cast<const llaisys::bf16_t *>(weight->data()),
            bias
                ? reinterpret_cast<const llaisys::bf16_t *>(bias->data())
                : nullptr,
            m,
            n,
            k);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
    }
}

} // namespace llaisys::ops
