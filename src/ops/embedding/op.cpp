#include "op.hpp"

#include "../../utils.hpp"

#include <cstring>

namespace llaisys::ops {

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    CHECK_ARGUMENT(
        index->ndim() == 1,
        "Embedding: index must be a 1D tensor.");

    CHECK_ARGUMENT(
        weight->ndim() == 2,
        "Embedding: weight must be a 2D tensor.");

    CHECK_ARGUMENT(
        out->ndim() == 2,
        "Embedding: out must be a 2D tensor.");

    CHECK_ARGUMENT(
        index->dtype() == LLAISYS_DTYPE_I64,
        "Embedding: index must have int64 dtype.");

    CHECK_ARGUMENT(
        out->shape()[0] == index->shape()[0],
        "Embedding: out first dimension must match index length.");

    CHECK_ARGUMENT(
        out->shape()[1] == weight->shape()[1],
        "Embedding: out second dimension must match embedding dimension.");

    ASSERT(
        out->isContiguous()
            && index->isContiguous()
            && weight->isContiguous(),
        "Embedding: all tensors must be contiguous.");

    if (out->deviceType() != LLAISYS_DEVICE_CPU) {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }

    const auto *indices =
        reinterpret_cast<const int64_t *>(index->data());

    auto *out_data = out->data();
    const auto *weight_data = weight->data();

    size_t num_indices = index->numel();
    size_t vocab_size = weight->shape()[0];
    size_t embedding_dim = weight->shape()[1];

    size_t row_bytes =
        embedding_dim * weight->elementSize();

    for (size_t i = 0; i < num_indices; ++i) {
        int64_t idx = indices[i];

        CHECK_ARGUMENT(
            idx >= 0 && static_cast<size_t>(idx) < vocab_size,
            "Embedding: index out of range.");

        std::memcpy(
            out_data + i * row_bytes,
            weight_data + static_cast<size_t>(idx) * row_bytes,
            row_bytes);
    }
}

} // namespace llaisys::ops
