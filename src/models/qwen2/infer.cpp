#include "model.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <numeric>
#include <vector>

namespace {

llaisys::tensor_t unwrap(llaisysTensor_t tensor) {
    return tensor->tensor;
}

} // namespace

namespace llaisys::models {

int64_t Qwen2Model::infer(
    const int64_t *token_ids,
    size_t ntoken) {

    CHECK_ARGUMENT(
        token_ids != nullptr,
        "Qwen2: token_ids must not be null.");

    CHECK_ARGUMENT(
        ntoken > 0,
        "Qwen2: input sequence must not be empty.");

    CHECK_ARGUMENT(
        ntoken <= _meta.maxseq,
        "Qwen2: input sequence exceeds maximum length.");

    CHECK_ARGUMENT(
        _device == LLAISYS_DEVICE_CPU,
        "Qwen2: inference currently supports CPU only.");

    const size_t kv_dim =
        _meta.nkvh * _meta.dh;

    const float scale =
        1.0f / std::sqrt(
            static_cast<float>(_meta.dh));

    // ------------------------------------------------------------
    // Token ids and position ids.
    // ------------------------------------------------------------
    auto tokens = Tensor::create(
        {ntoken},
        LLAISYS_DTYPE_I64,
        _device,
        _device_id);

    tokens->load(token_ids);

    std::vector<int64_t> position_data(ntoken);
    std::iota(
        position_data.begin(),
        position_data.end(),
        int64_t{0});

    auto positions = Tensor::create(
        {ntoken},
        LLAISYS_DTYPE_I64,
        _device,
        _device_id);

    positions->load(position_data.data());

    // ------------------------------------------------------------
    // Input embedding.
    // hidden: [seq, hs]
    // ------------------------------------------------------------
    auto hidden = Tensor::create(
        {ntoken, _meta.hs},
        _meta.dtype,
        _device,
        _device_id);

    ops::embedding(
        hidden,
        tokens,
        unwrap(_weights.in_embed));

    // ------------------------------------------------------------
    // Decoder layers.
    // ------------------------------------------------------------
    for (size_t layer = 0;
         layer < _meta.nlayer;
         ++layer) {

        // =========================
        // Self-attention block
        // =========================

        // residual = hidden
        auto residual = hidden;

        auto attn_norm = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            _device,
            _device_id);

        ops::rms_norm(
            attn_norm,
            hidden,
            unwrap(_weights.attn_norm_w[layer]),
            _meta.epsilon);

        // Q: [seq, hs]
        auto q_2d = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            _device,
            _device_id);

        ops::linear(
            q_2d,
            attn_norm,
            unwrap(_weights.attn_q_w[layer]),
            unwrap(_weights.attn_q_b[layer]));

        // K: [seq, nkvh * dh]
        auto k_2d = Tensor::create(
            {ntoken, kv_dim},
            _meta.dtype,
            _device,
            _device_id);

        ops::linear(
            k_2d,
            attn_norm,
            unwrap(_weights.attn_k_w[layer]),
            unwrap(_weights.attn_k_b[layer]));

        // V: [seq, nkvh * dh]
        auto v_2d = Tensor::create(
            {ntoken, kv_dim},
            _meta.dtype,
            _device,
            _device_id);

        ops::linear(
            v_2d,
            attn_norm,
            unwrap(_weights.attn_v_w[layer]),
            unwrap(_weights.attn_v_b[layer]));

        // Views:
        // Q [seq, nh, dh]
        // K [seq, nkvh, dh]
        // V [seq, nkvh, dh]
        auto q = q_2d->view(
            {ntoken, _meta.nh, _meta.dh});

        auto k = k_2d->view(
            {ntoken, _meta.nkvh, _meta.dh});

        auto v = v_2d->view(
            {ntoken, _meta.nkvh, _meta.dh});

        // RoPE output must be contiguous, so allocate
        // separate output tensors rather than doing it in-place.
        auto q_rope = Tensor::create(
            {ntoken, _meta.nh, _meta.dh},
            _meta.dtype,
            _device,
            _device_id);

        auto k_rope = Tensor::create(
            {ntoken, _meta.nkvh, _meta.dh},
            _meta.dtype,
            _device,
            _device_id);

        ops::rope(
            q_rope,
            q,
            positions,
            _meta.theta);

        ops::rope(
            k_rope,
            k,
            positions,
            _meta.theta);

        // Full-sequence causal attention.
        // No KV cache in this first implementation.
        auto attn_3d = Tensor::create(
            {ntoken, _meta.nh, _meta.dh},
            _meta.dtype,
            _device,
            _device_id);

        ops::self_attention(
            attn_3d,
            q_rope,
            k_rope,
            v,
            scale);

        auto attn_2d = attn_3d->view(
            {ntoken, _meta.hs});

        auto attn_out = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            _device,
            _device_id);

        ops::linear(
            attn_out,
            attn_2d,
            unwrap(_weights.attn_o_w[layer]),
            nullptr);

        // hidden = residual + attention_output
        auto post_attn = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            _device,
            _device_id);

        ops::add(
            post_attn,
            residual,
            attn_out);

        // =========================
        // MLP block
        // =========================

        auto mlp_residual = post_attn;

        auto mlp_norm = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            _device,
            _device_id);

        ops::rms_norm(
            mlp_norm,
            post_attn,
            unwrap(_weights.mlp_norm_w[layer]),
            _meta.epsilon);

        auto gate = Tensor::create(
            {ntoken, _meta.di},
            _meta.dtype,
            _device,
            _device_id);

        auto up = Tensor::create(
            {ntoken, _meta.di},
            _meta.dtype,
            _device,
            _device_id);

        ops::linear(
            gate,
            mlp_norm,
            unwrap(_weights.mlp_gate_w[layer]),
            nullptr);

        ops::linear(
            up,
            mlp_norm,
            unwrap(_weights.mlp_up_w[layer]),
            nullptr);

        auto activated = Tensor::create(
            {ntoken, _meta.di},
            _meta.dtype,
            _device,
            _device_id);

        ops::swiglu(
            activated,
            gate,
            up);

        auto mlp_out = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            _device,
            _device_id);

        ops::linear(
            mlp_out,
            activated,
            unwrap(_weights.mlp_down_w[layer]),
            nullptr);

        hidden = Tensor::create(
            {ntoken, _meta.hs},
            _meta.dtype,
            _device,
            _device_id);

        ops::add(
            hidden,
            mlp_residual,
            mlp_out);
    }

    // ------------------------------------------------------------
    // We only need logits for the LAST token.
    //
    // Avoid allocating:
    //   [ntoken, vocab_size]
    //
    // Slice first, then final norm and lm_head.
    // ------------------------------------------------------------
    auto last_hidden = hidden->slice(
        0,
        ntoken - 1,
        ntoken);

    auto final_hidden = Tensor::create(
        {1, _meta.hs},
        _meta.dtype,
        _device,
        _device_id);

    ops::rms_norm(
        final_hidden,
        last_hidden,
        unwrap(_weights.out_norm_w),
        _meta.epsilon);

    auto logits_2d = Tensor::create(
        {1, _meta.voc},
        _meta.dtype,
        _device,
        _device_id);

    ops::linear(
        logits_2d,
        final_hidden,
        unwrap(_weights.out_embed),
        nullptr);

    auto logits = logits_2d->view(
        {_meta.voc});

    auto max_idx = Tensor::create(
        {1},
        LLAISYS_DTYPE_I64,
        _device,
        _device_id);

    auto max_val = Tensor::create(
        {1},
        _meta.dtype,
        _device,
        _device_id);

    ops::argmax(
        max_idx,
        max_val,
        logits);

    return *reinterpret_cast<int64_t *>(
        max_idx->data());
}

} // namespace llaisys::models
