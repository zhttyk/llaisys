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
#include <algorithm>
#include <cstring>

namespace {

llaisys::tensor_t unwrap(llaisysTensor_t tensor) {
    return tensor->tensor;
}

} // namespace

namespace llaisys::models {

void Qwen2Model::resetCache() {
    _k_cache.clear();
    _v_cache.clear();

    _k_cache.resize(_meta.nlayer);
    _v_cache.resize(_meta.nlayer);

    _cache_len = 0;
    _cache_capacity = 0;
    _token_history.clear();
}

void Qwen2Model::ensureCacheCapacity(size_t required) {
    if (required <= _cache_capacity) {
        return;
    }

    size_t new_capacity =
        _cache_capacity == 0
            ? required
            : _cache_capacity;

    while (new_capacity < required) {
        new_capacity *= 2;

        if (new_capacity > _meta.maxseq) {
            new_capacity = _meta.maxseq;
            break;
        }
    }

    CHECK_ARGUMENT(
        new_capacity >= required,
        "Qwen2: unable to grow KV cache.");

    const size_t cached_bytes =
        _cache_len
        * _meta.nkvh
        * _meta.dh
        * utils::dsize(_meta.dtype);

    for (size_t layer = 0;
         layer < _meta.nlayer;
         ++layer) {

        auto new_k = Tensor::create(
            {new_capacity, _meta.nkvh, _meta.dh},
            _meta.dtype,
            _device,
            _device_id);

        auto new_v = Tensor::create(
            {new_capacity, _meta.nkvh, _meta.dh},
            _meta.dtype,
            _device,
            _device_id);

        if (_cache_len > 0) {
            std::memcpy(
                new_k->data(),
                _k_cache[layer]->data(),
                cached_bytes);

            std::memcpy(
                new_v->data(),
                _v_cache[layer]->data(),
                cached_bytes);
        }

        _k_cache[layer] = std::move(new_k);
        _v_cache[layer] = std::move(new_v);
    }

    _cache_capacity = new_capacity;
}

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

    bool decode =
        _cache_len > 0
        && _cache_len == _token_history.size()
        && ntoken == _token_history.size() + 1
        && std::equal(
            _token_history.begin(),
            _token_history.end(),
            token_ids);

    if (!decode) {
        resetCache();
    }

    const size_t qlen =
        decode ? 1 : ntoken;

    const size_t start_pos =
        decode ? _cache_len : 0;

    const size_t new_cache_len =
        start_pos + qlen;

    const int64_t *current_tokens =
        decode
            ? token_ids + ntoken - 1
            : token_ids;

    ensureCacheCapacity(new_cache_len);

    const size_t kv_dim =
        _meta.nkvh * _meta.dh;

    const float scale =
        1.0f / std::sqrt(
            static_cast<float>(_meta.dh));

    // ------------------------------------------------------------
    // Token ids and position ids.
    // ------------------------------------------------------------
    auto tokens = Tensor::create(
        {qlen},
        LLAISYS_DTYPE_I64,
        _device,
        _device_id);

    tokens->load(current_tokens);

    std::vector<int64_t> position_data(qlen);

    for (size_t i = 0; i < qlen; ++i) {
        position_data[i] =
            static_cast<int64_t>(start_pos + i);
    }

    auto positions = Tensor::create(
        {qlen},
        LLAISYS_DTYPE_I64,
        _device,
        _device_id);

    positions->load(position_data.data());

    // ------------------------------------------------------------
    // Input embedding.
    // hidden: [seq, hs]
    // ------------------------------------------------------------
    auto hidden = Tensor::create(
        {qlen, _meta.hs},
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
            {qlen, _meta.hs},
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
            {qlen, _meta.hs},
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
            {qlen, kv_dim},
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
            {qlen, kv_dim},
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
            {qlen, _meta.nh, _meta.dh});

        auto k = k_2d->view(
            {qlen, _meta.nkvh, _meta.dh});

        auto v = v_2d->view(
            {qlen, _meta.nkvh, _meta.dh});

        // RoPE output must be contiguous, so allocate
        // separate output tensors rather than doing it in-place.
        auto q_rope = Tensor::create(
            {qlen, _meta.nh, _meta.dh},
            _meta.dtype,
            _device,
            _device_id);

        auto k_rope = Tensor::create(
            {qlen, _meta.nkvh, _meta.dh},
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

        // Causal attention over the current query and cached K/V.
        auto attn_3d = Tensor::create(
            {qlen, _meta.nh, _meta.dh},
            _meta.dtype,
            _device,
            _device_id);

        const size_t cache_write_bytes =
            qlen
            * _meta.nkvh
            * _meta.dh
            * utils::dsize(_meta.dtype);

        auto k_dst = _k_cache[layer]->slice(
            0,
            start_pos,
            new_cache_len);

        auto v_dst = _v_cache[layer]->slice(
            0,
            start_pos,
            new_cache_len);

        std::memcpy(
            k_dst->data(),
            k_rope->data(),
            cache_write_bytes);

        std::memcpy(
            v_dst->data(),
            v->data(),
            cache_write_bytes);

        auto k_all = _k_cache[layer]->slice(
            0,
            0,
            new_cache_len);

        auto v_all = _v_cache[layer]->slice(
            0,
            0,
            new_cache_len);

        ops::self_attention(
            attn_3d,
            q_rope,
            k_all,
            v_all,
            scale);

        auto attn_2d = attn_3d->view(
            {qlen, _meta.hs});

        auto attn_out = Tensor::create(
            {qlen, _meta.hs},
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
            {qlen, _meta.hs},
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
            {qlen, _meta.hs},
            _meta.dtype,
            _device,
            _device_id);

        ops::rms_norm(
            mlp_norm,
            post_attn,
            unwrap(_weights.mlp_norm_w[layer]),
            _meta.epsilon);

        auto gate = Tensor::create(
            {qlen, _meta.di},
            _meta.dtype,
            _device,
            _device_id);

        auto up = Tensor::create(
            {qlen, _meta.di},
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
            {qlen, _meta.di},
            _meta.dtype,
            _device,
            _device_id);

        ops::swiglu(
            activated,
            gate,
            up);

        auto mlp_out = Tensor::create(
            {qlen, _meta.hs},
            _meta.dtype,
            _device,
            _device_id);

        ops::linear(
            mlp_out,
            activated,
            unwrap(_weights.mlp_down_w[layer]),
            nullptr);

        hidden = Tensor::create(
            {qlen, _meta.hs},
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
    //   [qlen, vocab_size]
    //
    // Slice first, then final norm and lm_head.
    // ------------------------------------------------------------
    auto last_hidden = hidden->slice(
        0,
        qlen - 1,
        qlen);

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

    int64_t next_token =
        *reinterpret_cast<int64_t *>(
            max_idx->data());

    _cache_len = new_cache_len;

    _token_history.assign(
        token_ids,
        token_ids + ntoken);

    return next_token;
}

} // namespace llaisys::models
