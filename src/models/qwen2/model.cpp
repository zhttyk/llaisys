#include "model.hpp"

#include "../../utils.hpp"

namespace llaisys::models {

llaisysTensor_t Qwen2Model::createTensor(
    const std::vector<size_t> &shape) {

    return new LlaisysTensor{
        llaisys::Tensor::create(
            shape,
            _meta.dtype,
            _device,
            _device_id)};
}

void Qwen2Model::destroyTensor(llaisysTensor_t tensor) {
    delete tensor;
}

Qwen2Model::Qwen2Model(
    const LlaisysQwen2Meta &meta,
    llaisysDeviceType_t device,
    int device_id)
    : _meta(meta),
      _device(device),
      _device_id(device_id) {

    CHECK_ARGUMENT(
        _meta.nlayer > 0,
        "Qwen2: number of layers must be positive.");

    CHECK_ARGUMENT(
        _meta.hs > 0,
        "Qwen2: hidden size must be positive.");

    CHECK_ARGUMENT(
        _meta.nh > 0 && _meta.nkvh > 0,
        "Qwen2: head counts must be positive.");

    CHECK_ARGUMENT(
        _meta.nh * _meta.dh == _meta.hs,
        "Qwen2: nh * dh must equal hidden size.");

    CHECK_ARGUMENT(
        _meta.nh % _meta.nkvh == 0,
        "Qwen2: query heads must be divisible by KV heads.");

    size_t kv_dim = _meta.nkvh * _meta.dh;

    // Global weights.
    _weights.in_embed =
        createTensor({_meta.voc, _meta.hs});

    _weights.out_embed =
        createTensor({_meta.voc, _meta.hs});

    _weights.out_norm_w =
        createTensor({_meta.hs});

    // Per-layer handle arrays.
    _attn_norm_w.resize(_meta.nlayer);

    _attn_q_w.resize(_meta.nlayer);
    _attn_q_b.resize(_meta.nlayer);

    _attn_k_w.resize(_meta.nlayer);
    _attn_k_b.resize(_meta.nlayer);

    _attn_v_w.resize(_meta.nlayer);
    _attn_v_b.resize(_meta.nlayer);

    _attn_o_w.resize(_meta.nlayer);

    _mlp_norm_w.resize(_meta.nlayer);
    _mlp_gate_w.resize(_meta.nlayer);
    _mlp_up_w.resize(_meta.nlayer);
    _mlp_down_w.resize(_meta.nlayer);

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        _attn_norm_w[i] =
            createTensor({_meta.hs});

        _attn_q_w[i] =
            createTensor({_meta.hs, _meta.hs});

        _attn_q_b[i] =
            createTensor({_meta.hs});

        _attn_k_w[i] =
            createTensor({kv_dim, _meta.hs});

        _attn_k_b[i] =
            createTensor({kv_dim});

        _attn_v_w[i] =
            createTensor({kv_dim, _meta.hs});

        _attn_v_b[i] =
            createTensor({kv_dim});

        _attn_o_w[i] =
            createTensor({_meta.hs, _meta.hs});

        _mlp_norm_w[i] =
            createTensor({_meta.hs});

        _mlp_gate_w[i] =
            createTensor({_meta.di, _meta.hs});

        _mlp_up_w[i] =
            createTensor({_meta.di, _meta.hs});

        _mlp_down_w[i] =
            createTensor({_meta.hs, _meta.di});
    }

    // Expose vectors through the public C structure.
    _weights.attn_norm_w = _attn_norm_w.data();

    _weights.attn_q_w = _attn_q_w.data();
    _weights.attn_q_b = _attn_q_b.data();

    _weights.attn_k_w = _attn_k_w.data();
    _weights.attn_k_b = _attn_k_b.data();

    _weights.attn_v_w = _attn_v_w.data();
    _weights.attn_v_b = _attn_v_b.data();

    _weights.attn_o_w = _attn_o_w.data();

    _weights.mlp_norm_w = _mlp_norm_w.data();
    _weights.mlp_gate_w = _mlp_gate_w.data();
    _weights.mlp_up_w = _mlp_up_w.data();
    _weights.mlp_down_w = _mlp_down_w.data();
}

Qwen2Model::~Qwen2Model() {
    destroyTensor(_weights.in_embed);
    destroyTensor(_weights.out_embed);
    destroyTensor(_weights.out_norm_w);

    for (size_t i = 0; i < _meta.nlayer; ++i) {
        destroyTensor(_attn_norm_w[i]);

        destroyTensor(_attn_q_w[i]);
        destroyTensor(_attn_q_b[i]);

        destroyTensor(_attn_k_w[i]);
        destroyTensor(_attn_k_b[i]);

        destroyTensor(_attn_v_w[i]);
        destroyTensor(_attn_v_b[i]);

        destroyTensor(_attn_o_w[i]);

        destroyTensor(_mlp_norm_w[i]);
        destroyTensor(_mlp_gate_w[i]);
        destroyTensor(_mlp_up_w[i]);
        destroyTensor(_mlp_down_w[i]);
    }
}

LlaisysQwen2Weights *Qwen2Model::weights() {
    return &_weights;
}

} // namespace llaisys::models