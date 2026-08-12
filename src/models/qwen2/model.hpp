#pragma once

#include "llaisys/models/qwen2.h"

#include "../../llaisys/llaisys_tensor.hpp"

#include <vector>

namespace llaisys::models {

class Qwen2Model {
public:
    Qwen2Model(
        const LlaisysQwen2Meta &meta,
        llaisysDeviceType_t device,
        int device_id);

    ~Qwen2Model();

    LlaisysQwen2Weights *weights();

    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;

    LlaisysQwen2Weights _weights{};

    std::vector<llaisysTensor_t> _attn_norm_w;

    std::vector<llaisysTensor_t> _attn_q_w;
    std::vector<llaisysTensor_t> _attn_q_b;

    std::vector<llaisysTensor_t> _attn_k_w;
    std::vector<llaisysTensor_t> _attn_k_b;

    std::vector<llaisysTensor_t> _attn_v_w;
    std::vector<llaisysTensor_t> _attn_v_b;

    std::vector<llaisysTensor_t> _attn_o_w;

    std::vector<llaisysTensor_t> _mlp_norm_w;
    std::vector<llaisysTensor_t> _mlp_gate_w;
    std::vector<llaisysTensor_t> _mlp_up_w;
    std::vector<llaisysTensor_t> _mlp_down_w;

    llaisysTensor_t createTensor(const std::vector<size_t> &shape);
    void destroyTensor(llaisysTensor_t tensor);
    // Dynamic KV cache.
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;

    size_t _cache_len = 0;
    size_t _cache_capacity = 0;

    std::vector<int64_t> _token_history;

    void resetCache();
    void ensureCacheCapacity(size_t required);
};

} // namespace llaisys::models
