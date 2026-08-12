#include "llaisys/models/qwen2.h"

#include "../models/qwen2/model.hpp"

#include <memory>

struct LlaisysQwen2Model {
    std::unique_ptr<llaisys::models::Qwen2Model> model;
};

__C {

LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta,
    llaisysDeviceType_t device,
    int *device_ids,
    int ndevice) {

    return new LlaisysQwen2Model{
        std::make_unique<llaisys::models::Qwen2Model>(
            *meta,
            device,
            device_ids[0])};
}

void llaisysQwen2ModelDestroy(
    LlaisysQwen2Model *model) {

    delete model;
}

LlaisysQwen2Weights *llaisysQwen2ModelWeights(
    LlaisysQwen2Model *model) {

    return model->model->weights();
}

int64_t llaisysQwen2ModelInfer(
    LlaisysQwen2Model *model,
    int64_t *token_ids,
    size_t ntoken) {

    return model->model->infer(
        token_ids,
        ntoken);
}

} // __C
