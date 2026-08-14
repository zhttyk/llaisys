#include "../runtime_api.hpp"

#include <musa_runtime.h>

namespace llaisys::device::musa {

namespace runtime_api {

static void checkMusa(musaError_t err) {
    ASSERT(err == musaSuccess, musaGetErrorString(err));
}

static musaMemcpyKind toMusaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return musaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return musaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return musaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return musaMemcpyDeviceToDevice;
    default:
        ASSERT(false, "Unsupported memcpy kind.");
        return musaMemcpyDefault;
    }
}

int getDeviceCount() {
    int count = 0;
    checkMusa(musaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    checkMusa(musaSetDevice(device));
}

void deviceSynchronize() {
    checkMusa(musaDeviceSynchronize());
}

llaisysStream_t createStream() {
    musaStream_t stream = nullptr;
    checkMusa(musaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    checkMusa(
        musaStreamDestroy(
            reinterpret_cast<musaStream_t>(stream)));
}

void streamSynchronize(llaisysStream_t stream) {
    checkMusa(
        musaStreamSynchronize(
            reinterpret_cast<musaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkMusa(musaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    checkMusa(musaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    checkMusa(musaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    checkMusa(musaFreeHost(ptr));
}

void memcpySync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind) {

    checkMusa(
        musaMemcpy(
            dst,
            src,
            size,
            toMusaMemcpyKind(kind)));
}

void memcpyAsync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream) {

    checkMusa(
        musaMemcpyAsync(
            dst,
            src,
            size,
            toMusaMemcpyKind(kind),
            reinterpret_cast<musaStream_t>(stream)));
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::musa
