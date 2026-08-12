#include "../runtime_api.hpp"

#include <cuda_runtime.h>

namespace llaisys::device::nvidia {

namespace runtime_api {

static void checkCuda(cudaError_t err) {
    ASSERT(err == cudaSuccess, cudaGetErrorString(err));
}

static cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        ASSERT(false, "Unsupported memcpy kind.");
        return cudaMemcpyDefault;
    }
}

int getDeviceCount() {
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count));
    return count;
}

void setDevice(int device) {
    checkCuda(cudaSetDevice(device));
}

void deviceSynchronize() {
    checkCuda(cudaDeviceSynchronize());
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream));
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    cudaError_t err =
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));

    // During process shutdown another CUDA user (for example PyTorch)
    // may unload the CUDA runtime before our thread-local Context is
    // destroyed. At that point stream cleanup can no longer be executed.
    if (err == cudaErrorCudartUnloading) {
        return;
    }

    checkCuda(err);
}

void streamSynchronize(llaisysStream_t stream) {
    checkCuda(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, size));
    return ptr;
}

void freeDevice(void *ptr) {
    checkCuda(cudaFree(ptr));
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    checkCuda(cudaMallocHost(&ptr, size));
    return ptr;
}

void freeHost(void *ptr) {
    checkCuda(cudaFreeHost(ptr));
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    checkCuda(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
}

void memcpyAsync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream) {

    checkCuda(cudaMemcpyAsync(
        dst,
        src,
        size,
        toCudaMemcpyKind(kind),
        reinterpret_cast<cudaStream_t>(stream)));
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

} // namespace llaisys::device::nvidia
