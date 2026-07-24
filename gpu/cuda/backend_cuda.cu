#include "backend.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

struct cuda_cached_buffer {
    const void *host_pointer;
    size_t bytes;
    float *device_pointer;
};

struct cuda_backend_state {
    cudaStream_t stream;
    std::vector<cuda_cached_buffer> weights;
};

struct matmul_params {
    uint32_t rows;
    uint32_t input_width;
    uint32_t output_width;
};

__global__ static void mynah_matmul_kernel(const float *input, const float *weight,
                                           const float *bias, float *output,
                                           matmul_params params) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = params.rows * params.output_width;
    if (index >= total) return;
    const uint32_t row = index / params.output_width;
    const uint32_t column = index - row * params.output_width;
    const float *input_row = input + row * params.input_width;
    const float *weight_row = weight + column * params.input_width;
    float value = bias == nullptr ? 0.0f : bias[column];
    for (uint32_t i = 0; i < params.input_width; ++i) value += input_row[i] * weight_row[i];
    output[index] = value;
}

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != nullptr && capacity > 0) std::snprintf(error, capacity, "%s", message);
}

static int cuda_error(cudaError_t result, char *error, size_t error_capacity) {
    if (result == cudaSuccess) return 0;
    std::snprintf(error, error_capacity, "CUDA error: %s", cudaGetErrorString(result));
    return -1;
}

static int cached_weight(cuda_backend_state *state, const float *host_pointer, size_t bytes,
                         float **device_pointer, char *error, size_t error_capacity) {
    for (const cuda_cached_buffer &cached : state->weights) {
        if (cached.host_pointer == host_pointer && cached.bytes == bytes) {
            *device_pointer = cached.device_pointer;
            return 0;
        }
    }
    float *device = nullptr;
    if (cuda_error(cudaMalloc(&device, bytes), error, error_capacity) != 0 ||
        cuda_error(cudaMemcpyAsync(device, host_pointer, bytes, cudaMemcpyHostToDevice,
                                   state->stream), error, error_capacity) != 0) {
        cudaFree(device);
        return -1;
    }
    state->weights.push_back({host_pointer, bytes, device});
    *device_pointer = device;
    return 0;
}

static int cuda_matmul(void *opaque, const float *input, float *output, size_t rows,
                       size_t input_width, size_t output_width, const float *weight,
                       const float *bias, char *error, size_t error_capacity) {
    if (rows > UINT32_MAX || input_width > UINT32_MAX || output_width > UINT32_MAX ||
        rows > SIZE_MAX / output_width || input_width > SIZE_MAX / output_width ||
        rows > UINT32_MAX / output_width) {
        set_error(error, error_capacity, "CUDA matmul dimensions are too large");
        return -1;
    }
    cuda_backend_state *state = static_cast<cuda_backend_state *>(opaque);
    float *device_input = nullptr;
    float *device_weight = nullptr;
    float *device_bias = nullptr;
    float *device_output = nullptr;
    const size_t input_bytes = rows * input_width * sizeof(float);
    const size_t weight_bytes = input_width * output_width * sizeof(float);
    const size_t output_bytes = rows * output_width * sizeof(float);
    const size_t bias_bytes = output_width * sizeof(float);
    int result = 0;
    if (cuda_error(cudaMalloc(&device_input, input_bytes), error, error_capacity) != 0 ||
        cached_weight(state, weight, weight_bytes, &device_weight, error, error_capacity) != 0 ||
        (bias != nullptr && cached_weight(state, bias, bias_bytes, &device_bias, error,
                                          error_capacity) != 0) ||
        cuda_error(cudaMalloc(&device_output, output_bytes), error, error_capacity) != 0 ||
        cuda_error(cudaMemcpyAsync(device_input, input, input_bytes, cudaMemcpyHostToDevice,
                                   state->stream), error, error_capacity) != 0 ||
        (bias != nullptr && cuda_error(cudaMemcpyAsync(device_bias, bias, bias_bytes,
                                                        cudaMemcpyHostToDevice, state->stream),
                                       error, error_capacity) != 0)) {
        result = -1;
        goto cleanup;
    }
    {
        const matmul_params params = {(uint32_t)rows, (uint32_t)input_width,
                                      (uint32_t)output_width};
        const uint32_t total = params.rows * params.output_width;
        mynah_matmul_kernel<<<(total + 255u) / 256u, 256u, 0, state->stream>>>(
            device_input, device_weight, device_bias, device_output, params);
        if (cuda_error(cudaGetLastError(), error, error_capacity) != 0 ||
            cuda_error(cudaMemcpyAsync(output, device_output, output_bytes,
                                       cudaMemcpyDeviceToHost, state->stream),
                       error, error_capacity) != 0 ||
            cuda_error(cudaStreamSynchronize(state->stream), error, error_capacity) != 0) {
            result = -1;
        }
    }
cleanup:
    cudaFree(device_input);
    cudaFree(device_output);
    return result;
}

static int cuda_self_test(void *opaque, char *error, size_t error_capacity) {
    const float input[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 2.0f};
    const float weight[12] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float bias[4] = {0.5f, -0.5f, 1.0f, 2.0f};
    const float expected[8] = {1.5f, 1.5f, 4.0f, 8.0f, -0.5f, 0.0f, 3.0f, 3.5f};
    float output[8] = {0};
    if (cuda_matmul(opaque, input, output, 2u, 3u, 4u, weight, bias,
                    error, error_capacity) != 0) return -1;
    for (size_t i = 0; i < 8u; ++i) {
        if (std::fabs(output[i] - expected[i]) > 1.0e-5f) {
            std::snprintf(error, error_capacity, "CUDA backend self-test mismatch at %zu", i);
            return -1;
        }
    }
    return 0;
}

static void cuda_close(void *opaque) {
    cuda_backend_state *state = static_cast<cuda_backend_state *>(opaque);
    if (state == nullptr) return;
    for (const cuda_cached_buffer &cached : state->weights) cudaFree(cached.device_pointer);
    cudaStreamDestroy(state->stream);
    delete state;
}

extern "C" int mynah_backend_cuda_open(void **state_out, mynah_backend_matmul_fn *matmul,
                                        mynah_backend_close_fn *close,
                                        mynah_backend_self_test_fn *self_test,
                                        char *error, size_t error_capacity) {
    int device_count = 0;
    if (cuda_error(cudaGetDeviceCount(&device_count), error, error_capacity) != 0 ||
        device_count == 0) {
        if (device_count == 0) set_error(error, error_capacity, "CUDA device is unavailable");
        return -1;
    }
    if (cuda_error(cudaSetDevice(0), error, error_capacity) != 0) return -1;
    cuda_backend_state *state = new (std::nothrow) cuda_backend_state();
    if (state == nullptr) {
        set_error(error, error_capacity, "out of memory creating CUDA backend");
        return -1;
    }
    if (cuda_error(cudaStreamCreate(&state->stream), error, error_capacity) != 0) {
        delete state;
        return -1;
    }
    *state_out = state;
    *matmul = cuda_matmul;
    *close = cuda_close;
    *self_test = cuda_self_test;
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return 0;
}
