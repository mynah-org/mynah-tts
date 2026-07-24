#include <metal_stdlib>

using namespace metal;

struct MatmulParams {
    uint rows;
    uint input_width;
    uint output_width;
};

kernel void mynah_matmul(device const float *input [[buffer(0)]],
                         device const float *weight [[buffer(1)]],
                         device const float *bias [[buffer(2)]],
                         device float *output [[buffer(3)]],
                         constant MatmulParams &params [[buffer(4)]],
                         uint index [[thread_position_in_grid]]) {
    const uint total = params.rows * params.output_width;
    if (index >= total) return;
    const uint row = index / params.output_width;
    const uint column = index - row * params.output_width;
    const device float *input_row = input + row * params.input_width;
    const device float *weight_row = weight + column * params.input_width;
    float value = bias == nullptr ? 0.0f : bias[column];
    for (uint i = 0; i < params.input_width; ++i) value += input_row[i] * weight_row[i];
    output[index] = value;
}
