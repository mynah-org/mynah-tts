#include <metal_stdlib>

using namespace metal;

struct MatmulParams {
    uint rows;
    uint input_width;
    uint output_width;
};

/* Tiled matmul: each threadgroup computes a TILE_M x TILE_N output tile.
 * Threads cooperatively load input and weight tiles into threadgroup memory,
 * then accumulate the dot products.  For the small Magpie matvecs (rows=1,
 * output_width up to 3072) this reduces global memory traffic and keeps the
 * SIMD groups fed. */
constant constexpr uint TILE_M = 4;
constant constexpr uint TILE_N = 64;
constant constexpr uint TILE_K = 32;

kernel void mynah_matmul_tiled(device const float *input   [[buffer(0)]],
                               device const float *weight  [[buffer(1)]],
                               device const float *bias    [[buffer(2)]],
                               device float       *output  [[buffer(3)]],
                               constant MatmulParams &p    [[buffer(4)]],
                               uint2 gid [[threadgroup_position_in_grid]],
                               uint2 lid [[thread_position_in_threadgroup]],
                               uint2 tgs [[threads_per_threadgroup]]) {
    const uint row_start = gid.y * TILE_M;
    const uint col_start = gid.x * TILE_N;
    const uint tx = lid.x;   /* 0..TILE_N-1 */
    const uint ty = lid.y;   /* 0..TILE_M-1 */
    const uint col = col_start + tx;
    const uint row = row_start + ty;

    float acc = 0.0f;
    for (uint k0 = 0; k0 < p.input_width; k0 += TILE_K) {
        const uint k_end = min(k0 + TILE_K, p.input_width);
        for (uint k = k0; k < k_end; ++k) {
            if (row < p.rows && col < p.output_width) {
                acc += input[row * p.input_width + k] *
                       weight[col * p.input_width + k];
            }
        }
    }
    if (row < p.rows && col < p.output_width) {
        float value = acc;
        if (bias != nullptr) value += bias[col];
        output[row * p.output_width + col] = value;
    }
}

/* Simple fallback kept for self-test compatibility. */
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
