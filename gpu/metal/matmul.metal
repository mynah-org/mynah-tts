#include <metal_stdlib>

using namespace metal;

struct MatmulParams {
    uint rows;
    uint input_width;
    uint output_width;
};

/* A cooperative 4x64 tile.  Decoder projections reuse the same input row
 * across many output columns, so loading input and weights through threadgroup
 * memory avoids repeatedly fetching them from device memory. */
constant constexpr uint TILE_M = 4;
constant constexpr uint TILE_N = 64;
constant constexpr uint TILE_K = 32;

/* Qwen-style simdgroup matvec: one output row per simdgroup, coalesced float4
 * reads and a hardware subgroup reduction.  This is the critical B=1 decode
 * path; the tiled kernel below is retained for comparison and fallback. */
kernel void mynah_matmul_row_simd(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant MatmulParams &p [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
    const uint row = tgid.x * nsg + sgid;
    if (row >= p.output_width) return;
    float acc = 0.0f;
    const uint n4 = p.input_width / 4u;
    device const float4 *input4 = (device const float4 *)input;
    device const float4 *weight4 =
        (device const float4 *)(weight + (ulong)row * p.input_width);
    for (uint c = lane; c < n4; c += 32u)
        acc += dot(weight4[c], input4[c]);
    for (uint k = n4 * 4u + lane; k < p.input_width; k += 32u)
        acc += input[k] * weight[(ulong)row * p.input_width + k];
    acc = simd_sum(acc);
    if (lane == 0)
        output[row] = acc + (bias == nullptr ? 0.0f : bias[row]);
}

/* A decoder step is a single row.  Use one 64-thread group per output tile:
 * the generic 4x64 kernel launches four row groups and does unnecessary input
 * loads/barriers when rows == 1. */
kernel void mynah_matmul_row_tiled(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant MatmulParams &p [[buffer(4)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint col = gid.x * 64u + tid;
    threadgroup float input_tile[TILE_K];
    float acc = 0.0f;
    for (uint k0 = 0; k0 < p.input_width; k0 += TILE_K) {
        if (tid < TILE_K)
            input_tile[tid] = k0 + tid < p.input_width ? input[k0 + tid] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (col < p.output_width) {
            const uint base = col * p.input_width + k0;
            for (uint k = 0; k < TILE_K && k0 + k < p.input_width; ++k)
                acc += input_tile[k] * weight[base + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (col < p.output_width)
        output[col] = acc + (bias == nullptr ? 0.0f : bias[col]);
}

kernel void mynah_matmul_tiled(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant MatmulParams &p [[buffer(4)]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]) {
    const uint row_start = gid.y * TILE_M;
    const uint col_start = gid.x * TILE_N;
    const uint tx = lid.x;
    const uint ty = lid.y;
    const uint col = col_start + tx;
    const uint row = row_start + ty;
    const uint linear = ty * TILE_N + tx;

    threadgroup float input_tile[TILE_M * TILE_K];
    threadgroup float weight_tile[TILE_N * TILE_K];
    float acc = 0.0f;

    for (uint k0 = 0; k0 < p.input_width; k0 += TILE_K) {
        for (uint i = linear; i < TILE_M * TILE_K; i += TILE_M * TILE_N) {
            const uint tile_row = i / TILE_K;
            const uint tile_k = i - tile_row * TILE_K;
            const uint src_row = row_start + tile_row;
            const uint src_k = k0 + tile_k;
            input_tile[i] = (src_row < p.rows && src_k < p.input_width)
                ? input[src_row * p.input_width + src_k] : 0.0f;
        }
        for (uint i = linear; i < TILE_N * TILE_K; i += TILE_M * TILE_N) {
            const uint tile_col = i / TILE_K;
            const uint tile_k = i - tile_col * TILE_K;
            const uint src_col = col_start + tile_col;
            const uint src_k = k0 + tile_k;
            weight_tile[i] = (src_col < p.output_width && src_k < p.input_width)
                ? weight[src_col * p.input_width + src_k] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE_K; ++k)
            acc += input_tile[ty * TILE_K + k] * weight_tile[tx * TILE_K + k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < p.rows && col < p.output_width) {
        output[row * p.output_width + col] =
            acc + (bias == nullptr ? 0.0f : bias[col]);
    }
}

kernel void mynah_matmul(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant MatmulParams &p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint total = p.rows * p.output_width;
    if (index >= total) return;
    const uint row = index / p.output_width;
    const uint column = index - row * p.output_width;
    const device float *input_row = input + row * p.input_width;
    const device float *weight_row = weight + column * p.input_width;
    float value = bias == nullptr ? 0.0f : bias[column];
    for (uint i = 0; i < p.input_width; ++i)
        value += input_row[i] * weight_row[i];
    output[index] = value;
}
