#include "backend.h"
#include "costmap.h"
#include "row_cap.h"

#include <cublas_v2.h>
#include <cublasLt.h>
/* Driver API TYPES only (MYNAH_CUDA_KV_VMM): the functions are resolved at
 * run time through cudaGetDriverEntryPoint, so nothing links libcuda and a
 * host without a driver still loads the binary. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <new>
#include <unordered_map>
#include <vector>

/*
 * CUDA backend — resident GPU inference.
 *
 * Weights are uploaded once and cached.  Activations live in a device-side
 * scratch buffer; element-wise ops (layer-norm, softmax, GELU, snake,
 * residual-add) run as tiny CUDA kernels so the AR loop never round-trips
 * through host memory.  The only H2D/D2H copies are:
 *   - input embedding at the start of each step
 *   - logits download for sampling
 *
 * On GB10 Grace Blackwell (unified memory) even those copies are page-table
 * remaps; on discrete GPUs they are pinned DMA.
 */

/* ------------------------------------------------------------------ */
/*  CUDA kernels                                                       */
/* ------------------------------------------------------------------ */

/* The first CUDA slice used one thread per norm row and accidentally ignored
 * the optional bias.  One block per row keeps the operation resident while
 * reducing the work in parallel.  The reduction order is fixed (warp tree),
 * so CPU/GPU parity only needs the documented floating-point tolerance. */
__global__ static void k_layer_norm(float *out, const float *in,
                                    const float *gain, const float *bias,
                                    int width, float eps, int nrows) {
    const int row = (int)blockIdx.x;
    if (row >= nrows) return;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float warp_sum[8];
    __shared__ float warp_sq[8];
    __shared__ float mean_shared;
    __shared__ float inv_shared;
    const float *x = in + (size_t)row * (size_t)width;
    float *y = out + (size_t)row * (size_t)width;
    float sum = 0.0f;
    for (int d = tid; d < width; d += (int)blockDim.x) sum += x[d];
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) warp_sum[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        const int warps = ((int)blockDim.x + 31) / 32;
        for (int w = 0; w < warps; ++w) total += warp_sum[w];
        mean_shared = total / (float)width;
    }
    __syncthreads();
    const float mean = mean_shared;
    float sq = 0.0f;
    for (int d = tid; d < width; d += (int)blockDim.x) {
        const float delta = x[d] - mean;
        sq += delta * delta;
    }
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_down_sync(0xffffffffu, sq, off);
    if (lane == 0) warp_sq[warp] = sq;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        const int warps = ((int)blockDim.x + 31) / 32;
        for (int w = 0; w < warps; ++w) total += warp_sq[w];
        inv_shared = rsqrtf(total / (float)width + eps);
    }
    __syncthreads();
    for (int d = tid; d < width; d += (int)blockDim.x) {
        const float b = bias == nullptr ? 0.0f : bias[d];
        const float g = gain == nullptr ? 1.0f : gain[d];
        y[d] = (x[d] - mean) * inv_shared * g + b;
    }
}

__device__ static float warp_max(float value) {
    for (int off = 16; off > 0; off >>= 1)
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, off));
    return value;
}

__device__ static float warp_sum(float value) {
    for (int off = 16; off > 0; off >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, off);
    return value;
}

__global__ static void k_softmax_causal(float *data, int cols, int valid) {
    const int row = (int)blockIdx.x;
    float *r = data + (size_t)row * (size_t)cols;
    const int v = valid < cols ? valid : cols;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float partial[8];
    __shared__ float maximum;
    __shared__ float denominator;
    float local_max = -1.0e30f;
    for (int i = tid; i < v; i += (int)blockDim.x) local_max = fmaxf(local_max, r[i]);
    local_max = warp_max(local_max);
    if (lane == 0) partial[warp] = local_max;
    __syncthreads();
    if (tid == 0) {
        maximum = -1.0e30f;
        const int warps = ((int)blockDim.x + 31) / 32;
        for (int w = 0; w < warps; ++w) maximum = fmaxf(maximum, partial[w]);
    }
    __syncthreads();
    float local_sum = 0.0f;
    for (int i = tid; i < v; i += (int)blockDim.x) {
        r[i] = expf(r[i] - maximum);
        local_sum += r[i];
    }
    local_sum = warp_sum(local_sum);
    if (lane == 0) partial[warp] = local_sum;
    __syncthreads();
    if (tid == 0) {
        denominator = 0.0f;
        const int warps = ((int)blockDim.x + 31) / 32;
        for (int w = 0; w < warps; ++w) denominator += partial[w];
    }
    __syncthreads();
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int i = tid; i < v; i += (int)blockDim.x) r[i] *= inv;
    for (int i = tid + v; i < cols; i += (int)blockDim.x) r[i] = 0.0f;
}

__global__ static void k_gelu(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        float c = 0.7978845608f * (x + 0.044715f * x * x * x);
        data[i] = 0.5f * x * (1.0f + tanhf(c));
    }
}

__global__ static void k_silu(float *data, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float x = data[i];
        data[i] = x / (1.0f + expf(-x));
    }
}

/* Flow-head variance RMSNorm: the variance is mean-subtracted and unbiased,
 * matching torch.var() in Pocket's timestep embedder. */
__global__ static void k_flow_var_rms_norm(const float *in, const float *alpha,
                                           float *out, int rows, int width,
                                           float epsilon) {
    const int row = (int)blockIdx.x;
    if (row >= rows) return;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float partial_sum[8];
    __shared__ float partial_sq[8];
    __shared__ float mean_shared;
    __shared__ float inv_shared;
    const float *x = in + (size_t)row * (size_t)width;
    float *y = out + (size_t)row * (size_t)width;
    float sum = 0.0f;
    for (int d = tid; d < width; d += (int)blockDim.x) sum += x[d];
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) partial_sum[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        const int warps = ((int)blockDim.x + 31) / 32;
        float total = 0.0f;
        for (int w = 0; w < warps; ++w) total += partial_sum[w];
        mean_shared = total / (float)width;
    }
    __syncthreads();
    float sq = 0.0f;
    for (int d = tid; d < width; d += (int)blockDim.x) {
        const float delta = x[d] - mean_shared;
        sq += delta * delta;
    }
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_down_sync(0xffffffffu, sq, off);
    if (lane == 0) partial_sq[warp] = sq;
    __syncthreads();
    if (tid == 0) {
        const int warps = ((int)blockDim.x + 31) / 32;
        float total = 0.0f;
        for (int w = 0; w < warps; ++w) total += partial_sq[w];
        inv_shared = rsqrtf(total / (float)(width - 1) + epsilon);
    }
    __syncthreads();
    for (int d = tid; d < width; d += (int)blockDim.x) {
        const float gain = alpha == nullptr ? 1.0f : alpha[d];
        y[d] = x[d] * gain * inv_shared;
    }
}

__global__ static void k_flow_modulate(float *data, const float *mod,
                                       int rows, int width, int mod_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * width;
    if (i >= total) return;
    const int row = i / width;
    const int col = i - row * width;
    const float *m = mod + (size_t)row * (size_t)mod_stride;
    data[i] = data[i] * (1.0f + m[col + width]) + m[col];
}

__global__ static void k_flow_gate_add(float *out, const float *gate,
                                       const float *update, int rows, int width,
                                       int gate_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * width;
    if (i >= total) return;
    const int row = i / width;
    const int col = i - row * width;
    out[i] += gate[(size_t)row * (size_t)gate_stride + col] * update[i];
}

__global__ static void k_snake(float *data, const float *alpha,
                               int channels, int length, int snake_ch) {
    int ch = blockIdx.x;
    if (ch >= channels) return;
    float *row = data + (size_t)ch * length;
    if (ch < snake_ch) {
        float a = alpha[ch];
        float inv_a = 1.0f / (a + 1e-9f);
        for (int t = threadIdx.x; t < length; t += blockDim.x) {
            float v = row[t];
            float sn = sinf(a * v);
            row[t] = v + sn * sn * inv_a;
        }
    } else {
        for (int t = threadIdx.x; t < length; t += blockDim.x)
            if (row[t] < 0.0f) row[t] *= 0.01f;
    }
}

__global__ static void k_residual_add(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += in[i];
}

__global__ static void k_scaled_residual_add(float *out, const float *in,
                                             const float *scale, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] += scale[i] * in[i];
}

__global__ static void k_scaled_residual_rows_add(float *out, const float *in,
                                                  const float *scale, int rows,
                                                  int width) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * width;
    if (index < total) {
        out[index] += scale[index % width] * in[index];
    }
}

__global__ static void k_bias_add(float *out, const float *bias, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[i] += bias[i % cols];
}

__global__ static void k_f32_to_f16(const float *in, half *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

__global__ static void k_f16_to_f32(const half *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

/* BF16 is used only for persistent attention K/V.  Keep conversion explicit
 * instead of relying on a device-specific intrinsic: this compiles for the
 * target range from older CUDA toolkits through Ada/Blackwell and preserves
 * the same round-to-nearest-even rule on every GPU. */
__device__ static uint16_t cuda_bf16_from_float(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t rounded = bits + UINT32_C(0x7fff) + ((bits >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

__device__ static float cuda_bf16_to_float(uint16_t value) {
    return __uint_as_float((uint32_t)value << 16);
}

__global__ static void k_f32_to_bf16(const float *in, uint16_t *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cuda_bf16_from_float(in[i]);
}

__global__ static void k_bf16_to_f32(const uint16_t *in, float *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cuda_bf16_to_float(in[i]);
}

/* ---- Fused BF16 decode linears (MYNAH_CUDA_BF16_FUSE) ----
 *
 * Each kernel below replaces a pair or a triple of the unfused BF16 path
 * (k_layer_norm + k_f32_to_bf16, k_bias_add + k_gelu + k_f32_to_bf16,
 * k_bias_add + k_rope_qk_batch, k_bias_add + k_residual_add) and computes the
 * same FP32 expressions in the same order, so the BF16 operands and the FP32
 * results are bit-identical to the unfused path: the FP32 intermediate that
 * used to round-trip through global memory now stays in a register, which is
 * the same value (no extended precision on the GPU, and no expression here
 * gives the compiler a new multiply-add to contract). */

/* k_layer_norm with the BF16 store of the cast kernel folded in.  The body
 * is a line-for-line copy of k_layer_norm up to the store. */
__global__ static void k_layer_norm_bf16(uint16_t *out, const float *in,
                                         const float *gain, const float *bias,
                                         int width, float eps, int nrows) {
    const int row = (int)blockIdx.x;
    if (row >= nrows) return;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float warp_sum[8];
    __shared__ float warp_sq[8];
    __shared__ float mean_shared;
    __shared__ float inv_shared;
    const float *x = in + (size_t)row * (size_t)width;
    uint16_t *y = out + (size_t)row * (size_t)width;
    float sum = 0.0f;
    for (int d = tid; d < width; d += (int)blockDim.x) sum += x[d];
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, off);
    if (lane == 0) warp_sum[warp] = sum;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        const int warps = ((int)blockDim.x + 31) / 32;
        for (int w = 0; w < warps; ++w) total += warp_sum[w];
        mean_shared = total / (float)width;
    }
    __syncthreads();
    const float mean = mean_shared;
    float sq = 0.0f;
    for (int d = tid; d < width; d += (int)blockDim.x) {
        const float delta = x[d] - mean;
        sq += delta * delta;
    }
    for (int off = 16; off > 0; off >>= 1)
        sq += __shfl_down_sync(0xffffffffu, sq, off);
    if (lane == 0) warp_sq[warp] = sq;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        const int warps = ((int)blockDim.x + 31) / 32;
        for (int w = 0; w < warps; ++w) total += warp_sq[w];
        inv_shared = rsqrtf(total / (float)width + eps);
    }
    __syncthreads();
    for (int d = tid; d < width; d += (int)blockDim.x) {
        const float b = bias == nullptr ? 0.0f : bias[d];
        const float g = gain == nullptr ? 1.0f : gain[d];
        const float value = (x[d] - mean) * inv_shared * g + b;
        y[d] = cuda_bf16_from_float(value);
    }
}

/* k_bias_add + k_gelu + k_f32_to_bf16: `in` is the bias-free GEMM output and
 * is not written. */
__global__ static void k_bias_gelu_bf16(const float *in, const float *bias,
                                        uint16_t *out, int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        const float x = bias == nullptr ? in[i] : in[i] + bias[i % cols];
        const float c = 0.7978845608f * (x + 0.044715f * x * x * x);
        const float value = 0.5f * x * (1.0f + tanhf(c));
        out[i] = cuda_bf16_from_float(value);
    }
}

/* k_bias_add on the GEMM output followed by k_residual_add into `out`. */
__global__ static void k_residual_bias_add(float *out, const float *in,
                                           const float *bias, int rows,
                                           int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        const float y = bias == nullptr ? in[i] : in[i] + bias[i % cols];
        out[i] += y;
    }
}

/* Element store for the SEANet GEMM operand builders (im2col columns and the
 * transposed-convolution gather).  The float overload is the plain store the
 * FP32 decoder has always done; the uint16_t overload rounds to BF16 (RNE, the
 * same helper as the BF16 KV cache) for MYNAH_CUDA_SEANET_BF16.  Only GEMM
 * operands go through the BF16 store: causal states, bias, residual and the
 * GEMM output stay FP32. */
__device__ __forceinline__ static void decoder_put(float *dst, size_t i,
                                                   float value) {
    dst[i] = value;
}

__device__ __forceinline__ static void decoder_put(uint16_t *dst, size_t i,
                                                   float value) {
    dst[i] = cuda_bf16_from_float(value);
}

/* Q8 activation packing used by the resident Pocket linear path.  The CPU
 * qmat reference uses symmetric per-row absmax, ties-away-from-zero rounding
 * and the signed range [-127, 127].  Keep those choices explicit here: CUDA's
 * default round-to-nearest-even would otherwise make the AR trajectory depend
 * on the backend at exact half-integers. */
__global__ static void k_q8_quantize_rows(const float *in, int8_t *out,
                                          float *scales, int rows, int cols) {
    const int row = (int)blockIdx.x;
    if (row >= rows) return;
    const int tid = (int)threadIdx.x;
    __shared__ float maxima[256];
    const float *src = in + (size_t)row * (size_t)cols;
    float amax = 0.0f;
    for (int col = tid; col < cols; col += (int)blockDim.x) {
        const float value = fabsf(src[col]);
        if (value > amax) amax = value;
    }
    maxima[tid] = amax;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && maxima[tid + stride] > maxima[tid])
            maxima[tid] = maxima[tid + stride];
        __syncthreads();
    }
    const float scale = maxima[0] == 0.0f ? 0.0f : maxima[0] / 127.0f;
    if (tid == 0) scales[row] = scale;
    const float inv = scale == 0.0f ? 0.0f : 127.0f / scale;
    for (int col = tid; col < cols; col += (int)blockDim.x) {
        const float value = src[col] * inv;
        /* Truncation toward zero after adding/subtracting 0.5 is ties-away. */
        int q = __float2int_rz(value >= 0.0f ? value + 0.5f : value - 0.5f);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out[(size_t)row * (size_t)cols + (size_t)col] = (int8_t)q;
    }
}

__global__ static void k_q8_epilogue(const int32_t *accum, const float *act_scale,
                                     const float *weight_scale, const float *bias,
                                     float *out, int rows, int cols) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = rows * cols;
    if (index >= total) return;
    const int row = index / cols;
    const int col = index - row * cols;
    const float row_scale = act_scale[row] * weight_scale[col];
    out[index] = fmaf((float)accum[index], row_scale,
                      bias == nullptr ? 0.0f : bias[col]);
}

__global__ static void k_copy_strided(float *dst, const float *src,
                                      int dst_stride, int src_stride,
                                      int width, int rows) {
    int r = blockIdx.x;
    if (r >= rows) return;
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        dst[(size_t)r * dst_stride + c] = src[(size_t)r * src_stride + c];
}

__global__ static void k_copy(float *dst, const float *src, int n) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n) dst[i] = src[i];
}

__global__ static void k_scale(float *data, float scale, int n) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n) data[i] *= scale;
}

__global__ static void k_clip(float *data, int n) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n) return;
    const float value = data[i];
    data[i] = isfinite(value) ? fminf(1.0f, fmaxf(-1.0f, value)) : 0.0f;
}

__global__ static void k_decoder_elu(const float *input, float *output,
                                     float alpha, int n) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n) return;
    const float x = input[i];
    output[i] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
}

/* Build the causal window used by one resident decoder convolution. The CPU
 * reference carries the last `tail` values of each input channel; the first
 * call sees zero padding for every Pocket decoder convolution. */
__global__ static void k_decoder_causal_window(const float *previous,
                                               const float *input, float *window,
                                               int channels, int length, int tail) {
    const int window_len = tail + length;
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = channels * window_len;
    if (index >= total) return;
    const int channel = index / window_len;
    const int pos = index - channel * window_len;
    if (pos < tail) {
        window[index] = previous[(size_t)channel * (size_t)tail + (size_t)pos];
    } else {
        window[index] = input[(size_t)channel * (size_t)length +
                              (size_t)(pos - tail)];
    }
}

template <typename T>
__global__ static void k_decoder_causal_columns(const float *window,
                                                T *columns, int channels,
                                                int length, int kernel,
                                                int dilation, int stride,
                                                int tail) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = channels * kernel * length;
    if (index >= total) return;
    const int out_pos = index % length;
    const int tap_channel = index / length;
    const int tap = tap_channel % kernel;
    const int channel = tap_channel / kernel;
    const int window_len = tail + length;
    /* `window` already starts with the carried left context.  The CPU
     * reference indexes that combined buffer at `out_pos * stride`, so adding
     * `tail` here would skip the causal padding and shift every convolution
     * into the future. */
    const int source = out_pos * stride + tap * dilation;
    decoder_put(columns, (size_t)index, source >= 0 && source < window_len
        ? window[(size_t)channel * (size_t)window_len + (size_t)source]
        : 0.0f);
}

__global__ static void k_decoder_copy_tail(const float *window, float *previous,
                                           int channels, int length, int tail) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = channels * tail;
    if (index >= total) return;
    const int channel = index / tail;
    const int pos = index - channel * tail;
    const int window_len = tail + length;
    previous[index] = window[(size_t)channel * (size_t)window_len +
                             (size_t)(window_len - tail + pos)];
}

__global__ static void k_decoder_convtr_fold_save(float *full, float *partial,
                                                  const float *bias,
                                                  int channels, int full_len,
                                                  int tail) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = channels * tail;
    if (index >= total) return;
    const int channel = index / tail;
    const int pos = index - channel * tail;
    float *row = full + (size_t)channel * (size_t)full_len;
    row[pos] += partial[index];
    partial[index] = row[full_len - tail + pos] -
                     (bias == nullptr ? 0.0f : bias[channel]);
}

__global__ static void k_decoder_copy_prefix(const float *full, float *output,
                                             int channels, int full_len,
                                             int output_len) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = channels * output_len;
    if (index >= total) return;
    const int channel = index / output_len;
    const int pos = index - channel * output_len;
    output[index] = full[(size_t)channel * (size_t)full_len + (size_t)pos];
}

/* The per-request decoder state is independent, but its elementwise and
 * causal convolution work has the same topology for every request in a
 * scheduler gang. These kernels keep the request pointer tables on device
 * and give one launch to the whole gang. They deliberately preserve the
 * scalar decoder's channel-major layout and accumulation order. */
__global__ static void k_decoder_elu_batch(float *const *inputs,
                                           float *const *outputs,
                                           int batch, int n, float alpha) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = batch * n;
    if (index >= total) return;
    const int request = index / n;
    const int offset = index - request * n;
    const float x = inputs[request][offset];
    outputs[request][offset] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
}

__global__ static void k_decoder_residual_batch(float *const *base,
                                                float *const *add, int batch,
                                                int n) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = batch * n;
    if (index >= total) return;
    const int request = index / n;
    const int offset = index - request * n;
    base[request][offset] += add[request][offset];
}

__global__ static void k_decoder_bias_batch(float *const *outputs,
                                            const float *bias, int batch,
                                            int channels, int length) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = channels * length;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int offset = index - request * per_request;
    outputs[request][offset] = bias == nullptr ? 0.0f : bias[offset / length];
}

__global__ static void k_decoder_causal_window_batch(
    float *const *previous, float *const *inputs, float *const *windows,
    int batch, int channels, int length, int tail) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int window_len = tail + length;
    const int per_request = channels * window_len;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int channel = local / window_len;
    const int pos = local - channel * window_len;
    windows[request][local] = pos < tail
        ? previous[request][(size_t)channel * (size_t)tail + (size_t)pos]
        : inputs[request][(size_t)channel * (size_t)length +
                          (size_t)(pos - tail)];
}

template <typename T>
__global__ static void k_decoder_causal_columns_batch(
    float *const *windows, T *const *columns, int batch, int channels,
    int length, int kernel, int dilation, int stride, int tail) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = channels * kernel * length;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int out_pos = local % length;
    const int tap_channel = local / length;
    const int tap = tap_channel % kernel;
    const int channel = tap_channel / kernel;
    const int window_len = tail + length;
    const int source = out_pos * stride + tap * dilation;
    decoder_put(columns[request], (size_t)local,
                source >= 0 && source < window_len
                    ? windows[request][(size_t)channel * (size_t)window_len +
                                       (size_t)source]
                    : 0.0f);
}

/* One-GEMM variant: every request's im2col columns side by side in one
 * [inner][batch * length] buffer, so a single GEMM reads the weights once for
 * the whole gang instead of once per request. */
template <typename T>
__global__ static void k_decoder_causal_columns_shared(
    float *const *windows, T *columns, int batch, int channels,
    int length, int kernel, int dilation, int stride, int tail) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = channels * kernel * length;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int out_pos = local % length;
    const int tap_channel = local / length;
    const int tap = tap_channel % kernel;
    const int channel = tap_channel / kernel;
    const int window_len = tail + length;
    const int source = out_pos * stride + tap * dilation;
    decoder_put(columns,
                (size_t)tap_channel * (size_t)batch * length +
                    (size_t)request * length + out_pos,
                source >= 0 && source < window_len
                    ? windows[request][(size_t)channel * (size_t)window_len +
                                       (size_t)source]
                    : 0.0f);
}

/* [channels][batch * length] back to each request's [channels][length],
 * adding the bias. */
__global__ static void k_decoder_scatter_bias(const float *shared,
                                              float *const *outputs,
                                              const float *bias, int batch,
                                              int channels, int length) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = channels * length;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int offset = index - request * per_request;
    const int channel = offset / length;
    const int t = offset - channel * length;
    outputs[request][offset] =
        shared[(size_t)channel * (size_t)batch * length +
               (size_t)request * length + t] +
        (bias == nullptr ? 0.0f : bias[channel]);
}

__global__ static void k_decoder_copy_tail_batch(float *const *windows,
                                                 float *const *previous,
                                                 int batch, int channels,
                                                 int length, int tail) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = channels * tail;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int channel = local / tail;
    const int pos = local - channel * tail;
    const int window_len = tail + length;
    previous[request][local] = windows[request][
        (size_t)channel * (size_t)window_len + (size_t)(window_len - tail + pos)];
}

/* MYNAH_CUDA_DECODER_FUSE: the same values as k_decoder_elu_batch followed
 * by k_decoder_causal_window_batch and k_decoder_causal_columns_batch, read
 * straight from the carried state and the input (ELU applied on the fly to
 * input values only; the carried state already holds post-ELU values), so the
 * ELU output and the window buffer are never written. Stride 1 only. */
__device__ __forceinline__ static float decoder_elu_value(float x, float alpha) {
    return x > 0.0f ? x : alpha * (expf(x) - 1.0f);
}

/* One input value of a fused decoder op, resolved on the fly.  The producer
 * may have left work for its reader (MYNAH_CUDA_DECODER_FUSE):
 *   bias_on: the producing GEMM ran with beta = 0, so `input` holds the bare
 *            accumulator and the bias is added here, (acc + bias) in fp32,
 *            the one rounding the beta = 1 epilogue on a bias-filled C did
 *            (alpha = beta = 1, so every epilogue form is round(acc + C)).
 *            A null bias adds +0.0f, as k_decoder_bias_batch wrote 0.0f.
 *   resid:   the residual-block sum was not written: the value is
 *            resid + (acc + bias), the order k_decoder_residual_batch used
 *            (base += add, with add the finished 1x1 output).
 *   elu:     the reader's pre-activation, last, as before. */
__device__ __forceinline__ static float decoder_lazy_value(
    float *const *inputs, float *const *resid, const float *bias, int bias_on,
    int request, int channel, size_t offset, int elu, float alpha) {
    float value = inputs[request][offset];
    if (bias_on) value = value + (bias == nullptr ? 0.0f : bias[channel]);
    if (resid != nullptr) value = resid[request][offset] + value;
    return elu ? decoder_elu_value(value, alpha) : value;
}

template <typename T>
__global__ static void k_decoder_causal_columns_fused(
    float *const *previous, float *const *inputs, T *const *columns,
    int batch, int channels, int length, int kernel, int dilation, int tail,
    int elu, float alpha, float *const *resid, const float *bias,
    int bias_on) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = channels * kernel * length;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int out_pos = local % length;
    const int tap_channel = local / length;
    const int tap = tap_channel % kernel;
    const int channel = tap_channel / kernel;
    const int source = out_pos + tap * dilation;
    float value = 0.0f;
    if (source < tail) {
        value = previous[request][(size_t)channel * (size_t)tail + (size_t)source];
    } else if (source < tail + length) {
        value = decoder_lazy_value(
            inputs, resid, bias, bias_on, request, channel,
            (size_t)channel * (size_t)length + (size_t)(source - tail), elu,
            alpha);
    }
    decoder_put(columns[request], (size_t)local, value);
}

/* The new carried state when tail <= length: the last `tail` input values of
 * each channel (post-ELU), the same numbers k_decoder_copy_tail_batch takes
 * from the window. Runs after the columns kernel has read the old state. */
__global__ static void k_decoder_copy_tail_fused(float *const *inputs,
                                                 float *const *previous,
                                                 int batch, int channels,
                                                 int length, int tail, int elu,
                                                 float alpha,
                                                 float *const *resid,
                                                 const float *bias,
                                                 int bias_on) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = channels * tail;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int channel = local / tail;
    const int pos = local - channel * tail;
    previous[request][local] = decoder_lazy_value(
        inputs, resid, bias, bias_on, request, channel,
        (size_t)channel * (size_t)length + (size_t)(length - tail + pos), elu,
        alpha);
}

/* MYNAH_CUDA_DECODER_FUSE: k_decoder_residual_batch for a 1x1 output whose
 * GEMM ran with beta = 0: base = base + (add + bias), the same two fp32
 * additions in the same order as the bias-filled GEMM plus the residual. */
__global__ static void k_decoder_residual_bias_batch(float *const *base,
                                                     float *const *add,
                                                     const float *bias,
                                                     int batch, int channels,
                                                     int length) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = channels * length;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int offset = index - request * per_request;
    const float value =
        add[request][offset] + (bias == nullptr ? 0.0f : bias[offset / length]);
    base[request][offset] = base[request][offset] + value;
}

__global__ static void k_decoder_convtr_batch(
    float *const *inputs, float *const *full, const float *weight,
    const float *bias, int batch, int in_channels, int out_channels,
    int length, int full_len, int kernel, int stride, int groups) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = out_channels * full_len;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int t = local % full_len;
    const int output_channel = local / full_len;
    const int in_per_group = in_channels / groups;
    const int out_per_group = out_channels / groups;
    const int group = output_channel / out_per_group;
    const int out_local = output_channel % out_per_group;
    float value = bias == nullptr ? 0.0f : bias[output_channel];
    for (int k = 0; k < kernel; ++k) {
        if (t < k || ((t - k) % stride) != 0) continue;
        const int input_t = (t - k) / stride;
        if (input_t >= length) continue;
        for (int input_local = 0; input_local < in_per_group; ++input_local) {
            const int input_channel = group * in_per_group + input_local;
            const size_t weight_index =
                ((size_t)input_channel * (size_t)out_per_group +
                 (size_t)out_local) * (size_t)kernel + (size_t)k;
            value += inputs[request][(size_t)input_channel * (size_t)length +
                                    (size_t)input_t] * weight[weight_index];
        }
    }
    full[request][local] = value;
}

/* GEMM form of the batched transposed convolution (groups == 1): gather
 * every request's input into one [in][batch * length] matrix, one GEMM makes
 * Y[(out, tap)][batch * length], and this fold adds the taps back onto time. */
template <typename T>
__global__ static void k_decoder_convtr_gather(float *const *inputs, T *x,
                                               int batch, int channels,
                                               int length, int elu = 0,
                                               float alpha = 0.0f,
                                               float *const *resid = nullptr,
                                               const float *bias = nullptr,
                                               int bias_on = 0) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int n = batch * length;
    if (index >= channels * n) return;
    const int channel = index / n;
    const int j = index - channel * n;
    const int request = j / length;
    const int t = j - request * length;
    decoder_put(x, (size_t)index,
                decoder_lazy_value(inputs, resid, bias, bias_on, request,
                                   channel, (size_t)channel * length + t, elu,
                                   alpha));
}

__global__ static void k_decoder_convtr_overlap(const float *y, float *const *full,
                                                const float *bias, int batch,
                                                int out_channels, int length,
                                                int full_len, int kernel,
                                                int stride) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = out_channels * full_len;
    if (index >= batch * per_request) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int oc = local / full_len;
    const int t = local - oc * full_len;
    const size_t m = (size_t)out_channels * kernel;
    float value = bias == nullptr ? 0.0f : bias[oc];
    for (int k = t % stride; k < kernel && k <= t; k += stride) {
        const int i = (t - k) / stride;
        if (i >= length) continue;
        value += y[(size_t)oc * kernel + k + (size_t)(request * length + i) * m];
    }
    full[request][local] = value;
}

/* MYNAH_CUDA_DECODER_FUSE: k_decoder_convtr_overlap, k_decoder_convtr_fold_batch
 * and k_decoder_copy_prefix_batch in one pass that never writes `full`.  The
 * thread of output sample t sums its taps exactly as the overlap kernel does
 * (bias first, taps in the same order); for t < tail it also adds the carried
 * partial (the fold's `row[pos] += partial`) and, being the only reader and
 * writer of partial[t], sums the tail sample output_len + t the same way and
 * stores it minus the bias as the next partial (the fold's second line).
 * Needs tail <= output_len, so no fold write lands on a sample the fold reads
 * (true for every SEANet transposed conv: tail = stride <= length * stride). */
__device__ __forceinline__ static float decoder_convtr_tap_sum(
    const float *y, const float *bias, int request, int oc, int t,
    int out_channels, int length, int kernel, int stride) {
    const size_t m = (size_t)out_channels * kernel;
    float value = bias == nullptr ? 0.0f : bias[oc];
    for (int k = t % stride; k < kernel && k <= t; k += stride) {
        const int i = (t - k) / stride;
        if (i >= length) continue;
        value += y[(size_t)oc * kernel + k + (size_t)(request * length + i) * m];
    }
    return value;
}

__global__ static void k_decoder_convtr_overlap_out(
    const float *y, float *const *outputs, float *const *partial,
    const float *bias, int batch, int out_channels, int length, int output_len,
    int kernel, int stride, int tail) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = out_channels * output_len;
    if (index >= batch * per_request) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int oc = local / output_len;
    const int t = local - oc * output_len;
    float value = decoder_convtr_tap_sum(y, bias, request, oc, t, out_channels,
                                         length, kernel, stride);
    if (t < tail) {
        float *state = partial[request] + (size_t)oc * (size_t)tail;
        value += state[t];
        const float next = decoder_convtr_tap_sum(
            y, bias, request, oc, output_len + t, out_channels, length, kernel,
            stride);
        state[t] = next - (bias == nullptr ? 0.0f : bias[oc]);
    }
    outputs[request][local] = value;
}

/* Single-request form of k_decoder_convtr_overlap (batch == 1, no pointer
 * table), used only by the BF16 SEANet path so a solo decode runs the same
 * GEMM arithmetic as the gang. */
__global__ static void k_decoder_convtr_overlap_one(const float *y, float *full,
                                                    const float *bias,
                                                    int out_channels, int length,
                                                    int full_len, int kernel,
                                                    int stride) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (index >= out_channels * full_len) return;
    const int oc = index / full_len;
    const int t = index - oc * full_len;
    const size_t m = (size_t)out_channels * kernel;
    float value = bias == nullptr ? 0.0f : bias[oc];
    for (int k = t % stride; k < kernel && k <= t; k += stride) {
        const int i = (t - k) / stride;
        if (i >= length) continue;
        value += y[(size_t)oc * kernel + k + (size_t)i * m];
    }
    full[index] = value;
}

__global__ static void k_decoder_convtr_fold_batch(
    float *const *full, float *const *partial, const float *bias, int batch,
    int channels, int full_len, int tail) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = channels * tail;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int channel = local / tail;
    const int pos = local - channel * tail;
    float *row = full[request] + (size_t)channel * (size_t)full_len;
    row[pos] += partial[request][local];
    partial[request][local] = row[full_len - tail + pos] -
        (bias == nullptr ? 0.0f : bias[channel]);
}

__global__ static void k_decoder_copy_prefix_batch(
    float *const *full, float *const *outputs, int batch, int channels,
    int full_len, int output_len) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int per_request = channels * output_len;
    const int total = batch * per_request;
    if (index >= total) return;
    const int request = index / per_request;
    const int local = index - request * per_request;
    const int channel = local / output_len;
    const int pos = local - channel * output_len;
    outputs[request][local] = full[request][
        (size_t)channel * (size_t)full_len + (size_t)pos];
}

/* The Mimi decoder-transformer produces one row-major [width] activation,
 * while SEANet consumes a channel-major [width][length] frame.  Keep this
 * layout conversion on the device so the next decoder step does not need an
 * avoidable host transpose plus H2D upload. */
__global__ static void k_scatter_row_to_channels(
    const float *row, float *output, int width, int length, int position) {
    const int channel = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (channel >= width) return;
    output[(size_t)channel * (size_t)length + (size_t)position] = row[channel];
}

__global__ static void k_scatter_rows_to_channels(
    const float *rows, float *const *outputs, int batch, int width, int length,
    int position) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = batch * width;
    if (index >= total) return;
    const int request = index / width;
    const int channel = index - request * width;
    outputs[request][(size_t)channel * (size_t)length + (size_t)position] =
        rows[index];
}

/* Mimi's quantizer upsample is a depthwise causal ConvTranspose1d with one
 * input sample per call.  The first `stride` positions are returned in
 * row-major form for the resident codec transformer; the remaining
 * `kernel-stride` positions become the next call's carried tail. */
/* The per-channel body is shared by the single-request and the
 * cross-request kernels so both compile to the same arithmetic: a request's
 * upsample output must not depend on whether it ran alone or in a gang. */
__device__ __forceinline__ static void causal_depthwise_step_channel(
    const float *input, float *output, float *partial, const float *weight,
    const float *bias, int channels, int kernel, int stride, int tail,
    int channel) {
    const float channel_bias = bias == nullptr ? 0.0f : bias[channel];
    for (int position = 0; position < kernel; ++position) {
        float value = channel_bias +
            input[channel] * weight[channel * kernel + position];
        if (position < tail)
            value += partial[channel * tail + position];
        if (position < stride) {
            output[position * channels + channel] = value;
        } else if (tail > 0) {
            partial[channel * tail + (position - stride)] = value - channel_bias;
        }
    }
}

__global__ static void k_conv_transpose_causal_depthwise_step(
    const float *input, float *output, float *partial, const float *weight,
    const float *bias, int channels, int kernel, int stride, int tail) {
    /* One thread owns one channel and walks the short kernel serially.  The
     * old implementation assigned one thread to each output position and
     * read/wrote `partial` in the same launch.  For the normal Mimi shape
     * tail == stride, position 0 reads partial[0] while position stride writes
     * partial[0]: that is a real read/write race, not merely an ordering
     * concern.  Keeping the per-channel loop also handles tail > stride,
     * where the newly emitted tail overlaps the carried prefix. */
    const int channel = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (channel >= channels) return;
    causal_depthwise_step_channel(input, output, partial, weight, bias,
                                  channels, kernel, stride, tail, channel);
}

/* The same step for a gang: thread (row, channel) reads its row of the
 * stacked projection and writes/advances that row's own output and tail
 * through device pointer tables.  No state is shared between rows. */
__global__ static void k_conv_transpose_causal_depthwise_step_rows(
    const float *inputs, float *const *outputs, float *const *partials,
    const float *weight, const float *bias, int rows, int channels,
    int kernel, int stride, int tail) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (index >= rows * channels) return;
    const int row = index / channels;
    const int channel = index - row * channels;
    causal_depthwise_step_channel(inputs + (size_t)row * (size_t)channels,
                                  outputs[row], partials[row], weight, bias,
                                  channels, kernel, stride, tail, channel);
}

__global__ static void k_gather_rows_to_batch(
    float *const *inputs, float *rows, int batch, int width) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = batch * width;
    if (index >= total) return;
    const int request = index / width;
    const int column = index - request * width;
    const float *input = inputs[request];
    if (input != nullptr) rows[index] = input[column];
}

__global__ static void k_argmax(const float *logits, unsigned *result,
                                int vocab, int codebook_size, unsigned eos_id,
                                int allow_eos) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float best = -1.0e30f;
    unsigned index = 0u;
    for (int i = 0; i < vocab; ++i) {
        const bool allowed = i < codebook_size ||
                             (allow_eos != 0 && (unsigned)i == eos_id);
        if (allowed && logits[i] > best) {
            best = logits[i];
            index = (unsigned)i;
        }
    }
    result[0] = index;
}

/* ------------------------------------------------------------------ */
/*  Backend state                                                      */
/* ------------------------------------------------------------------ */

struct cuda_cached_buffer {
    const void *host_pointer;
    size_t bytes;
    float *device_pointer;
};

struct cuda_cached_fp16 {
    const void *host_pointer;
    size_t n;          /* element count */
    half *device_ptr;
};

/* Resident BF16 copy of an f32 projection weight (MYNAH_CUDA_QUANT=bf16).
 * Stored as raw bf16 bit patterns; cuBLAS reads them as CUDA_R_16BF. */
struct cuda_cached_bf16 {
    const void *host_pointer;
    size_t n;          /* element count */
    uint16_t *device_ptr;
};

struct cuda_cached_int8 {
    const void *host_pointer;
    size_t rows;
    size_t cols;
    int8_t *device_q;
    float *device_scale;
};

struct cuda_graph_entry {
    size_t rows;
    size_t iw;
    size_t ow;
    const void *weight_pointer;
    const void *bias_pointer;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    bool valid;
};

/* A resident engine graph is different from the small host matmul graph
 * above: its activations belong to one scratch arena, while request KV
 * pointers and positions are supplied through persistent metadata buffers at
 * launch time.  Keeping the identity explicit prevents a graph from outliving
 * the arena whose addresses it captured. */
struct cuda_pipeline_graph_entry {
    size_t key;
    const void *identity;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    bool valid;
    bool capturing;
};

/* A batched decoder graph owns its pointer metadata.  The eager decoder batch
 * path reuses four backend-wide device tables and fills them from small stack
 * arrays.  That is correct for eager execution, but a graph would retain
 * those host source pointers after the call returned and later uploads would
 * overwrite the same device table before its captured kernel used it.
 *
 * A graph is therefore keyed by the exact stable decoder gang.  Each captured
 * pointer-table upload gets its own pinned host table and device table.  The
 * request data itself still changes every frame through the already-resident
 * input/output/state buffers; only the addresses are immutable.  This is a
 * bounded bridge until the scheduler has a physical row arena and can use one
 * graph per width bucket rather than one graph per changing gang. */
struct cuda_decoder_batch_graph_entry {
    size_t batch;
    size_t encoder_frames;
    std::vector<mynah_backend_decoder *> decoders;
    std::vector<const float *> inputs;
    std::vector<float *> outputs;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    float **device_tables;
    float **host_tables;
    size_t upload_slots;
    size_t used_slots;
    /* Topology signature for reuse across gangs: every per-row pointer the
     * graph touches lives in host_tables, so a graph captured for one gang
     * serves any gang of the same width once the tables are rewritten. */
    size_t sig_ops;
    const float *sig_weight;
    cudaEvent_t done;
    bool valid;
    /* MYNAH_CUDA_DECODER_TABLE_PATCH: the pointer-table channel each upload
     * slot used at capture (null with the flag off).  The instantiated
     * graph's memcpy node for slot s reads host table (s, slot_channels[s]),
     * so those are the only cells a new gang has to rewrite.  layout_key
     * hashes this sequence; a decoder's cached column is usable here only
     * under the same key.  layout_drift: a later re-record used another
     * channel for some slot (never expected), patching is off for good.
     * patch_verified: a real recording of this graph reproduced, cell for
     * cell, a column cached from another graph or another row. */
    uint8_t *slot_channels;
    unsigned long long layout_key;
    bool layout_recording;
    bool layout_drift;
    bool patch_verified;
};

struct cuda_backend_state;
static void destroy_graphs(cuda_backend_state *st);
static void destroy_decoder_batch_graphs(cuda_backend_state *st);
static void tile_release(cuda_backend_state *st);
static void codec_gang_release(cuda_backend_state *st);

/* Buffers of the cross-request transformer tile, grown outside any capture. */
struct cuda_tile_workspace {
    float *x = nullptr, *xn = nullptr, *qkv = nullptr, *att = nullptr,
          *proj = nullptr, *ffn_buf = nullptr;
    /* Capacities in floats: the backbone prefill (dim 1024) and the Mimi
     * tile (dim 512) share this workspace, so it is sized to the largest
     * request seen and never shrinks when the two alternate. */
    size_t md_cap = 0u, mf_cap = 0u;
    void *meta_dev = nullptr, *meta_host = nullptr;
    size_t meta_cap = 0u;
    cudaEvent_t meta_event = nullptr;
    float *splitk = nullptr;
    size_t splitk_cap = 0u; /* floats */
    /* The tile's activation rounded to bf16 for a cuBLAS bf16 GEMM when the
     * order need not be fixed (MYNAH_CUDA_QUANT=bf16 with
     * MYNAH_CUDA_PREFILL_FIXED=0, or a per-stage bf16 switch that leaves
     * st->quant_weights unset); sized here, never inside a capture. */
    uint16_t *a16 = nullptr;
    size_t a16_cap = 0u; /* elements */
    int sms = 0;
    bool fixed_order = false; /* the current call asked for invariant GEMMs */
};

/* Staging of the cross-request Pocket codec gang (quantizer + upsample in,
 * PCM out).  Grown outside any capture; every pinned block is reused per
 * call and guarded by an event so the host never overwrites a buffer an
 * earlier async copy has not consumed yet. */
struct cuda_codec_gang_workspace {
    void *up_meta_dev = nullptr, *up_meta_host = nullptr;
    size_t up_meta_cap = 0u;
    cudaEvent_t up_meta_event = nullptr;
    float *up_proj = nullptr;
    size_t up_proj_cap = 0u; /* floats */
    void *pcm_meta_dev = nullptr, *pcm_meta_host = nullptr;
    size_t pcm_meta_cap = 0u;
    cudaEvent_t pcm_meta_event = nullptr;
    float *pcm_dev = nullptr, *pcm_host = nullptr;
    size_t pcm_cap = 0u; /* floats */
};

/* MYNAH_CUDA_KV_VMM: growable device ranges built with the CUDA virtual
 * memory management API.  A range is one virtual address reservation
 * (cuMemAddressReserve) whose prefix [0, mapped) is backed by device memory
 * (cuMemCreate, CU_MEM_LOCATION_TYPE_DEVICE: VRAM, never host memory) in
 * chunks, each its own cuMemCreate + cuMemMap, so the tail can be unmapped
 * chunk by chunk.  Growing maps a new chunk right after the last one: the
 * base pointer never moves and nothing is copied. */
typedef CUresult (*cuda_pfn_cuDeviceGet)(CUdevice *, int);
typedef CUresult (*cuda_pfn_cuDeviceGetAttribute)(int *, CUdevice_attribute,
                                                  CUdevice);
typedef CUresult (*cuda_pfn_cuMemGetAllocationGranularity)(
    size_t *, const CUmemAllocationProp *, CUmemAllocationGranularity_flags);
typedef CUresult (*cuda_pfn_cuMemAddressReserve)(CUdeviceptr *, size_t, size_t,
                                                 CUdeviceptr, unsigned long long);
typedef CUresult (*cuda_pfn_cuMemAddressFree)(CUdeviceptr, size_t);
typedef CUresult (*cuda_pfn_cuMemCreate)(CUmemGenericAllocationHandle *, size_t,
                                         const CUmemAllocationProp *,
                                         unsigned long long);
typedef CUresult (*cuda_pfn_cuMemRelease)(CUmemGenericAllocationHandle);
typedef CUresult (*cuda_pfn_cuMemMap)(CUdeviceptr, size_t, size_t,
                                      CUmemGenericAllocationHandle,
                                      unsigned long long);
typedef CUresult (*cuda_pfn_cuMemUnmap)(CUdeviceptr, size_t);
typedef CUresult (*cuda_pfn_cuMemSetAccess)(CUdeviceptr, size_t,
                                            const CUmemAccessDesc *, size_t);
typedef CUresult (*cuda_pfn_cuGetErrorString)(CUresult, const char **);

struct cuda_kv_vmm_chunk {
    CUmemGenericAllocationHandle handle;
    size_t offset; /* bytes from the range base */
    size_t bytes;
};

struct cuda_kv_vmm_range {
    size_t reserved = 0u; /* virtual bytes */
    size_t mapped = 0u;   /* physical bytes behind [0, mapped) */
    std::vector<cuda_kv_vmm_chunk> chunks;
};

struct cuda_kv_vmm {
    std::mutex mutex;
    int status = 0; /* 0 not probed, 1 usable, -1 unsupported */
    char reason[192] = {0};
    int device = 0;
    size_t granularity = 0u;
    CUmemAllocationProp prop{};
    CUmemAccessDesc access{};
    cuda_pfn_cuDeviceGet DeviceGet = nullptr;
    cuda_pfn_cuDeviceGetAttribute DeviceGetAttribute = nullptr;
    cuda_pfn_cuMemGetAllocationGranularity MemGetAllocationGranularity = nullptr;
    cuda_pfn_cuMemAddressReserve MemAddressReserve = nullptr;
    cuda_pfn_cuMemAddressFree MemAddressFree = nullptr;
    cuda_pfn_cuMemCreate MemCreate = nullptr;
    cuda_pfn_cuMemRelease MemRelease = nullptr;
    cuda_pfn_cuMemMap MemMap = nullptr;
    cuda_pfn_cuMemUnmap MemUnmap = nullptr;
    cuda_pfn_cuMemSetAccess MemSetAccess = nullptr;
    cuda_pfn_cuGetErrorString GetErrorString = nullptr;
    std::unordered_map<uintptr_t, cuda_kv_vmm_range> ranges;
    /* Live ranges, read without the lock by mynah_cuda_dev_free. */
    std::atomic<size_t> live{0u};
    std::atomic<unsigned long long> mapped_bytes{0ull};
    std::atomic<unsigned long long> reserved_bytes{0ull};
    std::atomic<unsigned long long> maps{0ull};
    std::atomic<unsigned long long> unmaps{0ull};
};

struct cuda_backend_state {
    cublasHandle_t cublas;
    cudaStream_t stream;
    std::vector<cuda_cached_buffer> weights;
    std::vector<cuda_cached_fp16> weights_fp16;
    std::vector<cuda_cached_int8> weights_int8;
    std::vector<cuda_cached_bf16> weights_bf16;
    /* BF16 activation rows for the resident BF16-weight GEMM; reserved
     * before graph capture like the Q8 workspace. */
    uint16_t *dev_bf16_activation;
    size_t dev_bf16_activation_cap; /* elements */
    /* Device scratch for activations (grows on demand). */
    float *dev_scratch;
    size_t dev_scratch_cap;   /* bytes */
    /* Reusable FP32 staging for host<->BF16 KV conversion.  It is grown only
     * before a new resident prefix is submitted and never inside a graph. */
    float *dev_bf16_convert;
    size_t dev_bf16_convert_cap; /* floats */
    int8_t *dev_q8_activation;
    float *dev_q8_activation_scale;
    int32_t *dev_q8_accum;
    size_t dev_q8_activation_cap;
    size_t dev_q8_rows_cap;
    size_t dev_q8_accum_cap;
    /* Pinned host buffer for H2D/D2H. */
    float *host_buf;
    float *dev_buf;           /* mapped device pointer for host_buf */
    size_t host_buf_cap;
    unsigned *dev_argmax;
    void *cublas_workspace;
    size_t cublas_workspace_cap;
    /* Metadata for independent-request attention batches.  The arrays are
     * allocated with the backend, never from the autoregressive hot loop. */
    float **dev_batch_k_cache;
    float **dev_batch_v_cache;
    size_t *dev_batch_positions;
    size_t *dev_batch_cache_strides;
    /* MYNAH_CUDA_SHARED_VOICE: per-row voice-prefix K/V planes of the layer
     * being run (the model-owned device voice cache) and their length. */
    void **dev_batch_k_prefix;
    void **dev_batch_v_prefix;
    size_t *dev_batch_prefix_len;
    /* Pointer tables for the cross-request causal decoder.  Each table is
     * reused for one topology operation at a time; decoder state itself stays
     * in the per-request objects. */
    float **dev_decoder_ptr0;
    float **dev_decoder_ptr1;
    float **dev_decoder_ptr2;
    float **dev_decoder_ptr3;
    size_t batch_meta_cap;
    bool fast_math;
    bool tf32; /* MYNAH_CUDA_TF32: FP32 GEMMs on TF32 tensor cores */
    bool tile_cublas; /* MYNAH_CUDA_TILE_CUBLAS: tile projections via cuBLAS */
    /* MYNAH_CUDA_QUANT=bf16|int8 (or MYNAH_CUDA_Q8=1): the engine may hand
     * the GEMMs reduced-precision weights, whose tensor-core paths are not
     * batch-invariant. */
    bool quant_weights;
    /* MYNAH_CUDA_BF16_LT (default on; =0 is the rollback): the bf16 Linears
     * through cuBLASLt with a per-shape heuristic algorithm (what a PyTorch
     * bf16 F.linear does), instead of cublasGemmEx with CUBLAS_GEMM_DEFAULT.
     * Created by the first bf16 workspace reserve (cuda_lt_init), so a run
     * without bf16 weights never creates it. Null handle = off. */
    cublasLtHandle_t lt = nullptr;
    bool lt_tried = false;
    void *lt_ws = nullptr;
    size_t lt_ws_cap = 0u;
    struct lt_algo_entry { int m, n, k; bool ok; cublasLtMatmulAlgo_t algo; };
    std::vector<lt_algo_entry> lt_algos;
    bool graphs_enabled;
    std::vector<cuda_graph_entry> graph_cache;
    std::vector<cuda_pipeline_graph_entry> pipeline_graphs;
    std::vector<cuda_decoder_batch_graph_entry *> decoder_batch_graphs;
    cuda_decoder_batch_graph_entry *active_decoder_batch_graph;
    size_t active_decoder_upload_slot;
    float **active_decoder_tables[4];
    std::atomic<unsigned long long> h2d_bytes;
    std::atomic<unsigned long long> d2h_bytes;
    std::atomic<unsigned long long> h2d_calls;
    std::atomic<unsigned long long> d2h_calls;
    std::atomic<unsigned long long> sync_calls;
    std::atomic<unsigned long long> graph_captures;
    std::atomic<unsigned long long> graph_replays;
    std::atomic<unsigned long long> graph_fallbacks;
    std::atomic<unsigned long long> backbone_batch_calls;
    std::atomic<unsigned long long> backbone_batch_items;
    std::atomic<unsigned long long> backbone_batch_max_width;
    std::atomic<unsigned long long> codec_transformer_batch_calls;
    std::atomic<unsigned long long> codec_transformer_batch_items;
    std::atomic<unsigned long long> codec_transformer_batch_max_width;
    std::atomic<unsigned long long> codec_upsample_steps;
    std::atomic<unsigned long long> codec_upsample_fallbacks;
    std::atomic<unsigned long long> decoder_steps;
    std::atomic<unsigned long long> decoder_batch_calls;
    std::atomic<unsigned long long> decoder_batch_items;
    std::atomic<unsigned long long> decoder_batch_max_width;
    std::atomic<unsigned long long> decoder_batch_frames;
    std::atomic<unsigned long long> decoder_graph_captures;
    std::atomic<unsigned long long> decoder_graph_replays;
    std::atomic<unsigned long long> decoder_graph_fallbacks;
    /* Same-width gang changes served by a host re-record of the decoder
     * graph, and by MYNAH_CUDA_DECODER_TABLE_PATCH's column scatter. */
    std::atomic<unsigned long long> decoder_graph_rerecords;
    std::atomic<unsigned long long> decoder_graph_table_patches;
    /* MYNAH_CUDA_DECODER_VALIDATE_ONCE: next compatibility class id. */
    std::atomic<unsigned long long> decoder_compat_next;
    std::atomic<unsigned long long> decoder_failures;
    std::atomic<unsigned long long> resident_fallbacks;
    std::atomic<unsigned long long> matmul_calls;
    std::atomic<unsigned long long> matvec_calls;
    std::atomic<unsigned long long> q8_matmul_calls;
    std::atomic<unsigned long long> tile_calls;
    std::atomic<unsigned long long> width_hist[3][8];
    std::atomic<unsigned long long> tile_rows;
    cuda_tile_workspace tile;
    float *dec_cols = nullptr;   /* one-GEMM decoder: [inner][batch*len] */
    float *dec_out = nullptr;    /* one-GEMM decoder: [out][batch*len]   */
    size_t dec_cols_cap = 0u, dec_out_cap = 0u; /* floats */
    float *dec_tr_x = nullptr, *dec_tr_y = nullptr; /* convtr GEMM buffers */
    size_t dec_tr_x_cap = 0u, dec_tr_y_cap = 0u;
    /* MYNAH_CUDA_ROW_MEM_DIET: the single-request decoder scratch shared by
     * every lean decoder (see mynah_backend_decoder::lean).  Allocated by the
     * first lean decoder open, never moved or freed before cuda_close, so a
     * captured single-request decoder graph may bake these pointers.  All
     * decoder work is ordered on the one backend stream, and every buffer is
     * fully written before it is read within one decoder step. */
    float *solo_work_a = nullptr, *solo_work_b = nullptr, *solo_work_c = nullptr;
    float *solo_columns = nullptr, *solo_window = nullptr, *solo_full = nullptr;
    size_t solo_work_cap = 0u, solo_columns_cap = 0u, solo_window_cap = 0u,
           solo_full_cap = 0u; /* floats */
    std::vector<float *> solo_retired; /* replaced smaller sets */
    cuda_codec_gang_workspace codec_gang;
    std::atomic<unsigned long long> codec_gang_calls[2];
    std::atomic<unsigned long long> codec_gang_rows[2];
    std::atomic<unsigned long long> codec_gang_hist[2][8];
    std::atomic<unsigned long long> q8_rows;
    std::atomic<unsigned long long> q8_weight_uploads;
    std::atomic<unsigned long long> q8_weight_bytes;
    std::atomic<unsigned long long> q8_activation_bytes;
    std::atomic<unsigned long long> bf16_matmul_calls;
    std::atomic<unsigned long long> bf16_rows;
    std::atomic<unsigned long long> bf16_weight_bytes;
    bool q8_enabled;
    bool decoder_batch_enabled;
    cuda_kv_vmm kv_vmm;
    /* int8 backbone KV (default; mynah_backend_set_kv_int8): the BF16-KV
     * batched decode attention reads/writes int8 records. */
    bool kv_int8;
};

enum cuda_decoder_op_kind {
    CUDA_DECODER_CONV = 0,
    CUDA_DECODER_CONVTR = 1,
    CUDA_DECODER_RESBLOCK = 2
};

struct cuda_decoder_op {
    cuda_decoder_op_kind kind;
    int pre_elu;
    int in_channels;
    int out_channels;
    int kernel;
    int stride;
    int dilation;
    int groups;
    size_t max_in_len;
    size_t tail;
    size_t max_full_len;
    float *weight;
    float *bias;
    /* MYNAH_CUDA_SEANET_BF16: BF16 copy of `weight`, same layout, shared
     * through the backend BF16 weight cache (never freed by the op).  Null
     * when the flag is off or the op has no GEMM form; a null pointer selects
     * the FP32 path, so with the flag off nothing below changes. */
    uint16_t *weight_bf16;
    float *previous;
    float *window;
    float *partial;
    float *full;
};

struct mynah_backend_decoder {
    cuda_backend_state *backend;
    size_t channels;
    size_t dimension;
    size_t n_filters;
    size_t max_encoder_frames;
    float elu_alpha;
    size_t work_floats;
    size_t columns_floats;
    float *work_a;
    float *work_b;
    float *work_c;
    float *columns;
    std::vector<cuda_decoder_op> ops;
    /* MYNAH_CUDA_ROW_MEM_DIET (`lean`): the single-request path's scratch
     * (work_a/b/c, columns above, and the causal window / transposed-conv
     * `full` of every op, which stay null in the ops) is the backend's one
     * shared set (cuda_backend_state::solo_*), not owned here.  The
     * cross-request gang, which needs a private set per row, uses gang_a,
     * gang_b and gang_columns instead; its fused path needs neither work_c,
     * nor a window, nor `full`.  Off: every field below stays zero and the
     * decoder owns exactly the buffers it always did. */
    bool lean;
    float *solo_window;   /* shared, max over ops; lean only */
    float *solo_full;     /* shared, max over ops; lean only */
    float *gang_a;        /* owned, work_floats; lean only */
    float *gang_b;        /* owned, work_floats; lean only */
    void *gang_columns;   /* owned, gang_columns_bytes; lean only */
    size_t gang_columns_bytes;
    /* Topology sizes in floats, filled by decoder_build. */
    size_t max_window_floats;
    size_t max_full_floats;
    size_t sum_window_floats;
    size_t sum_full_floats;
    size_t state_floats;  /* previous + partial of every op */
    /* MYNAH_CUDA_DECODER_VALIDATE_ONCE: nonzero once this decoder's topology
     * was compared op by op with a gang's first row.  Every field
     * decoder_ops_compatible reads is fixed at open, so two decoders with
     * the same class are compatible without comparing again; ids come from a
     * backend counter and are never reused, a re-opened decoder starts at 0. */
    unsigned long long compat_class;
    /* MYNAH_CUDA_DECODER_TABLE_PATCH: this decoder's row of every pointer
     * table a decoder batch graph uploads (one entry per upload slot), as a
     * real recording wrote it for (table_input, table_output) under graph
     * layout table_layout (0 = none).  Its buffers do not change while it
     * lives, pooled or not, so the column stays valid.  table_source and
     * table_row say which graph and row it was read from (verification). */
    std::vector<float *> table_column;
    unsigned long long table_layout;
    const float *table_input;
    float *table_output;
    const void *table_source;
    size_t table_row;
};

static constexpr size_t CUDA_BATCH_META_CAP = MYNAH_ROW_CAP;
static constexpr size_t CUDA_DECODER_GRAPH_CAP = MYNAH_ROW_CAP;
/* A server can retain one decoder graph per live context in addition to the
 * width-bucketed backbone/flow graphs.  The condition-projection input graph
 * is a separate bucket because its input is already resident in `cuda_x` and
 * must not accidentally replay an H2D node from the ordinary bucket.  Keep
 * enough finite room for 128 request decoders plus the three 128-width bucket
 * families. */
static constexpr size_t CUDA_PIPELINE_GRAPH_CAP = MYNAH_ROW_CAP;

static void set_error(char *e, size_t c, const char *m) {
    if (e && c > 0) std::snprintf(e, c, "%s", m);
}

static bool cuda_size_mul(size_t a, size_t b, size_t *out) {
    if (a != 0u && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool cuda_size_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return false;
    *out = a + b;
    return true;
}

static int ce(cudaError_t r, char *e, size_t c) {
    if (r == cudaSuccess) return 0;
    std::snprintf(e, c, "CUDA: %s", cudaGetErrorString(r)); return -1;
}
static int cbe(cublasStatus_t s, char *e, size_t c) {
    if (s == CUBLAS_STATUS_SUCCESS) return 0;
    std::snprintf(e, c, "cuBLAS: %d", (int)s); return -1;
}

static bool cuda_fast_math_enabled(void) {
    const char *value = std::getenv("MYNAH_CUDA_FAST_MATH");
    return value != nullptr && std::strcmp(value, "0") != 0;
}

/* Mirrors the engine's resolution of MYNAH_CUDA_QUANT (f32|bf16|int8) and
 * the explicit MYNAH_CUDA_Q8 override, without linking the C helper. */
static bool cuda_env_enabled(const char *name, bool fallback);
static bool cuda_quant_weights_requested(void) {
    if (getenv("MYNAH_CUDA_Q8") != nullptr &&
        cuda_env_enabled("MYNAH_CUDA_Q8", false))
        return true;
    const char *v = getenv("MYNAH_CUDA_QUANT");
    if (v == nullptr || v[0] == '\0') return false;
    return strcmp(v, "bf16") == 0 || strcmp(v, "bfloat16") == 0 ||
           strcmp(v, "int8") == 0 || strcmp(v, "q8") == 0;
}

static bool cuda_env_enabled(const char *name, bool fallback) {
    const char *value = std::getenv(name);
    return value == nullptr ? fallback : std::strcmp(value, "0") != 0;
}

static bool cuda_decoder_batch_enabled(void) {
    return cuda_env_enabled("MYNAH_CUDA_DECODER_BATCH", true);
}

/* MYNAH_CUDA_SEANET_BF16=1 (default off): the resident SEANet decoder's
 * convolution GEMMs read BF16 operands (im2col columns / gathered input and a
 * BF16 weight copy made once when the decoder is built) on tensor cores with
 * FP32 accumulation and an FP32 output.  Causal states, bias, ELU, residual
 * adds and the returned audio stay FP32.  Read once per process: decoders
 * opened later must agree with the ones already in a gang. */
/* MYNAH_CUDA_SEANET_BF16 (default on; 0 = FP32 decoder convs): every GEMM-form decoder conv in BF16; the
 * diagnostic values "conv" and "convtr" restrict it to one kind, to locate a
 * solo-vs-gang divergence. */
static int cuda_seanet_bf16_mask(void) {
    static const int mask = [] {
        const char *v = std::getenv("MYNAH_CUDA_SEANET_BF16");
        if (v != nullptr && std::strcmp(v, "0") == 0) return 0;
        if (v == nullptr || *v == '\0') return 3; /* default on; =0 is the rollback */
        if (std::strcmp(v, "conv") == 0) return 1;
        if (std::strcmp(v, "convtr") == 0) return 2;
        return 3;
    }();
    return mask;
}
static bool cuda_seanet_bf16_enabled(void) { return cuda_seanet_bf16_mask() != 0; }

/* CUBLAS_PEDANTIC_MATH (MYNAH_CUDA_TF32=0) turns CUBLAS_COMPUTE_32F into its
 * pedantic form, which may refuse tensor cores for BF16 inputs.  The BF16
 * SEANet GEMMs ask for tensor cores explicitly, so they lift a pedantic mode
 * for the duration of the call and restore it.  The math mode is host-side
 * handle state, so this is safe inside a stream capture.  In the default
 * TF32 mode, and in fast-math mode, the guard does nothing. */
struct cuda_bf16_math_scope {
    cublasHandle_t handle;
    cublasMath_t saved;
    bool restore;
    explicit cuda_bf16_math_scope(cublasHandle_t h)
        : handle(h), saved(CUBLAS_DEFAULT_MATH), restore(false) {
        if (cublasGetMathMode(handle, &saved) == CUBLAS_STATUS_SUCCESS &&
            (((int)saved) & 0xf) == (int)CUBLAS_PEDANTIC_MATH &&
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) ==
                CUBLAS_STATUS_SUCCESS)
            restore = true;
    }
    ~cuda_bf16_math_scope() {
        if (restore) (void)cublasSetMathMode(handle, saved);
    }
    cuda_bf16_math_scope(const cuda_bf16_math_scope &) = delete;
    cuda_bf16_math_scope &operator=(const cuda_bf16_math_scope &) = delete;
};

static bool cuda_graphs_enabled(void) {
    const char *value = std::getenv("MYNAH_CUDA_GRAPHS");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

static cublasComputeType_t cuda_compute_type(const cuda_backend_state *st) {
    if (st->fast_math) return CUBLAS_COMPUTE_32F_FAST_16F;
    return st->tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

static cublasGemmAlgo_t cuda_gemm_algo(const cuda_backend_state *st) {
    return st->fast_math || st->tf32 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP
                                     : CUBLAS_GEMM_DEFAULT;
}

static int ensure_scratch(cuda_backend_state *st, size_t bytes, char *e, size_t ec) {
    if (st->dev_scratch_cap >= bytes) return 0;
    size_t rounded = 0;
    if (!cuda_size_add(bytes, (32u << 20) - 1u, &rounded)) {
        set_error(e, ec, "CUDA scratch size overflow");
        return -1;
    }
    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    destroy_graphs(st);
    if (st->dev_scratch) cudaFree(st->dev_scratch);
    st->dev_scratch = nullptr;
    st->dev_scratch_cap = 0;
    size_t cap = rounded & ~((32u << 20) - 1u);
    if (ce(cudaMalloc(&st->dev_scratch, cap), e, ec)) { st->dev_scratch = nullptr; st->dev_scratch_cap = 0; return -1; }
    st->dev_scratch_cap = cap;
    return 0;
}

static int ensure_host(cuda_backend_state *st, size_t bytes, char *e, size_t ec) {
    if (st->host_buf_cap >= bytes) return 0;
    size_t rounded = 0;
    if (!cuda_size_add(bytes, (16u << 20) - 1u, &rounded)) {
        set_error(e, ec, "CUDA host staging size overflow");
        return -1;
    }
    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    destroy_graphs(st);
    if (st->host_buf) cudaFreeHost(st->host_buf);
    st->host_buf = nullptr; st->dev_buf = nullptr; st->host_buf_cap = 0;
    size_t cap = rounded & ~((16u << 20) - 1u);
    if (ce(cudaHostAlloc(&st->host_buf, cap, cudaHostAllocMapped), e, ec)) return -1;
    if (ce(cudaHostGetDevicePointer(&st->dev_buf, st->host_buf, 0), e, ec)) {
        cudaFreeHost(st->host_buf); st->host_buf = nullptr; return -1;
    }
    st->host_buf_cap = cap;
    return 0;
}

static int ensure_bf16_convert(cuda_backend_state *st, size_t elements,
                               char *e, size_t ec) {
    if (st->dev_bf16_convert_cap >= elements) return 0;
    if (elements == 0u || elements > SIZE_MAX / sizeof(float)) {
        set_error(e, ec, "CUDA BF16 conversion size overflow");
        return -1;
    }
    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    destroy_graphs(st);
    if (st->dev_bf16_convert) cudaFree(st->dev_bf16_convert);
    st->dev_bf16_convert = nullptr;
    st->dev_bf16_convert_cap = 0u;
    if (ce(cudaMalloc(&st->dev_bf16_convert, elements * sizeof(float)), e, ec))
        return -1;
    st->dev_bf16_convert_cap = elements;
    return 0;
}

static int cached_weight(cuda_backend_state *st, const float *hp, size_t bytes,
                         float **dp, char *e, size_t ec) {
    if (st == nullptr || hp == nullptr || dp == nullptr || bytes == 0u) {
        set_error(e, ec, "invalid CUDA weight cache request");
        return -1;
    }
    for (auto &c : st->weights)
        if (c.host_pointer == hp && c.bytes == bytes) { *dp = c.device_pointer; return 0; }
    float *d = nullptr;
    if (ce(cudaMalloc(&d, bytes), e, ec)) return -1;
    if (ce(cudaMemcpy(d, hp, bytes, cudaMemcpyHostToDevice), e, ec)) { cudaFree(d); return -1; }
    st->weights.push_back({hp, bytes, d});
    *dp = d;
    return 0;
}

static int cached_weight_fp16(cuda_backend_state *st, const float *hp, size_t n,
                              half **dp, char *e, size_t ec) {
    size_t bytes = 0;
    if (st == nullptr || hp == nullptr || dp == nullptr || n == 0u ||
        n > (size_t)INT_MAX || !cuda_size_mul(n, sizeof(float), &bytes)) {
        set_error(e, ec, "invalid CUDA FP16 weight cache request");
        return -1;
    }
    for (auto &c : st->weights_fp16)
        if (c.host_pointer == hp && c.n == n) { *dp = c.device_ptr; return 0; }
    float *tmp = nullptr;
    half *d16 = nullptr;
    size_t half_bytes = 0;
    if (!cuda_size_mul(n, sizeof(half), &half_bytes)) {
        set_error(e, ec, "CUDA FP16 weight size overflow");
        return -1;
    }
    if (ce(cudaMalloc(&tmp, bytes), e, ec)) return -1;
    if (ce(cudaMalloc(&d16, half_bytes), e, ec)) { cudaFree(tmp); return -1; }
    if (ce(cudaMemcpyAsync(tmp, hp, bytes, cudaMemcpyHostToDevice,
                           st->stream), e, ec)) {
        cudaFree(tmp);
        cudaFree(d16);
        return -1;
    }
    k_f32_to_f16<<<((int)n+255)/256, 256, 0, st->stream>>>(tmp, d16, (int)n);
    if (ce(cudaGetLastError(), e, ec)) {
        cudaFree(tmp);
        cudaFree(d16);
        return -1;
    }
    /* The temporary is not part of the captured graph and can be reclaimed
     * after the stream reaches the conversion.  cudaFree provides the needed
     * ordering on supported CUDA runtimes. */
    if (ce(cudaFree(tmp), e, ec)) {
        cudaFree(d16);
        return -1;
    }
    st->weights_fp16.push_back({hp, n, d16});
    *dp = d16;
    return 0;
}

/* Same shape as cached_weight_fp16: one H2D of the f32 tensor into a
 * temporary, one device RNE conversion, then the f32 temporary is released.
 * The first (uncaptured) call of every resident stage warms this cache; a
 * graph capture then only records the GEMM reading the cached pointer. */
static int cached_weight_bf16(cuda_backend_state *st, const float *hp,
                              size_t n, uint16_t **dp, char *e, size_t ec) {
    size_t bytes = 0u;
    size_t bf16_bytes = 0u;
    if (st == nullptr || hp == nullptr || dp == nullptr || n == 0u ||
        n > (size_t)INT_MAX || !cuda_size_mul(n, sizeof(float), &bytes) ||
        !cuda_size_mul(n, sizeof(uint16_t), &bf16_bytes)) {
        set_error(e, ec, "invalid CUDA BF16 weight cache request");
        return -1;
    }
    for (auto &c : st->weights_bf16)
        if (c.host_pointer == hp && c.n == n) { *dp = c.device_ptr; return 0; }
    float *tmp = nullptr;
    uint16_t *d16 = nullptr;
    if (ce(cudaMalloc(&tmp, bytes), e, ec)) return -1;
    if (ce(cudaMalloc((void **)&d16, bf16_bytes), e, ec)) {
        cudaFree(tmp);
        return -1;
    }
    if (ce(cudaMemcpyAsync(tmp, hp, bytes, cudaMemcpyHostToDevice,
                           st->stream), e, ec)) {
        cudaFree(tmp);
        cudaFree(d16);
        return -1;
    }
    k_f32_to_bf16<<<((int)n + 255) / 256, 256, 0, st->stream>>>(tmp, d16,
                                                                 (int)n);
    if (ce(cudaGetLastError(), e, ec) ||
        ce(cudaStreamSynchronize(st->stream), e, ec)) {
        cudaFree(tmp);
        cudaFree(d16);
        return -1;
    }
    cudaFree(tmp);
    try {
        st->weights_bf16.push_back({hp, n, d16});
    } catch (const std::bad_alloc &) {
        cudaFree(d16);
        set_error(e, ec, "out of host memory indexing CUDA BF16 weight");
        return -1;
    }
    st->bf16_weight_bytes.fetch_add((unsigned long long)bf16_bytes,
                                    std::memory_order_relaxed);
    *dp = d16;
    return 0;
}

static int ensure_bf16_activation(cuda_backend_state *st, size_t elements,
                                  char *e, size_t ec) {
    if (st == nullptr || elements == 0u || elements > SIZE_MAX / 2u) {
        set_error(e, ec, "invalid CUDA BF16 activation workspace size");
        return -1;
    }
    if (st->dev_bf16_activation_cap >= elements) return 0;
    if (ce(cudaStreamSynchronize(st->stream), e, ec) != 0) return -1;
    destroy_graphs(st);
    if (st->dev_bf16_activation != nullptr) cudaFree(st->dev_bf16_activation);
    st->dev_bf16_activation = nullptr;
    st->dev_bf16_activation_cap = 0u;
    if (ce(cudaMalloc((void **)&st->dev_bf16_activation,
                      elements * sizeof(uint16_t)), e, ec) != 0)
        return -1;
    st->dev_bf16_activation_cap = elements;
    return 0;
}

/* MYNAH_CUDA_BF16_LT (default on): create the cuBLASLt handle and its
 * workspace once, on the first bf16 workspace reserve.  That reserve runs
 * before any graph capture and only when an engine really has bf16 weights,
 * so an f32 run (or another model on this backend) never pays the 32 MiB or
 * prints the line.  A failure leaves the bf16 Linears on cublasGemmEx. */
static void cuda_lt_init(cuda_backend_state *st) {
    if (st == nullptr || st->lt_tried) return;
    st->lt_tried = true;
    if (!cuda_env_enabled("MYNAH_CUDA_BF16_LT", true)) return;
    /* 32 MiB of workspace, allocated once, so no algorithm the heuristic
     * picks ever needs an allocation inside a graph capture. */
    const size_t ws = (size_t)32 << 20;
    if (cublasLtCreate(&st->lt) != CUBLAS_STATUS_SUCCESS ||
        cudaMalloc(&st->lt_ws, ws) != cudaSuccess) {
        if (st->lt) cublasLtDestroy(st->lt);
        st->lt = nullptr;
        st->lt_ws = nullptr;
        std::fprintf(stderr, "mynah-tts: MYNAH_CUDA_BF16_LT unavailable, "
                             "bf16 Linears stay on cublasGemmEx\n");
    } else {
        st->lt_ws_cap = ws;
        std::fprintf(stderr, "mynah-tts: CUDA bf16 Linears through cuBLASLt "
                             "with per-shape heuristic algorithms "
                             "(MYNAH_CUDA_BF16_LT=1)\n");
    }
}

extern "C" int mynah_cuda_bf16_reserve(void *opaque, size_t activation_count,
                                        char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    const int rc = ensure_bf16_activation(st, activation_count, e, ec);
    if (rc == 0) cuda_lt_init(st);
    return rc;
}

static int cuda_q8_round(float value) {
    int q = (int)(value >= 0.0f ? value + 0.5f : value - 0.5f);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    return q;
}

/* Host-side packing is intentionally the same small scalar formula as
 * src/qmat.c. It runs once per immutable weight tensor, not in the decode
 * loop; using it here also makes the device cache's bytes reproducible across
 * CUDA versions. */
static int cuda_pack_weight_q8(const float *host, size_t rows, size_t cols,
                              std::vector<int8_t> *q,
                              std::vector<float> *scales,
                              char *e, size_t ec) {
    if (host == nullptr || q == nullptr || scales == nullptr || rows == 0u ||
        cols == 0u || rows > (size_t)INT_MAX || cols > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA Q8 weight dimensions");
        return -1;
    }
    size_t elements = 0u;
    if (!cuda_size_mul(rows, cols, &elements)) {
        set_error(e, ec, "CUDA Q8 weight size overflow");
        return -1;
    }
    try {
        q->assign(elements, (int8_t)0);
        scales->assign(rows, 1.0f);
    } catch (const std::bad_alloc &) {
        set_error(e, ec, "out of host memory packing CUDA Q8 weight");
        return -1;
    }
    for (size_t row = 0; row < rows; ++row) {
        const float *src = host + row * cols;
        float amax = 0.0f;
        for (size_t col = 0; col < cols; ++col) {
            const float value = std::fabs(src[col]);
            if (value > amax) amax = value;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        (*scales)[row] = scale;
        const float inv = 1.0f / scale;
        int8_t *dst = q->data() + row * cols;
        for (size_t col = 0; col < cols; ++col)
            dst[col] = (int8_t)cuda_q8_round(src[col] * inv);
    }
    return 0;
}

static int cached_weight_q8(cuda_backend_state *st, const float *host,
                            size_t rows, size_t cols, int8_t **device_q,
                            float **device_scale, char *e, size_t ec) {
    if (st == nullptr || host == nullptr || device_q == nullptr ||
        device_scale == nullptr || rows == 0u || cols == 0u) {
        set_error(e, ec, "invalid CUDA Q8 weight cache request");
        return -1;
    }
    for (auto &entry : st->weights_int8) {
        if (entry.host_pointer == host && entry.rows == rows &&
            entry.cols == cols) {
            *device_q = entry.device_q;
            *device_scale = entry.device_scale;
            return 0;
        }
    }
    std::vector<int8_t> host_q;
    std::vector<float> host_scale;
    if (cuda_pack_weight_q8(host, rows, cols, &host_q, &host_scale, e, ec) != 0)
        return -1;
    size_t q_bytes = 0u;
    size_t scale_bytes = 0u;
    if (!cuda_size_mul(host_q.size(), sizeof(int8_t), &q_bytes) ||
        !cuda_size_mul(host_scale.size(), sizeof(float), &scale_bytes)) {
        set_error(e, ec, "CUDA Q8 weight byte size overflow");
        return -1;
    }
    int8_t *q = nullptr;
    float *scale = nullptr;
    if (ce(cudaMalloc((void **)&q, q_bytes), e, ec) != 0 ||
        ce(cudaMalloc((void **)&scale, scale_bytes), e, ec) != 0) {
        if (q != nullptr) cudaFree(q);
        if (scale != nullptr) cudaFree(scale);
        return -1;
    }
    if (ce(cudaMemcpy(q, host_q.data(), q_bytes, cudaMemcpyHostToDevice), e,
           ec) != 0 ||
        ce(cudaMemcpy(scale, host_scale.data(), scale_bytes,
                      cudaMemcpyHostToDevice), e, ec) != 0) {
        cudaFree(q);
        cudaFree(scale);
        return -1;
    }
    try {
        st->weights_int8.push_back({host, rows, cols, q, scale});
    } catch (const std::bad_alloc &) {
        cudaFree(q);
        cudaFree(scale);
        set_error(e, ec, "out of host memory indexing CUDA Q8 weight");
        return -1;
    }
    st->q8_weight_uploads.fetch_add(1ull, std::memory_order_relaxed);
    st->q8_weight_bytes.fetch_add((unsigned long long)(q_bytes + scale_bytes),
                                  std::memory_order_relaxed);
    *device_q = q;
    *device_scale = scale;
    return 0;
}

static int ensure_q8_workspace(cuda_backend_state *st, size_t activation_count,
                               size_t rows, size_t output_count, char *e,
                               size_t ec) {
    if (st == nullptr || activation_count == 0u || rows == 0u ||
        output_count == 0u) {
        set_error(e, ec, "invalid CUDA Q8 workspace dimensions");
        return -1;
    }
    if (st->dev_q8_activation_cap >= activation_count &&
        st->dev_q8_rows_cap >= rows && st->dev_q8_accum_cap >= output_count)
        return 0;
    size_t activation_bytes = 0u;
    size_t row_bytes = 0u;
    size_t output_bytes = 0u;
    if (!cuda_size_mul(activation_count, sizeof(int8_t), &activation_bytes) ||
        !cuda_size_mul(rows, sizeof(float), &row_bytes) ||
        !cuda_size_mul(output_count, sizeof(int32_t), &output_bytes)) {
        set_error(e, ec, "CUDA Q8 workspace size overflow");
        return -1;
    }
    if (ce(cudaStreamSynchronize(st->stream), e, ec) != 0) return -1;
    destroy_graphs(st);
    if (st->dev_q8_activation_cap < activation_count) {
        if (st->dev_q8_activation != nullptr) cudaFree(st->dev_q8_activation);
        st->dev_q8_activation = nullptr;
        st->dev_q8_activation_cap = 0u;
        if (ce(cudaMalloc((void **)&st->dev_q8_activation,
                          activation_bytes), e, ec) != 0)
            return -1;
        st->dev_q8_activation_cap = activation_count;
    }
    if (st->dev_q8_rows_cap < rows) {
        if (st->dev_q8_activation_scale != nullptr)
            cudaFree(st->dev_q8_activation_scale);
        st->dev_q8_activation_scale = nullptr;
        st->dev_q8_rows_cap = 0u;
        if (ce(cudaMalloc((void **)&st->dev_q8_activation_scale,
                          row_bytes), e, ec) != 0)
            return -1;
        st->dev_q8_rows_cap = rows;
    }
    if (st->dev_q8_accum_cap < output_count) {
        if (st->dev_q8_accum != nullptr) cudaFree(st->dev_q8_accum);
        st->dev_q8_accum = nullptr;
        st->dev_q8_accum_cap = 0u;
        if (ce(cudaMalloc((void **)&st->dev_q8_accum,
                          output_bytes), e, ec) != 0)
            return -1;
        st->dev_q8_accum_cap = output_count;
    }
    return 0;
}

/* Graph callers reserve this before capture.  A resize synchronizes and
 * invalidates graphs, so allowing the first larger batch to discover the
 * allocation from inside a captured projection would make capture fail. */
extern "C" int mynah_cuda_q8_reserve(void *opaque, size_t activation_count,
                                      size_t rows, size_t output_count,
                                      char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    return ensure_q8_workspace(st, activation_count, rows, output_count, e, ec);
}

static int validate_cuda_matmul(const void *input, const void *output,
                                const float *weight, size_t rows, size_t iw,
                                size_t ow, size_t *in_n, size_t *out_n,
                                size_t *weight_n, size_t *total_n,
                                char *e, size_t ec) {
    size_t input_bytes = 0, output_bytes = 0, weight_bytes = 0, total_bytes = 0;
    if (input == nullptr || output == nullptr || weight == nullptr || rows == 0u ||
        iw == 0u || ow == 0u || rows > (size_t)INT_MAX ||
        iw > (size_t)INT_MAX || ow > (size_t)INT_MAX ||
        !cuda_size_mul(rows, iw, in_n) ||
        !cuda_size_mul(rows, ow, out_n) ||
        !cuda_size_mul(iw, ow, weight_n) ||
        !cuda_size_add(*in_n, *out_n, total_n) ||
        !cuda_size_mul(*in_n, sizeof(float), &input_bytes) ||
        !cuda_size_mul(*out_n, sizeof(float), &output_bytes) ||
        !cuda_size_mul(*weight_n, sizeof(float), &weight_bytes) ||
        !cuda_size_mul(*total_n, sizeof(float), &total_bytes)) {
        set_error(e, ec, "invalid CUDA matmul dimensions");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  matmul / sgemm (existing interface, with sync)                     */
/* ------------------------------------------------------------------ */

static int cuda_matmul(void *opaque, const float *input, float *output, size_t rows,
                       size_t iw, size_t ow, const float *weight, const float *bias,
                       char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t in_n = 0, out_n = 0, w_n = 0, total_n = 0;
    if (st == nullptr || validate_cuda_matmul(input, output, weight, rows, iw, ow,
                                               &in_n, &out_n, &w_n, &total_n,
                                               e, ec) != 0) return -1;
    if (!st->fast_math) {
        float *dw = nullptr;
        if (cached_weight(st, weight, w_n * sizeof(float), &dw, e, ec)) return -1;
        float *db = nullptr;
        if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
        if (ensure_host(st, total_n * sizeof(float), e, ec)) return -1;
        std::memcpy(st->host_buf, input, in_n * sizeof(float));
        cublasSetStream(st->cublas, st->stream);
        const float alpha = 1.0f, beta = 0.0f;
        if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             (int)ow, (int)rows, (int)iw,
                             &alpha, dw, CUDA_R_32F, (int)iw,
                             st->dev_buf, CUDA_R_32F, (int)iw,
                             &beta, st->dev_buf + in_n, CUDA_R_32F, (int)ow,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), e, ec)) return -1;
        if (db) {
            k_bias_add<<<((int)(rows * ow) + 255) / 256, 256, 0, st->stream>>>(
                st->dev_buf + in_n, db, (int)rows, (int)ow);
            if (ce(cudaGetLastError(), e, ec)) return -1;
        }
        if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
        std::memcpy(output, st->host_buf + in_n, out_n * sizeof(float));
        return 0;
    }
    half *dw16 = nullptr;
    if (cached_weight_fp16(st, weight, w_n, &dw16, e, ec)) return -1;
    float *db = nullptr;
    if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
    if (ensure_scratch(st, in_n * sizeof(half), e, ec)) return -1;
    half *di16 = (half *)st->dev_scratch;
    size_t mapped_need = (in_n + out_n) * sizeof(float);
    if (ensure_host(st, mapped_need, e, ec)) return -1;
    float *d_out_mapped = st->dev_buf + in_n;
    std::memcpy(st->host_buf, input, in_n * sizeof(float));
    k_f32_to_f16<<<((int)in_n+255)/256, 256, 0, st->stream>>>(
        st->dev_buf, di16, (int)in_n);
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b0 = 0.0f;
    if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         (int)ow, (int)rows, (int)iw,
                         &a1, dw16, CUDA_R_16F, (int)iw,
                         di16, CUDA_R_16F, (int)iw,
                         &b0, d_out_mapped, CUDA_R_32F, (int)ow,
                         cuda_compute_type(st), cuda_gemm_algo(st)), e, ec)) return -1;
    if (db) {
        k_bias_add<<<((int)(rows*ow)+255)/256, 256, 0, st->stream>>>(
            d_out_mapped, db, (int)rows, (int)ow);
    }
    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    std::memcpy(output, st->host_buf + in_n, out_n * sizeof(float));
    return 0;
}

/* matmul to device buffer: same FP16 pipeline, no sync, no D2H.
 * Output stays on device at caller-provided d_out pointer. */
static int cuda_matmul_to_dev(void *opaque, const float *input, float *d_out,
                              size_t rows, size_t iw, size_t ow,
                              const float *weight, const float *bias,
                              char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t in_n = 0, out_n = 0, w_n = 0, total_n = 0;
    if (st == nullptr || validate_cuda_matmul(input, d_out, weight, rows, iw, ow,
                                               &in_n, &out_n, &w_n, &total_n,
                                               e, ec) != 0) return -1;
    if (!st->fast_math) {
        float *dw = nullptr;
        if (cached_weight(st, weight, w_n * sizeof(float), &dw, e, ec)) return -1;
        float *db = nullptr;
        if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
        if (ensure_host(st, in_n * sizeof(float), e, ec)) return -1;
        std::memcpy(st->host_buf, input, in_n * sizeof(float));
        cublasSetStream(st->cublas, st->stream);
        const float alpha = 1.0f, beta = 0.0f;
        if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             (int)ow, (int)rows, (int)iw,
                             &alpha, dw, CUDA_R_32F, (int)iw,
                             st->dev_buf, CUDA_R_32F, (int)iw,
                             &beta, d_out, CUDA_R_32F, (int)ow,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), e, ec)) return -1;
        if (db) {
            k_bias_add<<<((int)(rows * ow) + 255) / 256, 256, 0, st->stream>>>(
                d_out, db, (int)rows, (int)ow);
            if (ce(cudaGetLastError(), e, ec)) return -1;
        }
        return 0;
    }
    half *dw16 = nullptr;
    if (cached_weight_fp16(st, weight, w_n, &dw16, e, ec)) return -1;
    float *db = nullptr;
    if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
    if (ensure_scratch(st, in_n * sizeof(half), e, ec)) return -1;
    half *di16 = (half *)st->dev_scratch;
    if (ensure_host(st, in_n * sizeof(float), e, ec)) return -1;
    std::memcpy(st->host_buf, input, in_n * sizeof(float));
    k_f32_to_f16<<<((int)in_n+255)/256, 256, 0, st->stream>>>(st->dev_buf, di16, (int)in_n);
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b0 = 0.0f;
    if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         (int)ow, (int)rows, (int)iw,
                         &a1, dw16, CUDA_R_16F, (int)iw,
                         di16, CUDA_R_16F, (int)iw,
                         &b0, d_out, CUDA_R_32F, (int)ow,
                         cuda_compute_type(st), cuda_gemm_algo(st)), e, ec)) return -1;
    if (db) {
        k_bias_add<<<((int)(rows * ow) + 255) / 256, 256, 0, st->stream>>>(
            d_out, db, (int)rows, (int)ow);
        if (ce(cudaGetLastError(), e, ec)) return -1;
    }
    return 0; /* no sync */
}

extern "C" int mynah_cuda_matmul_to_dev(void *s, const float *in, float *dout,
                              size_t rows, size_t iw, size_t ow,
                              const float *w, const float *b,
                              char *e, size_t ec) {
    return cuda_matmul_to_dev(s, in, dout, rows, iw, ow, w, b, e, ec);
}

static int cuda_sgemm(void *opaque, int ta, int tb, size_t m, size_t n, size_t k,
                      float alpha, const float *a, size_t lda, const float *b, size_t ldb,
                      float beta, float *c, size_t ldc, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t ar = ta?k:m, ac = ta?m:k, br = tb?n:k, bc = tb?k:n;
    size_t ap = ar*ac, bp = br*bc, cp = m*n;
    size_t need = (ap+bp+cp)*sizeof(float);
    if (ensure_host(st, need, e, ec)) return -1;
    float *ha = st->host_buf, *hb = ha+ap, *hc = hb+bp;
    float *da = st->dev_buf,  *db2 = da+ap, *dc = db2+bp;
    /* Bulk copy when contiguous (common for im2col + weight). */
    if (lda == ac) std::memcpy(ha, a, ap*4);
    else for (size_t r=0;r<ar;r++) std::memcpy(ha+r*ac, a+r*lda, ac*4);
    if (ldb == bc) std::memcpy(hb, b, bp*4);
    else for (size_t r=0;r<br;r++) std::memcpy(hb+r*bc, b+r*ldb, bc*4);
    if (beta!=0.0f) {
        if (ldc == n) std::memcpy(hc, c, cp*4);
        else for (size_t r=0;r<m;r++) std::memcpy(hc+r*n, c+r*ldc, n*4);
    }
    cublasSetStream(st->cublas, st->stream);
    cublasOperation_t oa = tb?CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t ob = ta?CUBLAS_OP_T:CUBLAS_OP_N;
    if (cbe(cublasGemmEx(st->cublas,oa,ob,(int)n,(int)m,(int)k,&alpha,
                        db2,CUDA_R_32F,(int)bc,da,CUDA_R_32F,(int)ac,
                        &beta,dc,CUDA_R_32F,(int)n,
                        cuda_compute_type(st), cuda_gemm_algo(st)),e,ec)) return -1;
    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    if (ldc == n) std::memcpy(c, hc, cp*4);
    else for (size_t r=0;r<m;r++) std::memcpy(c+r*ldc, hc+r*n, n*4);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Device-side matmul: in/out are device pointers, no host copy.      */
/*  Weight/bias are host pointers (cached on device internally).       */
/*  Does NOT sync — caller calls mynah_backend_sync when needed.       */
/* ------------------------------------------------------------------ */

static int cuda_matmul_dev(void *opaque, const float *d_in, float *d_out,
                           size_t rows, size_t iw, size_t ow,
                           const float *weight, const float *bias,
                           char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t in_n = 0, out_n = 0, w_n = 0, total_n = 0;
    if (st == nullptr || validate_cuda_matmul(d_in, d_out, weight, rows, iw, ow,
                                               &in_n, &out_n, &w_n, &total_n,
                                               e, ec) != 0) return -1;
    float *dw = nullptr;
    if (cached_weight(st, weight, w_n * sizeof(float), &dw, e, ec)) return -1;
    float *db = nullptr;
    if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
    st->matmul_calls.fetch_add(1ull, std::memory_order_relaxed);
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b0 = 0.0f;
    if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         (int)ow, (int)rows, (int)iw,
                         &a1, dw, CUDA_R_32F, (int)iw,
                         d_in, CUDA_R_32F, (int)iw,
                         &b0, d_out, CUDA_R_32F, (int)ow,
                         cuda_compute_type(st), cuda_gemm_algo(st)), e, ec)) return -1;
    if (db) {
        k_bias_add<<<((int)(rows * ow) + 255) / 256, 256, 0, st->stream>>>(
            d_out, db, (int)rows, (int)ow);
        if (ce(cudaGetLastError(), e, ec)) return -1;
    }
    return 0; /* no sync */
}

/* Device-side sgemm: all pointers are device-side. No sync. */
static int cuda_sgemm_dev(void *opaque, int ta, int tb,
                          size_t m, size_t n, size_t k,
                          float alpha,
                          const float *d_a, size_t lda,
                          const float *d_b, size_t ldb,
                          float beta,
                          float *d_c, size_t ldc,
                          char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || d_a == nullptr || d_b == nullptr || d_c == nullptr ||
        (ta != 0 && ta != 1) || (tb != 0 && tb != 1) ||
        m == 0u || n == 0u || k == 0u) {
        set_error(e, ec, "invalid CUDA device sgemm dimensions");
        return -1;
    }
    cublasSetStream(st->cublas, st->stream);
    cublasOperation_t oa = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t ob = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    /* The row-major API is represented as C^T = B^T A^T in cuBLAS.  The
     * leading dimensions are therefore the caller's actual row strides, not
     * the packed shape columns.  Ignoring them silently corrupts padded and
     * strided batched projections. */
    const size_t a_cols = ta ? m : k;
    const size_t b_cols = tb ? k : n;
    if (lda < a_cols || ldb < b_cols || ldc < n ||
        m > (size_t)INT_MAX || n > (size_t)INT_MAX || k > (size_t)INT_MAX ||
        lda > (size_t)INT_MAX || ldb > (size_t)INT_MAX || ldc > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA strided sgemm dimensions");
        return -1;
    }
    return cbe(cublasGemmEx(st->cublas, oa, ob,
                           (int)n, (int)m, (int)k, &alpha,
                           d_b, CUDA_R_32F, (int)ldb,
                           d_a, CUDA_R_32F, (int)lda,
                           &beta, d_c, CUDA_R_32F, (int)ldc,
                           cuda_compute_type(st), cuda_gemm_algo(st)), e, ec) ? -1 : 0;
}

extern "C" int mynah_cuda_matmul_dev(void *s, const float *di, float *dout,
                           size_t rows, size_t iw, size_t ow,
                           const float *w, const float *b,
                           char *e, size_t ec) {
    return cuda_matmul_dev(s, di, dout, rows, iw, ow, w, b, e, ec);
}

extern "C" int mynah_cuda_sgemm_dev(void *s, int ta, int tb,
                          size_t m, size_t n, size_t k, float alpha,
                          const float *da, size_t lda,
                          const float *db, size_t ldb,
                          float beta, float *dc, size_t ldc,
                          char *e, size_t ec) {
    return cuda_sgemm_dev(s, ta, tb, m, n, k, alpha, da, lda, db, ldb, beta, dc, ldc, e, ec);
}

/* ------------------------------------------------------------------ */
/*  Device-side element-wise ops (CUDA implementations)                */
/* ------------------------------------------------------------------ */

extern "C" int mynah_cuda_upload(void *opaque, const float *host, size_t n,
                       float **dev_ptr, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t bytes = n * sizeof(float);
    if (ensure_scratch(st, bytes, e, ec)) return -1;
    *dev_ptr = st->dev_scratch;
    st->h2d_calls.fetch_add(1ull, std::memory_order_relaxed);
    st->h2d_bytes.fetch_add((unsigned long long)bytes, std::memory_order_relaxed);
    return ce(cudaMemcpyAsync(*dev_ptr, host, bytes, cudaMemcpyHostToDevice, st->stream), e, ec);
}

extern "C" int mynah_cuda_download(void *opaque, const float *dev_ptr, float *host,
                         size_t n, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    st->d2h_calls.fetch_add(1ull, std::memory_order_relaxed);
    st->d2h_bytes.fetch_add((unsigned long long)(n * sizeof(float)),
                            std::memory_order_relaxed);
    return ce(cudaMemcpyAsync(host, dev_ptr, n * sizeof(float),
                              cudaMemcpyDeviceToHost, st->stream), e, ec);
}

extern "C" int mynah_cuda_sync(void *opaque, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    st->sync_calls.fetch_add(1ull, std::memory_order_relaxed);
    return ce(cudaStreamSynchronize(st->stream), e, ec);
}

extern "C" int mynah_cuda_batch_begin(void *opaque, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    return cbe(cublasSetStream(st->cublas, st->stream), e, ec);
}

extern "C" int mynah_cuda_snake_dev(void *opaque, float *dev_data, const float *alpha,
                          size_t channels, size_t length, size_t snake_ch,
                          char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (channels == 0 || length == 0 || snake_ch > channels ||
        channels > (size_t)INT_MAX || length > (size_t)INT_MAX ||
        snake_ch > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA Snake dimensions");
        return -1;
    }
    float *d_alpha = nullptr;
    if (snake_ch > 0u &&
        cached_weight(st, alpha, snake_ch * sizeof(float), &d_alpha, e, ec)) return -1;
    k_snake<<<(int)channels, 256, 0, st->stream>>>(
        dev_data, d_alpha, (int)channels, (int)length, (int)snake_ch);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_gelu_dev(void *opaque, float *data, size_t n,
                         char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (data == nullptr || n == 0 || n > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA GELU dimensions");
        return -1;
    }
    k_gelu<<<((int)n + 255) / 256, 256, 0, st->stream>>>(data, (int)n);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    return 0;
}

extern "C" int mynah_cuda_layer_norm_dev(void *opaque, const float *in, float *out,
                               const float *gain, const float *bias,
                               size_t rows, size_t width,
                               char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (in == nullptr || out == nullptr || gain == nullptr || rows == 0 || width == 0 ||
        rows > (size_t)INT_MAX || width > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA layer-norm dimensions");
        return -1;
    }
    float *d_gain = nullptr;
    float *d_bias = nullptr;
    if (cached_weight(st, gain, width * sizeof(float), &d_gain, e, ec)) return -1;
    if (bias != nullptr && cached_weight(st, bias, width * sizeof(float), &d_bias, e, ec)) return -1;
    k_layer_norm<<<(int)rows, 256, 0, st->stream>>>(
        out, in, d_gain, d_bias, (int)width, 1e-5f, (int)rows);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    return 0;
}

extern "C" int mynah_cuda_residual_add_dev(void *opaque, float *out, const float *in,
                                 size_t n, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (out == nullptr || in == nullptr || n == 0 || n > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA residual dimensions");
        return -1;
    }
    k_residual_add<<<((int)n + 255) / 256, 256, 0, st->stream>>>(out, in, (int)n);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    return 0;
}

extern "C" int mynah_cuda_scaled_residual_add_dev(
    void *opaque, float *out, const float *in, const float *scale,
    size_t n, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || out == nullptr || in == nullptr || scale == nullptr ||
        n == 0u || n > (size_t)INT_MAX || n > SIZE_MAX / sizeof(float)) {
        set_error(e, ec, "invalid CUDA scaled residual dimensions");
        return -1;
    }
    float *d_scale = nullptr;
    if (cached_weight(st, scale, n * sizeof(float), &d_scale, e, ec)) return -1;
    k_scaled_residual_add<<<((int)n + 255) / 256, 256, 0, st->stream>>>(
        out, in, d_scale, (int)n);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    return 0;
}

extern "C" int mynah_cuda_scaled_residual_rows_dev(
    void *opaque, float *out, const float *in, const float *scale,
    size_t rows, size_t width, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t total = 0u;
    if (st == nullptr || out == nullptr || in == nullptr || scale == nullptr ||
        rows == 0u || width == 0u || rows > (size_t)INT_MAX ||
        width > (size_t)INT_MAX || !cuda_size_mul(rows, width, &total) ||
        total > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA row-wise scaled residual dimensions");
        return -1;
    }
    float *d_scale = nullptr;
    if (cached_weight(st, scale, width * sizeof(float), &d_scale, e, ec)) return -1;
    k_scaled_residual_rows_add<<<((int)total + 255) / 256, 256, 0, st->stream>>>(
        out, in, d_scale, (int)rows, (int)width);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    return 0;
}

extern "C" int mynah_cuda_copy_dev(void *opaque, float *dst, const float *src,
                                    size_t n, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (dst == nullptr || src == nullptr || n == 0 || n > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA copy dimensions");
        return -1;
    }
    k_copy<<<((int)n + 255) / 256, 256, 0, st->stream>>>(dst, src, (int)n);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_copy_dev_bytes(void *opaque, void *dst,
                                         const void *src, size_t bytes,
                                         char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || dst == nullptr || src == nullptr) {
        set_error(e, ec, "invalid CUDA byte copy");
        return -1;
    }
    if (bytes == 0u) return 0;
    return ce(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                              st->stream), e, ec);
}

/* Strided device-to-device copy of `rows` rows of `width` bytes: row r goes
 * from src + r * src_pitch to dst + r * dst_pitch. Stream-ordered. */
extern "C" int mynah_cuda_copy_dev_bytes_2d(void *opaque, void *dst,
                                            size_t dst_pitch, const void *src,
                                            size_t src_pitch, size_t width,
                                            size_t rows, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || dst == nullptr || src == nullptr ||
        dst_pitch < width || src_pitch < width) {
        set_error(e, ec, "invalid CUDA strided byte copy");
        return -1;
    }
    if (width == 0u || rows == 0u) return 0;
    return ce(cudaMemcpy2DAsync(dst, dst_pitch, src, src_pitch, width, rows,
                                cudaMemcpyDeviceToDevice, st->stream), e, ec);
}

extern "C" int mynah_cuda_scale_dev(void *opaque, float *data, size_t n,
                                     float scale, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (data == nullptr || n == 0 || n > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA scale dimensions");
        return -1;
    }
    k_scale<<<((int)n + 255) / 256, 256, 0, st->stream>>>(data, scale, (int)n);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_clip_dev(void *opaque, float *data, size_t n,
                                    char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (data == nullptr || n == 0 || n > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA clip dimensions");
        return -1;
    }
    k_clip<<<((int)n + 255) / 256, 256, 0, st->stream>>>(data, (int)n);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_argmax_dev(void *opaque, const float *logits,
                                      size_t vocab, size_t codebook_size,
                                      size_t eos_id, int allow_eos,
                                      unsigned *argmax, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (logits == nullptr || argmax == nullptr || vocab == 0 ||
        codebook_size > vocab || vocab > (size_t)INT_MAX ||
        codebook_size > (size_t)INT_MAX || eos_id > (size_t)UINT_MAX) {
        set_error(e, ec, "invalid CUDA argmax dimensions");
        return -1;
    }
    if (st->dev_argmax == nullptr &&
        ce(cudaMalloc(&st->dev_argmax, sizeof(unsigned)), e, ec)) return -1;
    k_argmax<<<1, 1, 0, st->stream>>>(logits, st->dev_argmax,
                                       (int)vocab, (int)codebook_size,
                                       (unsigned)eos_id, allow_eos != 0);
    if (ce(cudaGetLastError(), e, ec) ||
        ce(cudaMemcpyAsync(argmax, st->dev_argmax, sizeof(unsigned),
                           cudaMemcpyDeviceToHost, st->stream), e, ec) ||
        ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    return 0;
}

extern "C" int mynah_cuda_softmax_dev(void *opaque, float *data,
                                       size_t rows, size_t cols, size_t valid,
                                       char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (data == nullptr || rows == 0 || cols == 0 || rows > (size_t)INT_MAX ||
        cols > (size_t)INT_MAX || valid > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA softmax dimensions");
        return -1;
    }
    const int v = valid > (size_t)INT_MAX ? INT_MAX : (int)valid;
    k_softmax_causal<<<(int)rows, 256, 0, st->stream>>>(
        data, (int)cols, v);
    return ce(cudaGetLastError(), e, ec);
}

/* Forward declarations for the attention/RoPE entry points below the
 * model-free self-test. */
extern "C" int mynah_cuda_self_attention_dev(
    void *, const float *, float *, float *, size_t, size_t, size_t, size_t,
    size_t, float, float *, char *, size_t);
extern "C" int mynah_cuda_self_attention_batch_dev(
    void *, const float *, float *const *, float *const *, const size_t *,
    const size_t *, size_t, size_t, size_t, float, float *, char *, size_t);
extern "C" int mynah_cuda_gather_kv_batch(
    void *, float *const *, float *const *, const size_t *, const size_t *,
    size_t, size_t, size_t, float *, char *, size_t);
extern "C" int mynah_cuda_rope_dev(void *, float *, size_t, size_t, size_t,
                                    float, char *, size_t);
extern "C" int mynah_cuda_rope_batch_dev(
    void *, float *, const size_t *, size_t, size_t, size_t, float, char *,
    size_t);
extern "C" int mynah_cuda_conv1d_dev(void *, const float *, float *, int, int,
                                      int, int, int, const float *, const float *,
                                      char *, size_t);
extern "C" int mynah_cuda_conv_transpose_dev(void *, const float *, float *, int,
                                              int, int, int, int, int, int,
                                              const float *, const float *, char *,
                                              size_t);
extern "C" int mynah_cuda_conv_transpose_causal_step_dev(
    void *, const float *, float *, float *, int, int, int, const float *,
    const float *, char *, size_t);
extern "C" int mynah_cuda_scatter_row_to_channels_dev(
    void *, const float *, float *, size_t, size_t, size_t, char *, size_t);
extern "C" int mynah_cuda_scatter_rows_to_channels_dev(
    void *, const float *, float *const *, size_t, size_t, size_t, size_t,
    char *, size_t);
extern "C" int mynah_cuda_gather_rows_to_batch_dev(
    void *, float *const *, float *, size_t, size_t, char *, size_t);
extern "C" int mynah_cuda_zero_dev(void *, float *, size_t, char *, size_t);
static int cuda_decoder_self_test(void *opaque, char *e, size_t ec);
extern "C" int mynah_cuda_decoder_open(
    void *, const mynah_backend_decoder_desc *, size_t,
    mynah_backend_decoder **, char *, size_t);
extern "C" void mynah_cuda_decoder_close(void *, mynah_backend_decoder *);
extern "C" int mynah_cuda_decoder_reset(void *, mynah_backend_decoder *,
                                          char *, size_t);
extern "C" int mynah_cuda_decoder_step(void *, mynah_backend_decoder *,
                                         const float *, size_t, float *, char *,
                                         size_t);
extern "C" int mynah_cuda_decoder_step_batch(
    void *, mynah_backend_decoder *const *, const float *const *, size_t,
    size_t, float *const *, char *, size_t);

static int cuda_resident_kernel_self_test(void *opaque, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    float *d_in = nullptr;
    float *d_out = nullptr;
    float *d_k0 = nullptr;
    float *d_v0 = nullptr;
    float *d_k1 = nullptr;
    float *d_v1 = nullptr;
    const float norm_in[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                              -1.0f, 0.0f, 1.0f, 2.0f};
    const float gain[4] = {1.0f, 2.0f, 1.0f, 0.5f};
    const float bias[4] = {0.1f, -0.2f, 0.3f, -0.4f};
    float norm_out[8] = {0.0f};
    const float scaled_rows_in[8] = {1.0f, -2.0f, 3.0f, -4.0f,
                                     5.0f, -6.0f, 7.0f, -8.0f};
    const float scaled_rows_scale[4] = {0.5f, -1.0f, 2.0f, 0.25f};
    float scaled_rows_out[8] = {0.0f};
    const float softmax_in[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                                 0.0f, -1.0f, 2.0f, 9.0f};
    float softmax_out[8] = {0.0f};
    float qkv[24] = {0.0f};
    float attention_out[8] = {0.0f};
    float gathered_kv[16] = {0.0f};
    const float conv_input[3] = {1.0f, 2.0f, 3.0f};
    const float conv_weight[2] = {2.0f, -1.0f};
    const float conv_bias[1] = {0.5f};
    const float conv_expected[3] = {-0.5f, 0.5f, 1.5f};
    float conv_output[4] = {0.0f};
    const float convt_input[2] = {1.0f, 2.0f};
    const float convt_weight[2] = {2.0f, 3.0f};
    const float convt_expected[4] = {2.5f, 3.5f, 4.5f, 6.5f};
    float convt_output[4] = {0.0f};
    const float causal_step_weight[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float causal_step_input[1] = {2.0f};
    const float causal_step_input_next[1] = {3.0f};
    const float causal_step_expected[2] = {2.5f, 4.5f};
    const float causal_step_expected_next[2] = {9.5f, 14.5f};
    float causal_step_output[2] = {0.0f};
    float causal_step_partial[2] = {0.0f};
    /* Also cover tail < stride.  The production Mimi shape is kernel ==
     * 2*stride, but the backend primitive is intentionally generic and used
     * to index a negative partial slot for this valid causal shape. */
    const float causal_short_weight[3] = {1.0f, 2.0f, 3.0f};
    const float causal_short_expected[2] = {2.5f, 4.5f};
    const float causal_short_expected_next[2] = {9.5f, 6.5f};
    float causal_short_output[2] = {0.0f};
    float causal_short_partial[1] = {0.0f};
    /* Cover tail > stride as well.  This shape makes the carried prefix
     * overlap the newly produced tail, which is exactly where an in-place
     * parallel update can otherwise observe the wrong generation of state. */
    const float causal_long_weight[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    const float causal_long_expected[2] = {2.5f, 4.5f};
    const float causal_long_expected_next[2] = {9.5f, 14.5f};
    float causal_long_output[2] = {0.0f};
    float causal_long_partial[3] = {0.0f};
    const float scatter_rows[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                                   5.0f, 6.0f, 7.0f, 8.0f};
    float scatter_output[16] = {0.0f};
    float *scatter_dest[2] = {nullptr, nullptr};
    const float gather_input_a[4] = {9.0f, 8.0f, 7.0f, 6.0f};
    const float gather_input_b[4] = {5.0f, 4.0f, 3.0f, 2.0f};
    float gather_output[8] = {0.0f};
    float *gather_inputs[2] = {nullptr, nullptr};
    float *kcache[2] = {nullptr, nullptr};
    float *vcache[2] = {nullptr, nullptr};
    size_t positions[2] = {0u, 0u};
    size_t strides[2] = {4u, 4u};
    size_t rope_positions[2] = {7u, 9u};
    const float one_qkv[12] = {1.0f, 0.0f, 0.0f, 1.0f,
                               1.0f, 0.0f, 0.0f, 1.0f,
                               1.0f, 2.0f, 3.0f, 4.0f};
    const float batch_qkv[24] = {
        1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 3.0f, 4.0f,
        1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    const float rope_slope = (float)(-std::log(10000.0) * 2.0 / 4.0);

    if (ce(cudaMalloc(&d_in, 24u * sizeof(float)), e, ec) ||
        ce(cudaMalloc(&d_out, 24u * sizeof(float)), e, ec) ||
        ce(cudaMalloc(&d_k0, 4u * sizeof(float)), e, ec) ||
        ce(cudaMalloc(&d_v0, 4u * sizeof(float)), e, ec) ||
        ce(cudaMalloc(&d_k1, 4u * sizeof(float)), e, ec) ||
        ce(cudaMalloc(&d_v1, 4u * sizeof(float)), e, ec)) goto fail;

    if (ce(cudaMemcpy(d_in, norm_in, sizeof(norm_in), cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_layer_norm_dev(st, d_in, d_out, gain, bias, 2u, 4u, e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(norm_out, d_out, sizeof(norm_out), cudaMemcpyDeviceToHost), e, ec))
        goto fail;
    for (size_t r = 0; r < 2u; ++r) {
        float mean = 0.0f;
        for (size_t d = 0; d < 4u; ++d) mean += norm_in[r * 4u + d];
        mean /= 4.0f;
        float variance = 0.0f;
        for (size_t d = 0; d < 4u; ++d) {
            const float delta = norm_in[r * 4u + d] - mean;
            variance += delta * delta;
        }
        const float inv = 1.0f / sqrtf(variance / 4.0f + 1.0e-5f);
        for (size_t d = 0; d < 4u; ++d) {
            const float expected = (norm_in[r * 4u + d] - mean) * inv * gain[d] + bias[d];
            if (fabsf(norm_out[r * 4u + d] - expected) > 2.0e-3f) {
                std::snprintf(e, ec, "CUDA layer-norm self-test mismatch at %zu", r * 4u + d);
                goto fail;
            }
        }
    }

    if (ce(cudaMemcpy(d_in, scaled_rows_in, sizeof(scaled_rows_in),
                      cudaMemcpyHostToDevice), e, ec) ||
        ce(cudaMemcpy(d_out, scaled_rows_in, sizeof(scaled_rows_in),
                      cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_scaled_residual_rows_dev(st, d_out, d_in,
                                             scaled_rows_scale, 2u, 4u,
                                             e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(scaled_rows_out, d_out, sizeof(scaled_rows_out),
                      cudaMemcpyDeviceToHost), e, ec)) goto fail;
    for (size_t i = 0; i < 8u; ++i) {
        const float expected = scaled_rows_in[i] +
                               scaled_rows_scale[i % 4u] * scaled_rows_in[i];
        if (fabsf(scaled_rows_out[i] - expected) > 2.0e-4f) {
            std::snprintf(e, ec,
                          "CUDA row-wise LayerScale self-test mismatch at %zu", i);
            goto fail;
        }
    }

    if (ce(cudaMemcpy(d_in, softmax_in, sizeof(softmax_in), cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_softmax_dev(st, d_in, 2u, 4u, 3u, e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(softmax_out, d_in, sizeof(softmax_out), cudaMemcpyDeviceToHost), e, ec))
        goto fail;
    for (size_t r = 0; r < 2u; ++r) {
        float sum = 0.0f;
        for (size_t d = 0; d < 3u; ++d) sum += softmax_out[r * 4u + d];
        if (fabsf(sum - 1.0f) > 2.0e-4f || softmax_out[r * 4u + 3u] != 0.0f) {
            std::snprintf(e, ec, "CUDA softmax self-test mismatch at row %zu", r);
            goto fail;
        }
    }

    if (ce(cudaMemcpy(d_in, conv_input, sizeof(conv_input), cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_conv1d_dev(st, d_in, d_out, 1, 1, 3, 2, 1, conv_weight,
                              conv_bias, e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(conv_output, d_out, sizeof(conv_output), cudaMemcpyDeviceToHost),
           e, ec)) goto fail;
    for (size_t i = 0; i < 3u; ++i) {
        if (fabsf(conv_output[i] - conv_expected[i]) > 2.0e-4f) {
            std::snprintf(e, ec, "CUDA causal-conv self-test mismatch at %zu", i);
            goto fail;
        }
    }
    if (ce(cudaMemcpy(d_in, convt_input, sizeof(convt_input), cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_conv_transpose_dev(st, d_in, d_out, 1, 1, 2, 4, 2, 2, 1,
                                      convt_weight, conv_bias, e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(convt_output, d_out, sizeof(convt_output), cudaMemcpyDeviceToHost),
           e, ec)) goto fail;
    for (size_t i = 0; i < 4u; ++i) {
        if (fabsf(convt_output[i] - convt_expected[i]) > 2.0e-4f) {
            std::snprintf(e, ec, "CUDA transpose-conv self-test mismatch at %zu", i);
            goto fail;
        }
    }
    if (mynah_cuda_zero_dev(st, d_k0, 2u, e, ec) != 0 ||
        ce(cudaMemcpy(d_in, causal_step_input, sizeof(causal_step_input),
                      cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_conv_transpose_causal_step_dev(
            st, d_in, d_out, d_k0, 1, 4, 2, causal_step_weight, conv_bias, e,
            ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(causal_step_output, d_out, sizeof(causal_step_output),
                      cudaMemcpyDeviceToHost), e, ec) ||
        ce(cudaMemcpy(causal_step_partial, d_k0, sizeof(causal_step_partial),
                      cudaMemcpyDeviceToHost), e, ec)) goto fail;
    for (size_t i = 0; i < 2u; ++i) {
        if (fabsf(causal_step_output[i] - causal_step_expected[i]) > 2.0e-4f ||
            fabsf(causal_step_partial[i] - (i == 0u ? 6.0f : 8.0f)) > 2.0e-4f) {
            std::snprintf(e, ec, "CUDA causal transpose first-step mismatch at %zu", i);
            goto fail;
        }
    }
    if (ce(cudaMemcpy(d_in, causal_step_input_next, sizeof(causal_step_input_next),
                      cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_conv_transpose_causal_step_dev(
            st, d_in, d_out, d_k0, 1, 4, 2, causal_step_weight, conv_bias, e,
            ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(causal_step_output, d_out, sizeof(causal_step_output),
                      cudaMemcpyDeviceToHost), e, ec)) goto fail;
    for (size_t i = 0; i < 2u; ++i) {
        if (fabsf(causal_step_output[i] - causal_step_expected_next[i]) > 2.0e-4f) {
            std::snprintf(e, ec, "CUDA causal transpose second-step mismatch at %zu", i);
            goto fail;
        }
    }
    if (mynah_cuda_zero_dev(st, d_k1, 1u, e, ec) != 0 ||
        ce(cudaMemcpy(d_in, causal_step_input, sizeof(causal_step_input),
                      cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_conv_transpose_causal_step_dev(
            st, d_in, d_out, d_k1, 1, 3, 2, causal_short_weight, conv_bias,
            e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(causal_short_output, d_out,
                      sizeof(causal_short_output), cudaMemcpyDeviceToHost),
           e, ec) ||
        ce(cudaMemcpy(causal_short_partial, d_k1,
                      sizeof(causal_short_partial), cudaMemcpyDeviceToHost),
           e, ec)) goto fail;
    for (size_t i = 0; i < 2u; ++i) {
        if (fabsf(causal_short_output[i] - causal_short_expected[i]) > 2.0e-4f ||
            (i == 0u && fabsf(causal_short_partial[0] - 6.0f) > 2.0e-4f)) {
            std::snprintf(e, ec,
                          "CUDA short-tail causal transpose first-step mismatch at %zu",
                          i);
            goto fail;
        }
    }
    if (ce(cudaMemcpy(d_in, causal_step_input_next, sizeof(causal_step_input_next),
                      cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_conv_transpose_causal_step_dev(
            st, d_in, d_out, d_k1, 1, 3, 2, causal_short_weight, conv_bias,
            e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(causal_short_output, d_out,
                      sizeof(causal_short_output), cudaMemcpyDeviceToHost),
           e, ec)) goto fail;
    for (size_t i = 0; i < 2u; ++i) {
        if (fabsf(causal_short_output[i] - causal_short_expected_next[i]) >
            2.0e-4f) {
            std::snprintf(e, ec,
                          "CUDA short-tail causal transpose second-step mismatch at %zu",
                          i);
            goto fail;
        }
    }
    if (mynah_cuda_zero_dev(st, d_k1, 3u, e, ec) != 0 ||
        ce(cudaMemcpy(d_in, causal_step_input, sizeof(causal_step_input),
                      cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_conv_transpose_causal_step_dev(
            st, d_in, d_out, d_k1, 1, 5, 2, causal_long_weight, conv_bias,
            e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(causal_long_output, d_out,
                      sizeof(causal_long_output), cudaMemcpyDeviceToHost),
           e, ec) ||
        ce(cudaMemcpy(causal_long_partial, d_k1,
                      sizeof(causal_long_partial), cudaMemcpyDeviceToHost),
           e, ec)) goto fail;
    for (size_t i = 0; i < 2u; ++i) {
        if (fabsf(causal_long_output[i] - causal_long_expected[i]) >
                2.0e-4f ||
            fabsf(causal_long_partial[i] -
                  (i == 0u ? 6.0f : i == 1u ? 8.0f : 10.0f)) > 2.0e-4f) {
            std::snprintf(e, ec,
                          "CUDA long-tail causal transpose first-step mismatch at %zu",
                          i);
            goto fail;
        }
    }
    if (fabsf(causal_long_partial[2] - 10.0f) > 2.0e-4f) {
        std::snprintf(e, ec,
                      "CUDA long-tail causal transpose first tail mismatch");
        goto fail;
    }
    if (ce(cudaMemcpy(d_in, causal_step_input_next, sizeof(causal_step_input_next),
                      cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_conv_transpose_causal_step_dev(
            st, d_in, d_out, d_k1, 1, 5, 2, causal_long_weight, conv_bias,
            e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(causal_long_output, d_out,
                      sizeof(causal_long_output), cudaMemcpyDeviceToHost),
           e, ec) ||
        ce(cudaMemcpy(causal_long_partial, d_k1,
                      sizeof(causal_long_partial), cudaMemcpyDeviceToHost),
           e, ec)) goto fail;
    if (fabsf(causal_long_output[0] - causal_long_expected_next[0]) >
            2.0e-4f ||
        fabsf(causal_long_output[1] - causal_long_expected_next[1]) >
            2.0e-4f ||
        fabsf(causal_long_partial[0] - 19.0f) > 2.0e-4f ||
        fabsf(causal_long_partial[1] - 12.0f) > 2.0e-4f ||
        fabsf(causal_long_partial[2] - 15.0f) > 2.0e-4f) {
        std::snprintf(e, ec,
                      "CUDA long-tail causal transpose second-step mismatch");
        goto fail;
    }
    if (ce(cudaMemcpy(d_in, scatter_rows, sizeof(scatter_rows),
                      cudaMemcpyHostToDevice), e, ec) ||
        ce(cudaMemset(d_out, 0, 16u * sizeof(float)), e, ec) ||
        mynah_cuda_scatter_row_to_channels_dev(st, d_in, d_out, 4u, 2u, 1u,
                                               e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(scatter_output, d_out, 8u * sizeof(float),
                      cudaMemcpyDeviceToHost), e, ec)) goto fail;
    for (size_t c = 0; c < 4u; ++c) {
        if (scatter_output[c * 2u] != 0.0f ||
            fabsf(scatter_output[c * 2u + 1u] - scatter_rows[c]) > 2.0e-4f) {
            std::snprintf(e, ec, "CUDA row scatter self-test mismatch at %zu", c);
            goto fail;
        }
    }
    scatter_dest[0] = d_out;
    scatter_dest[1] = d_out + 8u;
    if (ce(cudaMemset(d_out, 0, 16u * sizeof(float)), e, ec) ||
        mynah_cuda_scatter_rows_to_channels_dev(st, d_in, scatter_dest, 2u,
                                                4u, 2u, 1u, e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(scatter_output, d_out, 16u * sizeof(float),
                      cudaMemcpyDeviceToHost), e, ec)) goto fail;
    for (size_t request = 0; request < 2u; ++request) {
        for (size_t c = 0; c < 4u; ++c) {
            const size_t at = request * 8u + c * 2u;
            if (scatter_output[at] != 0.0f ||
                fabsf(scatter_output[at + 1u] -
                      scatter_rows[request * 4u + c]) > 2.0e-4f) {
                std::snprintf(e, ec,
                              "CUDA batched row scatter self-test mismatch at %zu",
                              request * 4u + c);
                goto fail;
            }
        }
    }
    gather_inputs[0] = d_k0;
    gather_inputs[1] = d_k1;
    if (ce(cudaMemcpy(d_k0, gather_input_a, sizeof(gather_input_a),
                      cudaMemcpyHostToDevice), e, ec) ||
        ce(cudaMemcpy(d_k1, gather_input_b, sizeof(gather_input_b),
                      cudaMemcpyHostToDevice), e, ec) ||
        ce(cudaMemset(d_out, 0, sizeof(gather_output)), e, ec) ||
        mynah_cuda_gather_rows_to_batch_dev(st, gather_inputs, d_out, 2u, 4u,
                                            e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(gather_output, d_out, sizeof(gather_output),
                      cudaMemcpyDeviceToHost), e, ec)) goto fail;
    for (size_t i = 0; i < 4u; ++i) {
        if (gather_output[i] != gather_input_a[i] ||
            gather_output[4u + i] != gather_input_b[i]) {
            std::snprintf(e, ec, "CUDA row gather self-test mismatch at %zu", i);
            goto fail;
        }
    }

    /* RoPE at a non-zero position preserves each complex pair's norm. */
    for (size_t i = 0; i < 24u; ++i) qkv[i] = (float)(i + 1u) * 0.125f;
    if (ce(cudaMemcpy(d_in, qkv, 12u * sizeof(float), cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_rope_dev(st, d_in, 7u, 1u, 4u, 10000.0f, e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(qkv, d_in, 12u * sizeof(float), cudaMemcpyDeviceToHost), e, ec))
        goto fail;
    for (size_t base = 0; base < 4u; base += 2u) {
        const size_t pair = base / 2u;
        const float q0 = (float)(base + 1u) * 0.125f;
        const float q1 = (float)(base + 2u) * 0.125f;
        const float k0 = (float)(4u + base + 1u) * 0.125f;
        const float k1 = (float)(4u + base + 2u) * 0.125f;
        const float frequency = expf((float)pair * rope_slope);
        const float angle = 7.0f * frequency;
        const float sine = sinf(angle);
        const float cosine = cosf(angle);
        const float expected0 = q0 * cosine - q1 * sine;
        const float expected1 = q0 * sine + q1 * cosine;
        const float expected_k0 = k0 * cosine - k1 * sine;
        const float expected_k1 = k0 * sine + k1 * cosine;
        const float before = q0 * q0 + q1 * q1;
        const float before_k = k0 * k0 + k1 * k1;
        const float after = qkv[base] * qkv[base] + qkv[base + 1u] * qkv[base + 1u];
        const float after_k = qkv[4u + base] * qkv[4u + base] +
                              qkv[4u + base + 1u] * qkv[4u + base + 1u];
        if (fabsf(qkv[base] - expected0) > 3.0e-3f ||
            fabsf(qkv[base + 1u] - expected1) > 3.0e-3f ||
            fabsf(qkv[4u + base] - expected_k0) > 3.0e-3f ||
            fabsf(qkv[4u + base + 1u] - expected_k1) > 3.0e-3f ||
            fabsf(before - after) > 2.0e-3f ||
            fabsf(before_k - after_k) > 2.0e-3f) {
            std::snprintf(e, ec, "CUDA RoPE self-test mismatch at pair %zu", base / 2u);
            goto fail;
        }
    }

    for (size_t i = 0; i < 24u; ++i) qkv[i] = (float)(i + 1u) * 0.125f;
    if (ce(cudaMemcpy(d_in, qkv, 24u * sizeof(float), cudaMemcpyHostToDevice),
           e, ec) ||
        mynah_cuda_rope_batch_dev(st, d_in, rope_positions, 2u, 1u, 4u,
                                  10000.0f, e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(qkv, d_in, 24u * sizeof(float), cudaMemcpyDeviceToHost),
           e, ec)) goto fail;
    for (size_t request = 0; request < 2u; ++request) {
        const size_t row = request * 12u;
        for (size_t base = 0; base < 8u; base += 2u) {
            const float q0 = (float)(row + base + 1u) * 0.125f;
            const float q1 = (float)(row + base + 2u) * 0.125f;
            const float before_q = q0 * q0 + q1 * q1;
            const float after_q = qkv[row + base] * qkv[row + base] +
                                  qkv[row + base + 1u] * qkv[row + base + 1u];
            const float k0 = (float)(row + base + 5u) * 0.125f;
            const float k1 = (float)(row + base + 6u) * 0.125f;
            const float before_k = k0 * k0 + k1 * k1;
            const float after_k = qkv[row + 4u + base] * qkv[row + 4u + base] +
                                  qkv[row + 4u + base + 1u] *
                                  qkv[row + 4u + base + 1u];
            if (fabsf(before_q - after_q) > 2.0e-3f ||
                fabsf(before_k - after_k) > 2.0e-3f) {
                std::snprintf(e, ec,
                              "CUDA batched RoPE self-test mismatch at request %zu",
                              request);
                goto fail;
            }
        }
    }

    /* Single and independent-request batched attention both reduce to the
     * only valid value at position zero. */
    if (ce(cudaMemcpy(d_in, one_qkv, sizeof(one_qkv), cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_self_attention_dev(st, d_in, d_k0, d_v0, 0u, 4u, 1u, 1u, 4u,
                                       0.5f, d_out, e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(attention_out, d_out, 4u * sizeof(float), cudaMemcpyDeviceToHost), e, ec))
        goto fail;
    for (size_t d = 0; d < 4u; ++d) {
        if (fabsf(attention_out[d] - one_qkv[8u + d]) > 2.0e-4f) {
            std::snprintf(e, ec, "CUDA attention self-test mismatch at %zu", d);
            goto fail;
        }
    }
    kcache[0] = d_k0; kcache[1] = d_k1;
    vcache[0] = d_v0; vcache[1] = d_v1;
    if (ce(cudaMemcpy(d_in, batch_qkv, sizeof(batch_qkv), cudaMemcpyHostToDevice), e, ec) ||
        mynah_cuda_self_attention_batch_dev(st, d_in, kcache, vcache, positions,
                                             strides, 2u, 1u, 4u, 0.5f, d_out,
                                             e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(attention_out, d_out, 8u * sizeof(float), cudaMemcpyDeviceToHost), e, ec))
        goto fail;
    for (size_t d = 0; d < 4u; ++d) {
        if (fabsf(attention_out[d] - batch_qkv[8u + d]) > 2.0e-4f ||
            fabsf(attention_out[4u + d] - batch_qkv[20u + d]) > 2.0e-4f) {
            std::snprintf(e, ec, "CUDA batched attention self-test mismatch at %zu", d);
            goto fail;
        }
    }
    if (mynah_cuda_gather_kv_batch(st, kcache, vcache, positions, strides,
                                   2u, 1u, 4u, d_out, e, ec) != 0 ||
        mynah_cuda_sync(st, e, ec) != 0 ||
        ce(cudaMemcpy(gathered_kv, d_out, sizeof(gathered_kv),
                      cudaMemcpyDeviceToHost), e, ec)) goto fail;
    for (size_t d = 0; d < 4u; ++d) {
        if (fabsf(gathered_kv[d] - batch_qkv[4u + d]) > 2.0e-4f ||
            fabsf(gathered_kv[4u + d] - batch_qkv[8u + d]) > 2.0e-4f ||
            fabsf(gathered_kv[8u + d] - batch_qkv[16u + d]) > 2.0e-4f ||
            fabsf(gathered_kv[12u + d] - batch_qkv[20u + d]) > 2.0e-4f) {
            std::snprintf(e, ec, "CUDA K/V gather self-test mismatch at %zu", d);
            goto fail;
        }
    }
    if (cuda_decoder_self_test(st, e, ec) != 0) goto fail;
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_k0); cudaFree(d_v0);
    cudaFree(d_k1); cudaFree(d_v1);
    return 0;

fail:
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_k0); cudaFree(d_v0);
    cudaFree(d_k1); cudaFree(d_v1);
    return -1;
}

/* ------------------------------------------------------------------ */
/*  Self-test                                                          */
/* ------------------------------------------------------------------ */

/* Defined with the resident matmul entry points below. */
extern "C" int mynah_cuda_matmul_q8_d2d(void *, const float *, float *, size_t,
                                         size_t, size_t, const float *,
                                         const float *, char *, size_t);

static int cuda_q8_self_test(void *opaque, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr) return -1;
    /* The INT8 tensor-core GEMM requires aligned m/n/k dimensions on the
     * supported CUDA path.  Keep this model-free probe on a 4x4x4 shape;
     * production Pocket projections are all much larger and aligned. */
    const float input[16] = {1.0f, 2.0f, 3.0f, -1.0f,
                             -1.0f, 0.5f, 2.0f, 4.0f,
                             3.0f, -2.0f, 1.0f, 0.25f,
                             0.5f, 1.5f, -3.0f, 2.0f};
    const float weight[16] = {1.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 1.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 1.0f};
    const float bias[4] = {0.5f, -0.5f, 1.0f, 2.0f};
    std::vector<int8_t> qweight;
    std::vector<float> wscale;
    if (cuda_pack_weight_q8(weight, 4u, 4u, &qweight, &wscale, e, ec) != 0)
        return -1;
    int8_t qinput[16] = {0};
    float xscale[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (size_t row = 0; row < 4u; ++row) {
        float amax = 0.0f;
        for (size_t col = 0; col < 4u; ++col)
            amax = std::fmax(amax, std::fabs(input[row * 4u + col]));
        xscale[row] = amax == 0.0f ? 0.0f : amax / 127.0f;
        const float inv = xscale[row] == 0.0f ? 0.0f : 127.0f / xscale[row];
        for (size_t col = 0; col < 4u; ++col)
            qinput[row * 4u + col] = (int8_t)cuda_q8_round(
                input[row * 4u + col] * inv);
    }
    float *d_input = nullptr;
    float *d_output = nullptr;
    if (ce(cudaMalloc((void **)&d_input, sizeof(input)), e, ec) != 0 ||
        ce(cudaMalloc((void **)&d_output, 16u * sizeof(float)), e, ec) != 0) {
        if (d_input != nullptr) cudaFree(d_input);
        if (d_output != nullptr) cudaFree(d_output);
        return -1;
    }
    int result = 0;
    do {
        if (ce(cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice),
               e, ec) != 0 ||
            mynah_cuda_matmul_q8_d2d(st, d_input, d_output, 4u, 4u, 4u,
                                     weight, bias, e, ec) != 0 ||
            ce(cudaStreamSynchronize(st->stream), e, ec) != 0) {
            result = -1;
            break;
        }
        float output[16] = {0.0f};
        if (ce(cudaMemcpy(output, d_output, sizeof(output),
                          cudaMemcpyDeviceToHost), e, ec) != 0) {
            result = -1;
            break;
        }
        for (size_t row = 0; row < 4u && result == 0; ++row) {
            for (size_t col = 0; col < 4u; ++col) {
                int32_t dot = 0;
                for (size_t k = 0; k < 4u; ++k)
                    dot += (int32_t)qinput[row * 4u + k] *
                           (int32_t)qweight[col * 4u + k];
                const float row_scale = xscale[row] * wscale[col];
                const float expected = std::fmaf((float)dot, row_scale,
                                                 bias[col]);
                if (std::fabs(output[row * 4u + col] - expected) > 2.0e-5f) {
                    std::snprintf(e, ec, "Q8 matmul mismatch %zu: %f != %f",
                                  row * 4u + col, output[row * 4u + col],
                                  expected);
                    result = -1;
                    break;
                }
            }
        }
    } while (false);
    cudaFree(d_input);
    cudaFree(d_output);
    return result;
}

static int cuda_codec_gang_self_test(cuda_backend_state *st, char *e,
                                     size_t ec);
static int cuda_bf16_fuse_self_test(cuda_backend_state *st, char *e,
                                    size_t ec);
static int cuda_attn_split_self_test(cuda_backend_state *st, char *e,
                                     size_t ec);
static int cuda_kv_vmm_self_test(cuda_backend_state *st, char *e, size_t ec);

static int cuda_self_test(void *opaque, char *e, size_t ec) {
    const float in[6]={1,2,3,-1,0.5f,2};
    const float w[12]={1,0,0,0,1,0,0,0,1,1,1,1};
    const float bi[4]={0.5f,-0.5f,1,2};
    const float exp[8]={1.5f,1.5f,4,8,-0.5f,0,3,3.5f};
    float out[8]={0};
    if (cuda_matmul(opaque,in,out,2,3,4,w,bi,e,ec)) return -1;
    for (int i=0;i<8;i++) if (std::fabs(out[i]-exp[i])>1e-4f) {
        std::snprintf(e,ec,"matmul mismatch %d: %f!=%f",i,out[i],exp[i]); return -1; }
    float so[8]={0};
    if (cuda_sgemm(opaque,0,1,2,4,3,1.0f,in,3,w,3,0.0f,so,4,e,ec)) return -1;
    const float en[8]={1,2,3,6,-1,0.5f,2,1.5f};
    for (int i=0;i<8;i++) if (std::fabs(so[i]-en[i])>1e-4f) {
        std::snprintf(e,ec,"sgemm mismatch %d: %f!=%f",i,so[i],en[i]); return -1; }
    if (cuda_resident_kernel_self_test(opaque, e, ec) != 0) return -1;
    if (cuda_codec_gang_self_test(static_cast<cuda_backend_state *>(opaque),
                                  e, ec) != 0)
        return -1;
    if (cuda_bf16_fuse_self_test(static_cast<cuda_backend_state *>(opaque),
                                 e, ec) != 0)
        return -1;
    if (cuda_attn_split_self_test(static_cast<cuda_backend_state *>(opaque),
                                  e, ec) != 0)
        return -1;
    if (cuda_kv_vmm_self_test(static_cast<cuda_backend_state *>(opaque), e,
                              ec) != 0)
        return -1;
    return cuda_q8_self_test(opaque, e, ec);
}

/* ------------------------------------------------ MYNAH_CUDA_KV_VMM
 *
 * Driver entry points through the runtime (no -lcuda): the runtime already
 * loaded the driver, and a binary built here must still start on a host
 * without one (the compile-only CI runs --gpu-self-test with no driver). */
static void *cuda_driver_symbol(const char *name) {
    void *fn = nullptr;
    cudaError_t rc = cudaSuccess;
#if CUDART_VERSION >= 12050
    cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
    rc = cudaGetDriverEntryPointByVersion(name, &fn, 12000u, cudaEnableDefault,
                                          &query);
    if (query != cudaDriverEntryPointSuccess) fn = nullptr;
#elif CUDART_VERSION >= 12000
    cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
    rc = cudaGetDriverEntryPoint(name, &fn, cudaEnableDefault, &query);
    if (query != cudaDriverEntryPointSuccess) fn = nullptr;
#elif CUDART_VERSION >= 11030
    rc = cudaGetDriverEntryPoint(name, &fn, cudaEnableDefault);
#else
    (void)name;
#endif
    if (rc != cudaSuccess) {
        fn = nullptr;
        /* A failed lookup must not surface later as a kernel launch error. */
        (void)cudaGetLastError();
    }
    return fn;
}

static int cuda_vmm_check(const cuda_kv_vmm &v, CUresult r, const char *what,
                          char *e, size_t ec) {
    if (r == CUDA_SUCCESS) return 0;
    const char *text = nullptr;
    if (v.GetErrorString == nullptr || v.GetErrorString(r, &text) != CUDA_SUCCESS ||
        text == nullptr)
        text = "unknown driver error";
    if (e != nullptr && ec > 0u)
        std::snprintf(e, ec, "CUDA VMM %s: %s (%d)", what, text, (int)r);
    return -1;
}

/* Resolve the entry points and the device's support once. Caller holds
 * v.mutex. */
static int cuda_kv_vmm_probe_locked(cuda_kv_vmm &v) {
    if (v.status != 0) return v.status > 0 ? 0 : -1;
    v.status = -1;
#define CUDA_VMM_SYMBOL(field, name)                                       \
    v.field = reinterpret_cast<cuda_pfn_##name>(cuda_driver_symbol(#name)); \
    if (v.field == nullptr) {                                              \
        std::snprintf(v.reason, sizeof(v.reason),                          \
                      "the CUDA driver does not expose %s", #name);        \
        return -1;                                                         \
    }
    CUDA_VMM_SYMBOL(GetErrorString, cuGetErrorString)
    CUDA_VMM_SYMBOL(DeviceGet, cuDeviceGet)
    CUDA_VMM_SYMBOL(DeviceGetAttribute, cuDeviceGetAttribute)
    CUDA_VMM_SYMBOL(MemGetAllocationGranularity, cuMemGetAllocationGranularity)
    CUDA_VMM_SYMBOL(MemAddressReserve, cuMemAddressReserve)
    CUDA_VMM_SYMBOL(MemAddressFree, cuMemAddressFree)
    CUDA_VMM_SYMBOL(MemCreate, cuMemCreate)
    CUDA_VMM_SYMBOL(MemRelease, cuMemRelease)
    CUDA_VMM_SYMBOL(MemMap, cuMemMap)
    CUDA_VMM_SYMBOL(MemUnmap, cuMemUnmap)
    CUDA_VMM_SYMBOL(MemSetAccess, cuMemSetAccess)
#undef CUDA_VMM_SYMBOL
    int ordinal = 0;
    if (cudaGetDevice(&ordinal) != cudaSuccess) {
        (void)cudaGetLastError();
        std::snprintf(v.reason, sizeof(v.reason), "no current CUDA device");
        return -1;
    }
    CUdevice device = 0;
    int supported = 0;
    char detail[160];
    detail[0] = '\0';
    if (cuda_vmm_check(v, v.DeviceGet(&device, ordinal), "cuDeviceGet", detail,
                       sizeof(detail)) ||
        cuda_vmm_check(v,
                       v.DeviceGetAttribute(
                           &supported,
                           CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                           device),
                       "cuDeviceGetAttribute", detail, sizeof(detail))) {
        std::snprintf(v.reason, sizeof(v.reason), "%s", detail);
        return -1;
    }
    if (!supported) {
        std::snprintf(v.reason, sizeof(v.reason),
                      "the device does not support virtual memory management");
        return -1;
    }
    std::memset(&v.prop, 0, sizeof(v.prop));
    v.prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    v.prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; /* VRAM */
    v.prop.location.id = device;
    size_t granularity = 0u;
    if (cuda_vmm_check(v,
                       v.MemGetAllocationGranularity(
                           &granularity, &v.prop,
                           CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                       "cuMemGetAllocationGranularity", detail,
                       sizeof(detail)) ||
        granularity == 0u) {
        std::snprintf(v.reason, sizeof(v.reason), "%s",
                      detail[0] != '\0' ? detail : "zero allocation granularity");
        return -1;
    }
    std::memset(&v.access, 0, sizeof(v.access));
    v.access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    v.access.location.id = device;
    v.access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    v.device = ordinal;
    v.granularity = granularity;
    v.status = 1;
    return 0;
}

static size_t cuda_round_up(size_t value, size_t unit, bool *ok) {
    if (unit == 0u || value > SIZE_MAX - (unit - 1u)) {
        *ok = false;
        return 0u;
    }
    return (value + unit - 1u) / unit * unit;
}

/* Back [range.mapped, range.mapped + bytes) with one new device chunk.
 * Caller holds v.mutex; on failure nothing changed. */
static int cuda_kv_vmm_map_locked(cuda_kv_vmm &v, CUdeviceptr base,
                                  cuda_kv_vmm_range &range, size_t bytes,
                                  char *e, size_t ec) {
    CUmemGenericAllocationHandle handle = 0;
    const CUdeviceptr at = base + (CUdeviceptr)range.mapped;
    if (cuda_vmm_check(v, v.MemCreate(&handle, bytes, &v.prop, 0ull),
                       "cuMemCreate", e, ec))
        return -1;
    if (cuda_vmm_check(v, v.MemMap(at, bytes, 0u, handle, 0ull), "cuMemMap", e,
                       ec)) {
        (void)v.MemRelease(handle);
        return -1;
    }
    if (cuda_vmm_check(v, v.MemSetAccess(at, bytes, &v.access, 1u),
                       "cuMemSetAccess", e, ec)) {
        (void)v.MemUnmap(at, bytes);
        (void)v.MemRelease(handle);
        return -1;
    }
    try {
        range.chunks.push_back(cuda_kv_vmm_chunk{handle, range.mapped, bytes});
    } catch (...) {
        (void)v.MemUnmap(at, bytes);
        (void)v.MemRelease(handle);
        set_error(e, ec, "out of host memory for a CUDA VMM chunk");
        return -1;
    }
    range.mapped += bytes;
    v.mapped_bytes.fetch_add((unsigned long long)bytes, std::memory_order_relaxed);
    v.maps.fetch_add(1ull, std::memory_order_relaxed);
    return 0;
}

/* Unmap and release the last chunk. Caller holds v.mutex. cuMemUnmap may
 * synchronize the device; callers only shrink idle ranges. */
static void cuda_kv_vmm_unmap_last_locked(cuda_kv_vmm &v, CUdeviceptr base,
                                          cuda_kv_vmm_range &range) {
    const cuda_kv_vmm_chunk chunk = range.chunks.back();
    range.chunks.pop_back();
    (void)v.MemUnmap(base + (CUdeviceptr)chunk.offset, chunk.bytes);
    (void)v.MemRelease(chunk.handle);
    range.mapped = chunk.offset;
    v.mapped_bytes.fetch_sub((unsigned long long)chunk.bytes,
                             std::memory_order_relaxed);
    v.unmaps.fetch_add(1ull, std::memory_order_relaxed);
}

static void cuda_kv_vmm_destroy_locked(cuda_kv_vmm &v, CUdeviceptr base,
                                       cuda_kv_vmm_range &range) {
    while (!range.chunks.empty()) cuda_kv_vmm_unmap_last_locked(v, base, range);
    (void)v.MemAddressFree(base, range.reserved);
    v.reserved_bytes.fetch_sub((unsigned long long)range.reserved,
                               std::memory_order_relaxed);
}

/* 0 and the granularity when the VMM path is usable; -1 and the reason
 * otherwise. Probes once per backend. */
extern "C" int mynah_cuda_kv_vmm_probe(void *opaque, size_t *granularity,
                                        char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr) {
        set_error(e, ec, "no CUDA backend");
        return -1;
    }
    cuda_kv_vmm &v = st->kv_vmm;
    std::lock_guard<std::mutex> lock(v.mutex);
    if (cuda_kv_vmm_probe_locked(v) != 0) {
        set_error(e, ec, v.reason[0] != '\0' ? v.reason : "CUDA VMM unavailable");
        return -1;
    }
    if (granularity != nullptr) *granularity = v.granularity;
    return 0;
}

/* Reserve `reserve` virtual bytes and map the first `map` of them (both
 * rounded up to the granularity; `map` may be 0). */
extern "C" int mynah_cuda_kv_vmm_alloc(void *opaque, size_t reserve,
                                        size_t map, void **dev_ptr,
                                        size_t *mapped, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || dev_ptr == nullptr || reserve == 0u || map > reserve) {
        set_error(e, ec, "invalid CUDA VMM allocation");
        return -1;
    }
    *dev_ptr = nullptr;
    if (mapped != nullptr) *mapped = 0u;
    cuda_kv_vmm &v = st->kv_vmm;
    std::lock_guard<std::mutex> lock(v.mutex);
    if (cuda_kv_vmm_probe_locked(v) != 0) {
        set_error(e, ec, v.reason);
        return -1;
    }
    bool ok = true;
    const size_t reserve_bytes = cuda_round_up(reserve, v.granularity, &ok);
    const size_t map_bytes = map == 0u ? 0u : cuda_round_up(map, v.granularity, &ok);
    if (!ok || map_bytes > reserve_bytes) {
        set_error(e, ec, "CUDA VMM allocation size overflow");
        return -1;
    }
    /* Driver calls need the runtime's primary context current here. */
    if (ce(cudaSetDevice(v.device), e, ec)) return -1;
    CUdeviceptr base = 0;
    if (cuda_vmm_check(v, v.MemAddressReserve(&base, reserve_bytes,
                                              v.granularity, 0, 0ull),
                       "cuMemAddressReserve", e, ec))
        return -1;
    v.reserved_bytes.fetch_add((unsigned long long)reserve_bytes,
                               std::memory_order_relaxed);
    cuda_kv_vmm_range *range = nullptr;
    try {
        range = &v.ranges[(uintptr_t)base];
        range->chunks.reserve(8u);
    } catch (...) {
        if (range != nullptr) v.ranges.erase((uintptr_t)base);
        (void)v.MemAddressFree(base, reserve_bytes);
        v.reserved_bytes.fetch_sub((unsigned long long)reserve_bytes,
                                   std::memory_order_relaxed);
        set_error(e, ec, "out of host memory for a CUDA VMM range");
        return -1;
    }
    range->reserved = reserve_bytes;
    if (map_bytes != 0u &&
        cuda_kv_vmm_map_locked(v, base, *range, map_bytes, e, ec) != 0) {
        cuda_kv_vmm_destroy_locked(v, base, *range);
        v.ranges.erase((uintptr_t)base);
        return -1;
    }
    v.live.fetch_add(1u, std::memory_order_relaxed);
    *dev_ptr = reinterpret_cast<void *>((uintptr_t)base);
    if (mapped != nullptr) *mapped = map_bytes;
    return 0;
}

/* Make the mapped prefix of the range at `dev_ptr` at least `want` bytes
 * (rounded up to the granularity) by mapping one new chunk, or, when it is
 * larger than `want`, unmap whole trailing chunks while what stays mapped
 * still covers `want`. The base never moves and mapped bytes are never
 * copied. Growth is stream-independent: it only adds pages after the ones in
 * use, so work already queued on them is unaffected. */
extern "C" int mynah_cuda_kv_vmm_resize(void *opaque, void *dev_ptr,
                                         size_t want, size_t *mapped,
                                         char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || dev_ptr == nullptr) {
        set_error(e, ec, "invalid CUDA VMM resize");
        return -1;
    }
    cuda_kv_vmm &v = st->kv_vmm;
    std::lock_guard<std::mutex> lock(v.mutex);
    auto it = v.ranges.find((uintptr_t)dev_ptr);
    if (it == v.ranges.end()) {
        set_error(e, ec, "CUDA VMM resize of an unknown range");
        return -1;
    }
    cuda_kv_vmm_range &range = it->second;
    const CUdeviceptr base = (CUdeviceptr)(uintptr_t)dev_ptr;
    bool ok = true;
    const size_t want_bytes = cuda_round_up(want, v.granularity, &ok);
    if (!ok || want_bytes > range.reserved) {
        set_error(e, ec, "CUDA VMM resize past the reservation");
        return -1;
    }
    if (want_bytes > range.mapped) {
        if (ce(cudaSetDevice(v.device), e, ec) ||
            cuda_kv_vmm_map_locked(v, base, range, want_bytes - range.mapped, e,
                                   ec) != 0)
            return -1;
    } else {
        while (!range.chunks.empty() &&
               range.chunks.back().offset >= want_bytes)
            cuda_kv_vmm_unmap_last_locked(v, base, range);
    }
    if (mapped != nullptr) *mapped = range.mapped;
    return 0;
}

/* Reserved bytes of the range at `dev_ptr`, 0 when it is not one. */
extern "C" size_t mynah_cuda_kv_vmm_reserved(void *opaque, const void *dev_ptr) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || dev_ptr == nullptr ||
        st->kv_vmm.live.load(std::memory_order_relaxed) == 0u)
        return 0u;
    std::lock_guard<std::mutex> lock(st->kv_vmm.mutex);
    auto it = st->kv_vmm.ranges.find((uintptr_t)dev_ptr);
    return it == st->kv_vmm.ranges.end() ? 0u : it->second.reserved;
}

/* Unmap, release and unreserve. Nothing queued may still use the range
 * (callers drain first, as for cudaFree). Returns 1 when `dev_ptr` is not a
 * VMM range (nothing done). */
static int cuda_kv_vmm_free(cuda_backend_state *st, void *dev_ptr) {
    if (st == nullptr || dev_ptr == nullptr ||
        st->kv_vmm.live.load(std::memory_order_relaxed) == 0u)
        return 1;
    cuda_kv_vmm &v = st->kv_vmm;
    std::lock_guard<std::mutex> lock(v.mutex);
    auto it = v.ranges.find((uintptr_t)dev_ptr);
    if (it == v.ranges.end()) return 1;
    (void)cudaSetDevice(v.device);
    cuda_kv_vmm_destroy_locked(v, (CUdeviceptr)(uintptr_t)dev_ptr, it->second);
    v.ranges.erase(it);
    v.live.fetch_sub(1u, std::memory_order_relaxed);
    return 0;
}

extern "C" void mynah_cuda_kv_vmm_free(void *opaque, void *dev_ptr) {
    (void)cuda_kv_vmm_free(static_cast<cuda_backend_state *>(opaque), dev_ptr);
}

static void cuda_kv_vmm_release_all(cuda_backend_state *st) {
    cuda_kv_vmm &v = st->kv_vmm;
    std::lock_guard<std::mutex> lock(v.mutex);
    if (v.ranges.empty()) return;
    (void)cudaSetDevice(v.device);
    for (auto &entry : v.ranges)
        cuda_kv_vmm_destroy_locked(v, (CUdeviceptr)entry.first, entry.second);
    v.ranges.clear();
    v.live.store(0u, std::memory_order_relaxed);
}

static void cuda_close(void *opaque) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (!st) return;
    /* Decoder graph entries own device pointer tables used by the stream.  A
     * model close normally arrives after request teardown, but make the
     * backend lifetime ordering explicit for direct library users too. */
    (void)cudaStreamSynchronize(st->stream);
    destroy_decoder_batch_graphs(st);
    destroy_graphs(st);
    tile_release(st);
    cuda_kv_vmm_release_all(st);
    cudaFree(st->dec_cols);
    cudaFree(st->dec_out);
    cudaFree(st->dec_tr_x);
    cudaFree(st->dec_tr_y);
    cudaFree(st->solo_work_a);
    cudaFree(st->solo_work_b);
    cudaFree(st->solo_work_c);
    cudaFree(st->solo_columns);
    cudaFree(st->solo_window);
    cudaFree(st->solo_full);
    for (float *q : st->solo_retired) cudaFree(q);
    st->solo_retired.clear();
    codec_gang_release(st);
    for (auto &c : st->weights) cudaFree(c.device_pointer);
    for (auto &c : st->weights_fp16) cudaFree(c.device_ptr);
    for (auto &c : st->weights_int8) {
        cudaFree(c.device_q);
        cudaFree(c.device_scale);
    }
    for (auto &c : st->weights_bf16) cudaFree(c.device_ptr);
    if (st->dev_bf16_activation) cudaFree(st->dev_bf16_activation);
    if (st->dev_scratch) cudaFree(st->dev_scratch);
    if (st->dev_bf16_convert) cudaFree(st->dev_bf16_convert);
    if (st->dev_q8_activation) cudaFree(st->dev_q8_activation);
    if (st->dev_q8_activation_scale) cudaFree(st->dev_q8_activation_scale);
    if (st->dev_q8_accum) cudaFree(st->dev_q8_accum);
    if (st->host_buf) cudaFreeHost(st->host_buf);
    if (st->dev_argmax) cudaFree(st->dev_argmax);
    if (st->cublas_workspace) cudaFree(st->cublas_workspace);
    if (st->dev_batch_k_cache) cudaFree(st->dev_batch_k_cache);
    if (st->dev_batch_v_cache) cudaFree(st->dev_batch_v_cache);
    if (st->dev_batch_positions) cudaFree(st->dev_batch_positions);
    if (st->dev_batch_cache_strides) cudaFree(st->dev_batch_cache_strides);
    if (st->dev_batch_k_prefix) cudaFree(st->dev_batch_k_prefix);
    if (st->dev_batch_v_prefix) cudaFree(st->dev_batch_v_prefix);
    if (st->dev_batch_prefix_len) cudaFree(st->dev_batch_prefix_len);
    if (st->dev_decoder_ptr0) cudaFree(st->dev_decoder_ptr0);
    if (st->dev_decoder_ptr1) cudaFree(st->dev_decoder_ptr1);
    if (st->dev_decoder_ptr2) cudaFree(st->dev_decoder_ptr2);
    if (st->dev_decoder_ptr3) cudaFree(st->dev_decoder_ptr3);
    if (st->lt_ws) cudaFree(st->lt_ws);
    if (st->lt) cublasLtDestroy(st->lt);
    cublasDestroy(st->cublas);
    cudaStreamDestroy(st->stream);
    delete st;
}

static void cuda_nvtx_push(const char *name) { nvtxRangePushA(name); }
static void cuda_nvtx_pop(void) { nvtxRangePop(); }

extern "C" int mynah_backend_cuda_open(void **state_out, mynah_backend_matmul_fn *matmul,
                                       mynah_backend_sgemm_fn *sgemm,
                                       mynah_backend_close_fn *close,
                                       mynah_backend_self_test_fn *self_test,
                                       char *e, size_t ec) {
    /* Mapped host staging is part of the FP32 bring-up path.  Device flags
     * must be set before any runtime call that can initialise the primary
     * context; otherwise cudaHostGetDevicePointer may be unavailable. */
    unsigned int sched = cudaDeviceScheduleAuto;
    {
        /* MYNAH_CUDA_SYNC picks how the host waits on the device: the
         * default lets the driver spin, which burns a whole core per
         * waiting thread; "blocking" sleeps on an OS primitive instead. */
        const char *s = getenv("MYNAH_CUDA_SYNC");
        if (s != nullptr && strcmp(s, "blocking") == 0) sched = cudaDeviceScheduleBlockingSync;
        else if (s != nullptr && strcmp(s, "yield") == 0) sched = cudaDeviceScheduleYield;
        else if (s != nullptr && strcmp(s, "spin") == 0) sched = cudaDeviceScheduleSpin;
    }
    if (ce(cudaSetDeviceFlags(cudaDeviceMapHost | sched), e, ec)) return -1;
    int dc = 0;
    if (ce(cudaGetDeviceCount(&dc), e, ec) || dc == 0) {
        if (dc == 0) set_error(e, ec, "no CUDA device"); return -1; }
    if (ce(cudaSetDevice(0), e, ec)) return -1;
    auto *st = new (std::nothrow) cuda_backend_state();
    if (!st) { set_error(e,ec,"oom"); return -1; }
    {
        /* MYNAH_NVTX=1 with MYNAH_COST_MAP=2 puts every engine region on the
         * Nsight Systems timeline next to the kernels it launched. */
        const char *nvtx = getenv("MYNAH_NVTX");
        if (nvtx != nullptr && strcmp(nvtx, "0") != 0)
            mynah_costmap_set_range_hooks(cuda_nvtx_push, cuda_nvtx_pop);
    }
    st->dev_scratch = nullptr; st->dev_scratch_cap = 0;
    st->dev_bf16_convert = nullptr; st->dev_bf16_convert_cap = 0u;
    st->dev_q8_activation = nullptr;
    st->dev_q8_activation_scale = nullptr;
    st->dev_q8_accum = nullptr;
    st->dev_q8_activation_cap = 0u;
    st->dev_q8_rows_cap = 0u;
    st->dev_q8_accum_cap = 0u;
    st->dev_bf16_activation = nullptr;
    st->dev_bf16_activation_cap = 0u;
    st->host_buf = nullptr; st->dev_buf = nullptr; st->host_buf_cap = 0;
    st->dev_argmax = nullptr;
    st->cublas_workspace = nullptr; st->cublas_workspace_cap = 0;
    st->dev_batch_k_cache = nullptr;
    st->dev_batch_v_cache = nullptr;
    st->dev_batch_positions = nullptr;
    st->dev_batch_cache_strides = nullptr;
    st->dev_batch_k_prefix = nullptr;
    st->dev_batch_v_prefix = nullptr;
    st->dev_batch_prefix_len = nullptr;
    st->dev_decoder_ptr0 = nullptr;
    st->dev_decoder_ptr1 = nullptr;
    st->dev_decoder_ptr2 = nullptr;
    st->dev_decoder_ptr3 = nullptr;
    st->active_decoder_batch_graph = nullptr;
    st->active_decoder_upload_slot = 0u;
    for (size_t i = 0; i < 4u; ++i) st->active_decoder_tables[i] = nullptr;
    st->h2d_bytes.store(0ull, std::memory_order_relaxed);
    st->d2h_bytes.store(0ull, std::memory_order_relaxed);
    st->h2d_calls.store(0ull, std::memory_order_relaxed);
    st->d2h_calls.store(0ull, std::memory_order_relaxed);
    st->sync_calls.store(0ull, std::memory_order_relaxed);
    st->graph_captures.store(0ull, std::memory_order_relaxed);
    st->graph_replays.store(0ull, std::memory_order_relaxed);
    st->graph_fallbacks.store(0ull, std::memory_order_relaxed);
    st->backbone_batch_calls.store(0ull, std::memory_order_relaxed);
    st->backbone_batch_items.store(0ull, std::memory_order_relaxed);
    st->backbone_batch_max_width.store(0ull, std::memory_order_relaxed);
    st->codec_transformer_batch_calls.store(0ull, std::memory_order_relaxed);
    st->codec_transformer_batch_items.store(0ull, std::memory_order_relaxed);
    st->codec_transformer_batch_max_width.store(0ull, std::memory_order_relaxed);
    st->codec_upsample_steps.store(0ull, std::memory_order_relaxed);
    st->codec_upsample_fallbacks.store(0ull, std::memory_order_relaxed);
    st->decoder_steps.store(0ull, std::memory_order_relaxed);
    st->decoder_batch_calls.store(0ull, std::memory_order_relaxed);
    st->decoder_batch_items.store(0ull, std::memory_order_relaxed);
    st->decoder_batch_max_width.store(0ull, std::memory_order_relaxed);
    st->decoder_batch_frames.store(0ull, std::memory_order_relaxed);
    st->decoder_graph_captures.store(0ull, std::memory_order_relaxed);
    st->decoder_graph_replays.store(0ull, std::memory_order_relaxed);
    st->decoder_graph_fallbacks.store(0ull, std::memory_order_relaxed);
    st->decoder_graph_rerecords.store(0ull, std::memory_order_relaxed);
    st->decoder_graph_table_patches.store(0ull, std::memory_order_relaxed);
    st->decoder_compat_next.store(0ull, std::memory_order_relaxed);
    st->decoder_failures.store(0ull, std::memory_order_relaxed);
    st->resident_fallbacks.store(0ull, std::memory_order_relaxed);
    st->matmul_calls.store(0ull, std::memory_order_relaxed);
    st->matvec_calls.store(0ull, std::memory_order_relaxed);
    st->q8_matmul_calls.store(0ull, std::memory_order_relaxed);
    st->tile_calls.store(0ull, std::memory_order_relaxed);
    for (auto &stage : st->width_hist)
        for (auto &bucket : stage) bucket.store(0ull, std::memory_order_relaxed);
    st->tile_rows.store(0ull, std::memory_order_relaxed);
    for (int stage = 0; stage < 2; ++stage) {
        st->codec_gang_calls[stage].store(0ull, std::memory_order_relaxed);
        st->codec_gang_rows[stage].store(0ull, std::memory_order_relaxed);
        for (auto &bucket : st->codec_gang_hist[stage])
            bucket.store(0ull, std::memory_order_relaxed);
    }
    st->q8_rows.store(0ull, std::memory_order_relaxed);
    st->q8_weight_uploads.store(0ull, std::memory_order_relaxed);
    st->q8_weight_bytes.store(0ull, std::memory_order_relaxed);
    st->q8_activation_bytes.store(0ull, std::memory_order_relaxed);
    st->bf16_matmul_calls.store(0ull, std::memory_order_relaxed);
    st->bf16_rows.store(0ull, std::memory_order_relaxed);
    st->bf16_weight_bytes.store(0ull, std::memory_order_relaxed);
    st->batch_meta_cap = CUDA_BATCH_META_CAP;
    st->fast_math = cuda_fast_math_enabled();
    st->graphs_enabled = cuda_graphs_enabled();
    st->q8_enabled = true;
    st->decoder_batch_enabled = cuda_decoder_batch_enabled();
    if (ce(cudaStreamCreate(&st->stream), e, ec)) { delete st; return -1; }
    if (cublasCreate(&st->cublas) != CUBLAS_STATUS_SUCCESS) {
        set_error(e,ec,"cuBLAS init"); cudaStreamDestroy(st->stream); delete st; return -1; }
    if (ce(cudaMalloc(&st->dev_batch_k_cache,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_batch_k_cache)), e, ec) ||
        ce(cudaMalloc(&st->dev_batch_v_cache,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_batch_v_cache)), e, ec) ||
        ce(cudaMalloc(&st->dev_batch_positions,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_batch_positions)), e, ec) ||
        ce(cudaMalloc(&st->dev_batch_cache_strides,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_batch_cache_strides)), e, ec) ||
        ce(cudaMalloc(&st->dev_batch_k_prefix,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_batch_k_prefix)), e, ec) ||
        ce(cudaMalloc(&st->dev_batch_v_prefix,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_batch_v_prefix)), e, ec) ||
        ce(cudaMalloc(&st->dev_batch_prefix_len,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_batch_prefix_len)), e, ec) ||
        ce(cudaMalloc(&st->dev_decoder_ptr0,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_decoder_ptr0)), e, ec) ||
        ce(cudaMalloc(&st->dev_decoder_ptr1,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_decoder_ptr1)), e, ec) ||
        ce(cudaMalloc(&st->dev_decoder_ptr2,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_decoder_ptr2)), e, ec) ||
        ce(cudaMalloc(&st->dev_decoder_ptr3,
                      CUDA_BATCH_META_CAP * sizeof(*st->dev_decoder_ptr3)), e, ec)) {
        cuda_close(st);
        return -1;
    }
    /* The parity path must not silently use TF32 on Ampere/Ada.  Fast math is
     * an explicit opt-in experiment; its Tensor-Core error budget is a later
     * stage gate, not the default CUDA result. */
    st->tf32 = !st->fast_math && cuda_env_enabled("MYNAH_CUDA_TF32", true);
    st->tile_cublas = cuda_env_enabled("MYNAH_CUDA_TILE_CUBLAS", true);
    st->quant_weights = cuda_quant_weights_requested();
    const cublasMath_t math_mode = st->fast_math
        ? CUBLAS_DEFAULT_MATH
        : (st->tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_PEDANTIC_MATH);
    if (cublasSetMathMode(st->cublas, math_mode) != CUBLAS_STATUS_SUCCESS) {
        set_error(e, ec, "cuBLAS math mode setup failed");
        cuda_close(st);
        return -1;
    }
    *state_out = st;
    *matmul = cuda_matmul;
    *sgemm = cuda_sgemm;
    *close = cuda_close;
    *self_test = cuda_self_test;
    if (e && ec > 0) e[0] = '\0';
    return 0;
}

extern "C" int mynah_cuda_metrics_get(void *opaque,
                                       mynah_tts_backend_metrics *metrics) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || metrics == nullptr) return -1;
    metrics->h2d_bytes = st->h2d_bytes.load(std::memory_order_relaxed);
    metrics->d2h_bytes = st->d2h_bytes.load(std::memory_order_relaxed);
    metrics->h2d_calls = st->h2d_calls.load(std::memory_order_relaxed);
    metrics->d2h_calls = st->d2h_calls.load(std::memory_order_relaxed);
    metrics->sync_calls = st->sync_calls.load(std::memory_order_relaxed);
    metrics->graph_captures = st->graph_captures.load(std::memory_order_relaxed);
    metrics->graph_replays = st->graph_replays.load(std::memory_order_relaxed);
    metrics->graph_fallbacks = st->graph_fallbacks.load(std::memory_order_relaxed);
    metrics->backbone_batch_calls =
        st->backbone_batch_calls.load(std::memory_order_relaxed);
    metrics->backbone_batch_items =
        st->backbone_batch_items.load(std::memory_order_relaxed);
    metrics->backbone_batch_max_width =
        st->backbone_batch_max_width.load(std::memory_order_relaxed);
    metrics->codec_transformer_batch_calls =
        st->codec_transformer_batch_calls.load(std::memory_order_relaxed);
    for (int stage = 0; stage < 3; ++stage)
        for (int bucket = 0; bucket < 8; ++bucket)
            metrics->batch_width_hist[stage][bucket] =
                st->width_hist[stage][bucket].load(std::memory_order_relaxed);
    for (int stage = 0; stage < 2; ++stage) {
        metrics->codec_gang_calls[stage] =
            st->codec_gang_calls[stage].load(std::memory_order_relaxed);
        metrics->codec_gang_rows[stage] =
            st->codec_gang_rows[stage].load(std::memory_order_relaxed);
        for (int bucket = 0; bucket < 8; ++bucket)
            metrics->codec_gang_width_hist[stage][bucket] =
                st->codec_gang_hist[stage][bucket].load(std::memory_order_relaxed);
    }
    metrics->codec_transformer_batch_items =
        st->codec_transformer_batch_items.load(std::memory_order_relaxed);
    metrics->codec_transformer_batch_max_width =
        st->codec_transformer_batch_max_width.load(std::memory_order_relaxed);
    metrics->codec_upsample_steps =
        st->codec_upsample_steps.load(std::memory_order_relaxed);
    metrics->codec_upsample_fallbacks =
        st->codec_upsample_fallbacks.load(std::memory_order_relaxed);
    metrics->decoder_steps = st->decoder_steps.load(std::memory_order_relaxed);
    metrics->decoder_batch_calls =
        st->decoder_batch_calls.load(std::memory_order_relaxed);
    metrics->decoder_batch_items =
        st->decoder_batch_items.load(std::memory_order_relaxed);
    metrics->decoder_batch_max_width =
        st->decoder_batch_max_width.load(std::memory_order_relaxed);
    metrics->decoder_batch_frames =
        st->decoder_batch_frames.load(std::memory_order_relaxed);
    metrics->decoder_graph_captures =
        st->decoder_graph_captures.load(std::memory_order_relaxed);
    metrics->decoder_graph_replays =
        st->decoder_graph_replays.load(std::memory_order_relaxed);
    metrics->decoder_graph_fallbacks =
        st->decoder_graph_fallbacks.load(std::memory_order_relaxed);
    metrics->decoder_graph_rerecords =
        st->decoder_graph_rerecords.load(std::memory_order_relaxed);
    metrics->decoder_graph_table_patches =
        st->decoder_graph_table_patches.load(std::memory_order_relaxed);
    metrics->decoder_failures = st->decoder_failures.load(std::memory_order_relaxed);
    metrics->resident_fallbacks = st->resident_fallbacks.load(std::memory_order_relaxed);
    metrics->matmul_calls = st->matmul_calls.load(std::memory_order_relaxed);
    metrics->matvec_calls = st->matvec_calls.load(std::memory_order_relaxed);
    metrics->q8_matmul_calls =
        st->q8_matmul_calls.load(std::memory_order_relaxed);
    metrics->q8_rows = st->q8_rows.load(std::memory_order_relaxed);
    metrics->q8_weight_uploads =
        st->q8_weight_uploads.load(std::memory_order_relaxed);
    metrics->q8_weight_bytes =
        st->q8_weight_bytes.load(std::memory_order_relaxed);
    metrics->q8_activation_bytes =
        st->q8_activation_bytes.load(std::memory_order_relaxed);
    metrics->bf16_matmul_calls =
        st->bf16_matmul_calls.load(std::memory_order_relaxed);
    metrics->bf16_rows = st->bf16_rows.load(std::memory_order_relaxed);
    metrics->bf16_weight_bytes =
        st->bf16_weight_bytes.load(std::memory_order_relaxed);
    size_t free_bytes = 0u;
    size_t total_bytes = 0u;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        metrics->device_memory_bytes = (unsigned long long)total_bytes;
        metrics->device_memory_free_bytes = (unsigned long long)free_bytes;
    }
    metrics->kv_vmm_rows =
        (unsigned long long)st->kv_vmm.live.load(std::memory_order_relaxed);
    metrics->kv_vmm_mapped_bytes =
        st->kv_vmm.mapped_bytes.load(std::memory_order_relaxed);
    metrics->kv_vmm_reserved_bytes =
        st->kv_vmm.reserved_bytes.load(std::memory_order_relaxed);
    metrics->kv_vmm_maps = st->kv_vmm.maps.load(std::memory_order_relaxed);
    metrics->kv_vmm_unmaps = st->kv_vmm.unmaps.load(std::memory_order_relaxed);
    metrics->graphs_enabled = st->graphs_enabled ? 1u : 0u;
    metrics->fast_math_enabled = st->fast_math ? 1u : 0u;
    metrics->decoder_batch_enabled = st->decoder_batch_enabled ? 1u : 0u;
    metrics->q8_enabled = st->q8_enabled ? 1u : 0u;
    return 0;
}

extern "C" int mynah_cuda_dev_alloc(void *opaque, size_t n, float **dev_ptr,
                          char *e, size_t ec) {
    (void)opaque;
    if (dev_ptr == nullptr || n == 0u || n > SIZE_MAX / sizeof(float)) {
        set_error(e, ec, "invalid CUDA device allocation size");
        return -1;
    }
    return ce(cudaMalloc(dev_ptr, n * sizeof(float)), e, ec);
}

extern "C" int mynah_cuda_dev_alloc_bytes(void *opaque, size_t bytes,
                                            void **dev_ptr, char *e,
                                            size_t ec) {
    (void)opaque;
    if (dev_ptr == nullptr || bytes == 0u) {
        set_error(e, ec, "invalid CUDA byte device allocation size");
        return -1;
    }
    return ce(cudaMalloc(dev_ptr, bytes), e, ec);
}

extern "C" void mynah_cuda_dev_free(void *opaque, float *dev_ptr) {
    if (dev_ptr == nullptr) return;
    /* A MYNAH_CUDA_KV_VMM range is not a cudaMalloc pointer; route it to its
     * own release (no lookup at all while no range is live). */
    if (cuda_kv_vmm_free(static_cast<cuda_backend_state *>(opaque), dev_ptr) == 0)
        return;
    cudaFree(dev_ptr);
}

extern "C" int mynah_cuda_host_alloc(void *opaque, size_t n, float **host_ptr,
                                      char *e, size_t ec) {
    (void)opaque;
    if (host_ptr == nullptr || n == 0u || n > SIZE_MAX / sizeof(float)) {
        set_error(e, ec, "invalid CUDA host allocation size");
        return -1;
    }
    return ce(cudaHostAlloc((void **)host_ptr, n * sizeof(float),
                            cudaHostAllocPortable), e, ec);
}

extern "C" void mynah_cuda_host_free(void *opaque, float *host_ptr) {
    (void)opaque;
    if (host_ptr != nullptr) cudaFreeHost(host_ptr);
}

extern "C" int mynah_cuda_h2d(void *opaque, const float *host, float *dev_ptr,
                    size_t n, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || host == nullptr || dev_ptr == nullptr || n == 0u ||
        n > SIZE_MAX / sizeof(float)) {
        set_error(e, ec, "invalid CUDA host-to-device copy");
        return -1;
    }
    st->h2d_bytes.fetch_add((unsigned long long)(n * sizeof(float)),
                            std::memory_order_relaxed);
    st->h2d_calls.fetch_add(1ull, std::memory_order_relaxed);
    return ce(cudaMemcpyAsync(dev_ptr, host, n*sizeof(float),
                              cudaMemcpyHostToDevice, st->stream), e, ec);
}

extern "C" int mynah_cuda_d2h(void *opaque, const float *dev_ptr, float *host,
                    size_t n, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || dev_ptr == nullptr || host == nullptr || n == 0u ||
        n > SIZE_MAX / sizeof(float)) {
        set_error(e, ec, "invalid CUDA device-to-host copy");
        return -1;
    }
    st->d2h_bytes.fetch_add((unsigned long long)(n * sizeof(float)),
                            std::memory_order_relaxed);
    st->d2h_calls.fetch_add(1ull, std::memory_order_relaxed);
    return ce(cudaMemcpyAsync(host, dev_ptr, n*sizeof(float),
                              cudaMemcpyDeviceToHost, st->stream), e, ec);
}

extern "C" int mynah_cuda_h2d_bf16(void *opaque, const float *host,
                                     void *dev_ptr, size_t n, char *e,
                                     size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || host == nullptr || dev_ptr == nullptr || n == 0u ||
        n > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA BF16 host-to-device copy");
        return -1;
    }
    if (ensure_bf16_convert(st, n, e, ec)) return -1;
    const size_t bytes = n * sizeof(float);
    st->h2d_bytes.fetch_add((unsigned long long)bytes, std::memory_order_relaxed);
    st->h2d_calls.fetch_add(1ull, std::memory_order_relaxed);
    if (ce(cudaMemcpyAsync(st->dev_bf16_convert, host, bytes,
                           cudaMemcpyHostToDevice, st->stream), e, ec)) return -1;
    k_f32_to_bf16<<<((int)n + 255) / 256, 256, 0, st->stream>>>(
        st->dev_bf16_convert, static_cast<uint16_t *>(dev_ptr), (int)n);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_d2h_bf16(void *opaque, const void *dev_ptr,
                                     float *host, size_t n, char *e,
                                     size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || dev_ptr == nullptr || host == nullptr || n == 0u ||
        n > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA BF16 device-to-host copy");
        return -1;
    }
    if (ensure_bf16_convert(st, n, e, ec)) return -1;
    k_bf16_to_f32<<<((int)n + 255) / 256, 256, 0, st->stream>>>(
        static_cast<const uint16_t *>(dev_ptr), st->dev_bf16_convert, (int)n);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    st->d2h_bytes.fetch_add((unsigned long long)(n * sizeof(float)),
                            std::memory_order_relaxed);
    st->d2h_calls.fetch_add(1ull, std::memory_order_relaxed);
    return ce(cudaMemcpyAsync(host, st->dev_bf16_convert, n * sizeof(float),
                              cudaMemcpyDeviceToHost, st->stream), e, ec);
}

/* Custom matvec kernel: out[n] = sum_k(in[k] * W[n*K + k]) + bias[n].
 * One thread per output element.  For 1×768 matvecs this beats cuBLAS
 * by eliminating launch + workspace overhead (~3μs vs ~15μs). */
__device__ __forceinline__ static float matvec_column(
    const float *__restrict__ in, const float *__restrict__ weight,
    const float *__restrict__ bias, int K, int col) {
    const float *w = weight + (size_t)col * K;
    float sum = bias ? bias[col] : 0.0f;
    for (int k = 0; k < K; ++k) sum += in[k] * w[k];
    return sum;
}

__global__ static void k_matvec(const float *__restrict__ in,
                                const float *__restrict__ weight,
                                const float *__restrict__ bias,
                                float *__restrict__ out,
                                int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    out[col] = matvec_column(in, weight, bias, K, col);
}

/* M independent matvecs sharing one weight: [rows][K] -> [rows][N].  This is
 * deliberately not a cuBLAS GEMM: each output keeps k_matvec's serial
 * reduction, so a row is bit-identical to the single-request projection
 * whatever the gang width (the Mimi decoder amplifies ulp differences). */
__global__ static void k_matvec_rows(const float *__restrict__ in,
                                     const float *__restrict__ weight,
                                     const float *__restrict__ bias,
                                     float *__restrict__ out, int K, int N) {
    const int col = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int row = (int)blockIdx.y;
    if (col >= N) return;
    out[(size_t)row * (size_t)N + (size_t)col] =
        matvec_column(in + (size_t)row * (size_t)K, weight, bias, K, col);
}

extern "C" int mynah_cuda_matvec_dev(void *opaque, const float *d_in, float *d_out,
                           size_t K, size_t N,
                           const float *weight, const float *bias,
                           char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t weight_n = 0;
    if (st == nullptr || d_in == nullptr || d_out == nullptr || weight == nullptr ||
        K == 0 || N == 0 || K > (size_t)INT_MAX || N > (size_t)INT_MAX ||
        !cuda_size_mul(K, N, &weight_n) ||
        !cuda_size_mul(weight_n, sizeof(float), &weight_n)) {
        set_error(e, ec, "invalid CUDA matvec dimensions");
        return -1;
    }
    float *dw = nullptr;
    if (cached_weight(st, weight, weight_n, &dw, e, ec)) return -1;
    float *db = nullptr;
    size_t bias_bytes = 0;
    if (bias && (!cuda_size_mul(N, sizeof(float), &bias_bytes) ||
                 cached_weight(st, bias, bias_bytes, &db, e, ec))) return -1;
    st->matvec_calls.fetch_add(1ull, std::memory_order_relaxed);
    k_matvec<<<((int)N + 255) / 256, 256, 0, st->stream>>>(
        d_in, dw, db, d_out, (int)K, (int)N);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    return 0; /* no sync */
}

/* ---- Inplace device-side ops (no copy, no sync) ---- */

extern "C" int mynah_cuda_gelu_inplace(void *opaque, float *dev_data, size_t n,
                             char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    k_gelu<<<((int)n+255)/256, 256, 0, st->stream>>>(dev_data, (int)n);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_residual_inplace(void *opaque, float *dev_out,
                                 const float *dev_in, size_t n,
                                 char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    k_residual_add<<<((int)n+255)/256, 256, 0, st->stream>>>(dev_out, dev_in, (int)n);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_layer_norm_inplace(void *opaque, const float *dev_in,
                                   float *dev_out, const float *gain,
                                   size_t rows, size_t width,
                                   char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (dev_in == nullptr || dev_out == nullptr || gain == nullptr ||
        rows == 0 || width == 0 || rows > (size_t)INT_MAX ||
        width > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA inplace layer-norm dimensions");
        return -1;
    }
    /* Cache gain on device (same pointer = same layer, reused across steps). */
    float *d_gain = nullptr;
    if (cached_weight(st, gain, width * sizeof(float), &d_gain, e, ec)) return -1;
    k_layer_norm<<<(int)rows, 256, 0, st->stream>>>(
        dev_out, dev_in, d_gain, nullptr, (int)width, 1e-5f, (int)rows);
    return ce(cudaGetLastError(), e, ec);
}

/* Matmul device-to-device: input already on GPU, no host round-trip.  The
 * default is FP32 accumulation/input for parity; MYNAH_CUDA_FAST_MATH=1 opts
 * into the tensor-core compute mode after the parity gate has been measured. */
extern "C" int mynah_cuda_matmul_d2d(void *opaque, const float *d_in, float *d_out,
                           size_t rows, size_t iw, size_t ow,
                           const float *weight, const float *bias,
                           char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t in_n = 0, out_n = 0, w_n = 0, total_n = 0;
    if (st == nullptr || validate_cuda_matmul(d_in, d_out, weight, rows, iw, ow,
                                               &in_n, &out_n, &w_n, &total_n,
                                               e, ec) != 0) return -1;
    float *dw = nullptr;
    if (cached_weight(st, weight, w_n * sizeof(float), &dw, e, ec)) return -1;
    float *db = nullptr;
    if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
    st->matmul_calls.fetch_add(1ull, std::memory_order_relaxed);
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b0 = 0.0f;
    if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         (int)ow, (int)rows, (int)iw,
                         &a1, dw, CUDA_R_32F, (int)iw,
                         d_in, CUDA_R_32F, (int)iw,
                         &b0, d_out, CUDA_R_32F, (int)ow,
                         cuda_compute_type(st), cuda_gemm_algo(st)), e, ec)) return -1;
    if (db) {
        k_bias_add<<<((int)(rows * ow) + 255) / 256, 256, 0, st->stream>>>(
            d_out, db, (int)rows, (int)ow);
        if (ce(cudaGetLastError(), e, ec)) return -1;
    }
    return 0; /* no sync */
}

/* Resident Q8 matmul. The activation is quantized on device per row, then a
 * signed INT8 x INT8 -> INT32 cuBLAS GEMM reads each cached weight row once for
 * the whole request batch. The small f32 epilogue restores the CPU qmat
 * representation (activation scale x weight-row scale + bias) into the normal
 * resident f32 activation buffer. No host copy or stream sync is performed. */
extern "C" int mynah_cuda_matmul_q8_d2d(void *opaque, const float *d_in,
                                         float *d_out, size_t rows, size_t iw,
                                         size_t ow, const float *weight,
                                         const float *bias, char *e,
                                         size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t in_n = 0u;
    size_t out_n = 0u;
    size_t weight_n = 0u;
    size_t total_n = 0u;
    if (st == nullptr || validate_cuda_matmul(d_in, d_out, weight, rows, iw, ow,
                                               &in_n, &out_n, &weight_n,
                                               &total_n, e, ec) != 0 ||
        out_n > (size_t)INT_MAX) {
        if (st != nullptr && out_n > (size_t)INT_MAX)
            set_error(e, ec, "CUDA Q8 output is too large for one launch");
        return -1;
    }
    int8_t *device_weight = nullptr;
    float *device_weight_scale = nullptr;
    if (cached_weight_q8(st, weight, ow, iw, &device_weight,
                         &device_weight_scale, e, ec) != 0)
        return -1;
    float *device_bias = nullptr;
    if (bias != nullptr &&
        cached_weight(st, bias, ow * sizeof(float), &device_bias, e, ec) != 0)
        return -1;
    if (ensure_q8_workspace(st, in_n, rows, out_n, e, ec) != 0) return -1;
    if (rows > (size_t)INT_MAX || iw > (size_t)INT_MAX || ow > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA Q8 GEMM dimensions exceed cuBLAS limits");
        return -1;
    }
    size_t quantize_blocks = 0u;
    size_t epilogue_blocks = 0u;
    if (!cuda_size_add(out_n, 255u, &epilogue_blocks) ||
        !cuda_size_add(rows, 255u, &quantize_blocks) ||
        (epilogue_blocks /= 256u) > (size_t)INT_MAX ||
        (quantize_blocks /= 256u) > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA Q8 launch size overflow");
        return -1;
    }
    k_q8_quantize_rows<<<(int)rows, 256, 0, st->stream>>>(
        d_in, st->dev_q8_activation, st->dev_q8_activation_scale,
        (int)rows, (int)iw);
    if (ce(cudaGetLastError(), e, ec) != 0) return -1;
    cublasSetStream(st->cublas, st->stream);
    const int32_t alpha = 1;
    const int32_t beta = 0;
    if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         (int)ow, (int)rows, (int)iw,
                         &alpha, device_weight, CUDA_R_8I, (int)iw,
                         st->dev_q8_activation, CUDA_R_8I, (int)iw,
                         &beta, st->dev_q8_accum, CUDA_R_32I, (int)ow,
                         CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT), e, ec) != 0)
        return -1;
    k_q8_epilogue<<<(int)epilogue_blocks, 256, 0, st->stream>>>(
        st->dev_q8_accum, st->dev_q8_activation_scale,
        device_weight_scale, device_bias, d_out, (int)rows, (int)ow);
    if (ce(cudaGetLastError(), e, ec) != 0) return -1;
    st->matmul_calls.fetch_add(1ull, std::memory_order_relaxed);
    st->q8_matmul_calls.fetch_add(1ull, std::memory_order_relaxed);
    st->q8_rows.fetch_add((unsigned long long)rows, std::memory_order_relaxed);
    st->q8_activation_bytes.fetch_add((unsigned long long)in_n,
                                      std::memory_order_relaxed);
    return 0;
}

/* Resident BF16-weight matmul.  Activations are rounded to BF16 on device
 * (RNE, the same helper as the BF16 KV cache), then one cuBLAS BF16 x BF16
 * GEMM with FP32 accumulation and FP32 output runs on the tensor cores.  The
 * cached weight is half the bytes of the f32 view, which is the point: the
 * decode step is bound by weight traffic, not arithmetic. */
/* Y[rows][ow] (fp32) = X[rows][iw] (bf16) * W[ow][iw]^T (bf16), fp32
 * accumulate, through cuBLASLt (MYNAH_CUDA_BF16_LT). The algorithm comes from
 * cublasLtMatmulAlgoGetHeuristic once per (m, n, k) and is cached, so a graph
 * replay always runs the same kernel. Returns 1 when cuBLASLt is off or has
 * no algorithm for the shape (the caller then uses cublasGemmEx), 0 on
 * success, -1 on a launch error. */
static int cuda_lt_bf16_gemm(cuda_backend_state *st, const uint16_t *w,
                             const uint16_t *x, float *y, size_t rows,
                             size_t iw, size_t ow, char *e, size_t ec) {
    if (st->lt == nullptr) return 1;
    const int m = (int)ow, n = (int)rows, k = (int)iw;
    cuda_backend_state::lt_algo_entry *hit = nullptr;
    for (auto &a : st->lt_algos)
        if (a.m == m && a.n == n && a.k == k) { hit = &a; break; }
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    int rc = 1;
    if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta,
                                       sizeof(ta)) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb,
                                       sizeof(tb)) != CUBLAS_STATUS_SUCCESS ||
        /* Column-major view: A is W stored k x m (ld k), B is X k x n, C m x n. */
        cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, k, m, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, k, n, k) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, m, n, m) != CUBLAS_STATUS_SUCCESS)
        goto done;
    if (hit == nullptr) {
        cuda_backend_state::lt_algo_entry entry{m, n, k, false, {}};
        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulHeuristicResult_t result{};
        int found = 0;
        if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &st->lt_ws_cap,
                sizeof(st->lt_ws_cap)) == CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulAlgoGetHeuristic(st->lt, op, la, lb, lc, lc, pref, 1,
                                           &result, &found) == CUBLAS_STATUS_SUCCESS &&
            found > 0) {
            entry.ok = true;
            entry.algo = result.algo;
        }
        if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
        try { st->lt_algos.push_back(entry); } catch (...) { goto done; }
        hit = &st->lt_algos.back();
    }
    if (!hit->ok) goto done;
    {
        const float alpha = 1.0f, beta = 0.0f;
        rc = cbe(cublasLtMatmul(st->lt, op, &alpha, w, la, x, lb, &beta, y, lc,
                                y, lc, &hit->algo, st->lt_ws, st->lt_ws_cap,
                                st->stream), e, ec) != 0 ? -1 : 0;
    }
done:
    if (lc) cublasLtMatrixLayoutDestroy(lc);
    if (lb) cublasLtMatrixLayoutDestroy(lb);
    if (la) cublasLtMatrixLayoutDestroy(la);
    if (op) cublasLtMatmulDescDestroy(op);
    return rc;
}

extern "C" int mynah_cuda_matmul_bf16_d2d(void *opaque, const float *d_in,
                                           float *d_out, size_t rows,
                                           size_t iw, size_t ow,
                                           const float *weight,
                                           const float *bias, char *e,
                                           size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t in_n = 0u, out_n = 0u, w_n = 0u, total_n = 0u;
    if (st == nullptr || validate_cuda_matmul(d_in, d_out, weight, rows, iw, ow,
                                               &in_n, &out_n, &w_n, &total_n,
                                               e, ec) != 0) return -1;
    if (in_n > (size_t)INT_MAX || out_n > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA BF16 matmul is too large for one launch");
        return -1;
    }
    uint16_t *dw = nullptr;
    if (cached_weight_bf16(st, weight, w_n, &dw, e, ec)) return -1;
    float *db = nullptr;
    if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
    if (ensure_bf16_activation(st, in_n, e, ec) != 0) return -1;
    k_f32_to_bf16<<<((int)in_n + 255) / 256, 256, 0, st->stream>>>(
        d_in, st->dev_bf16_activation, (int)in_n);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    const int lt = cuda_lt_bf16_gemm(st, dw, st->dev_bf16_activation, d_out,
                                     rows, iw, ow, e, ec);
    if (lt < 0) return -1;
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b0 = 0.0f;
    if (lt > 0 &&
        cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         (int)ow, (int)rows, (int)iw,
                         &a1, dw, CUDA_R_16BF, (int)iw,
                         st->dev_bf16_activation, CUDA_R_16BF, (int)iw,
                         &b0, d_out, CUDA_R_32F, (int)ow,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), e, ec))
        return -1;
    if (db) {
        k_bias_add<<<((int)out_n + 255) / 256, 256, 0, st->stream>>>(
            d_out, db, (int)rows, (int)ow);
        if (ce(cudaGetLastError(), e, ec)) return -1;
    }
    st->matmul_calls.fetch_add(1ull, std::memory_order_relaxed);
    st->bf16_matmul_calls.fetch_add(1ull, std::memory_order_relaxed);
    st->bf16_rows.fetch_add((unsigned long long)rows, std::memory_order_relaxed);
    return 0;
}

/* ---- MYNAH_CUDA_BF16_FUSE producers/consumer of the staged activation ----
 * The staged activation is st->dev_bf16_activation, the buffer
 * mynah_cuda_matmul_bf16_d2d casts into, so the staged GEMM below is the
 * identical cuBLAS call (same pointers, shapes, types and algorithm). */

extern "C" int mynah_cuda_layer_norm_bf16_stage_dev(
    void *opaque, const float *in, const float *gain, const float *bias,
    size_t rows, size_t width, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t n = 0u;
    if (st == nullptr || in == nullptr || gain == nullptr || rows == 0 ||
        width == 0 || rows > (size_t)INT_MAX || width > (size_t)INT_MAX ||
        !cuda_size_mul(rows, width, &n)) {
        set_error(e, ec, "invalid CUDA BF16 layer-norm dimensions");
        return -1;
    }
    float *d_gain = nullptr;
    float *d_bias = nullptr;
    if (cached_weight(st, gain, width * sizeof(float), &d_gain, e, ec)) return -1;
    if (bias != nullptr &&
        cached_weight(st, bias, width * sizeof(float), &d_bias, e, ec)) return -1;
    if (ensure_bf16_activation(st, n, e, ec) != 0) return -1;
    k_layer_norm_bf16<<<(int)rows, 256, 0, st->stream>>>(
        st->dev_bf16_activation, in, d_gain, d_bias, (int)width, 1e-5f,
        (int)rows);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_bias_gelu_bf16_stage_dev(
    void *opaque, const float *in, const float *bias, size_t rows,
    size_t cols, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t n = 0u;
    if (st == nullptr || in == nullptr || rows == 0u || cols == 0u ||
        !cuda_size_mul(rows, cols, &n) || n > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA BF16 GELU dimensions");
        return -1;
    }
    float *d_bias = nullptr;
    if (bias != nullptr &&
        cached_weight(st, bias, cols * sizeof(float), &d_bias, e, ec)) return -1;
    if (ensure_bf16_activation(st, n, e, ec) != 0) return -1;
    k_bias_gelu_bf16<<<((int)n + 255) / 256, 256, 0, st->stream>>>(
        in, d_bias, st->dev_bf16_activation, (int)rows, (int)cols);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_matmul_bf16_staged_d2d(void *opaque, float *d_out,
                                                  size_t rows, size_t iw,
                                                  size_t ow,
                                                  const float *weight, char *e,
                                                  size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t in_n = 0u, out_n = 0u, w_n = 0u, total_n = 0u;
    if (st == nullptr ||
        validate_cuda_matmul(st->dev_bf16_activation, d_out, weight, rows, iw,
                             ow, &in_n, &out_n, &w_n, &total_n, e, ec) != 0)
        return -1;
    if (in_n > st->dev_bf16_activation_cap || out_n > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA BF16 staged matmul exceeds the staged activation");
        return -1;
    }
    uint16_t *dw = nullptr;
    if (cached_weight_bf16(st, weight, w_n, &dw, e, ec)) return -1;
    const int lt = cuda_lt_bf16_gemm(st, dw, st->dev_bf16_activation, d_out,
                                     rows, iw, ow, e, ec);
    if (lt < 0) return -1;
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b0 = 0.0f;
    if (lt > 0 &&
        cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         (int)ow, (int)rows, (int)iw,
                         &a1, dw, CUDA_R_16BF, (int)iw,
                         st->dev_bf16_activation, CUDA_R_16BF, (int)iw,
                         &b0, d_out, CUDA_R_32F, (int)ow,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), e, ec))
        return -1;
    st->matmul_calls.fetch_add(1ull, std::memory_order_relaxed);
    st->bf16_matmul_calls.fetch_add(1ull, std::memory_order_relaxed);
    st->bf16_rows.fetch_add((unsigned long long)rows, std::memory_order_relaxed);
    return 0;
}

extern "C" int mynah_cuda_residual_bias_add_dev(void *opaque, float *out,
                                                 const float *in,
                                                 const float *bias,
                                                 size_t rows, size_t cols,
                                                 char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t n = 0u;
    if (st == nullptr || out == nullptr || in == nullptr || rows == 0u ||
        cols == 0u || !cuda_size_mul(rows, cols, &n) || n > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA residual dimensions");
        return -1;
    }
    float *d_bias = nullptr;
    if (bias != nullptr &&
        cached_weight(st, bias, cols * sizeof(float), &d_bias, e, ec)) return -1;
    k_residual_bias_add<<<((int)n + 255) / 256, 256, 0, st->stream>>>(
        out, in, d_bias, (int)rows, (int)cols);
    return ce(cudaGetLastError(), e, ec);
}

static int cuda_flow_layer_norm(cuda_backend_state *st, const float *in,
                                float *out, const float *gain,
                                const float *bias, size_t rows, size_t width,
                                float epsilon, char *e, size_t ec) {
    float *d_gain = nullptr;
    float *d_bias = nullptr;
    if (gain != nullptr && cached_weight(st, gain, width * sizeof(float),
                                          &d_gain, e, ec)) return -1;
    if (bias != nullptr && cached_weight(st, bias, width * sizeof(float),
                                          &d_bias, e, ec)) return -1;
    k_layer_norm<<<(int)rows, 256, 0, st->stream>>>(
        out, in, d_gain, d_bias, (int)width, epsilon, (int)rows);
    return ce(cudaGetLastError(), e, ec);
}

static int cuda_flow_linear(cuda_backend_state *st, const float *in, float *out,
                            size_t rows, size_t input_width,
                            size_t output_width,
                            const mynah_backend_flow_linear *linear,
                            char *e, size_t ec) {
    if (linear == nullptr || linear->weight == nullptr) {
        set_error(e, ec, "missing CUDA flow projection");
        return -1;
    }
    if (linear->qtype == 1)
        return mynah_cuda_matmul_q8_d2d(st, in, out, rows, input_width,
                                        output_width, linear->weight,
                                        linear->bias, e, ec);
    if (linear->qtype == MYNAH_BACKEND_QTYPE_BF16)
        return mynah_cuda_matmul_bf16_d2d(st, in, out, rows, input_width,
                                          output_width, linear->weight,
                                          linear->bias, e, ec);
    if (linear->qtype != 0) {
        set_error(e, ec, "unsupported CUDA flow projection precision");
        return -1;
    }
    return mynah_cuda_matmul_d2d(st, in, out, rows, input_width, output_width,
                                 linear->weight, linear->bias, e, ec);
}

extern "C" int mynah_cuda_flow_batch_dev(
    void *opaque, const mynah_backend_flow_batch *flow, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || flow == nullptr || flow->batch == 0u ||
        flow->latent_dim == 0u || flow->cond_dim == 0u ||
        flow->hidden_dim < 2u || flow->depth == 0u ||
        flow->num_time_conds == 0u || flow->freq_embed_dim < 2u ||
        flow->dev_cond == nullptr || flow->dev_noise == nullptr ||
        flow->dev_time_embed == nullptr || flow->dev_y == nullptr ||
        flow->dev_silu == nullptr || flow->dev_x == nullptr ||
        flow->dev_norm == nullptr || flow->dev_hidden == nullptr ||
        flow->dev_scratch == nullptr || flow->dev_mod == nullptr ||
        flow->dev_final_mod == nullptr || flow->dev_time_hidden == nullptr ||
        flow->dev_time_output == nullptr || flow->dev_time_sum == nullptr ||
        flow->dev_out == nullptr || flow->time_mlp_in == nullptr ||
        flow->time_mlp_out == nullptr || flow->time_alpha == nullptr ||
        flow->cond_embed == nullptr || flow->input_proj == nullptr ||
        flow->blocks == nullptr || flow->final_adaln == nullptr ||
        flow->final_linear == nullptr) {
        set_error(e, ec, "invalid CUDA flow batch descriptor");
        return -1;
    }
    const size_t h = flow->hidden_dim;
    size_t three_h = 0u;
    size_t two_h = 0u;
    size_t batch_hidden = 0u;
    size_t time_hidden = 0u;
    if (!cuda_size_mul(h, 3u, &three_h) || !cuda_size_mul(h, 2u, &two_h) ||
        !cuda_size_mul(flow->batch, h, &batch_hidden) ||
        !cuda_size_mul(flow->num_time_conds, h, &time_hidden) ||
        flow->batch > (size_t)INT_MAX || h > (size_t)INT_MAX ||
        three_h > (size_t)INT_MAX || two_h > (size_t)INT_MAX ||
        flow->latent_dim > (size_t)INT_MAX || flow->cond_dim > (size_t)INT_MAX ||
        flow->freq_embed_dim > (size_t)INT_MAX ||
        flow->num_time_conds > (size_t)INT_MAX ||
        batch_hidden > (size_t)INT_MAX || time_hidden > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA flow batch dimensions overflow");
        return -1;
    }

    /* y = cond_embed(cond) + mean(time_embed(t)).  Time features are supplied
     * by the engine in persistent pinned staging, while all learned time MLPs
     * stay on the device after their first cache lookup. */
    if (cuda_flow_linear(st, flow->dev_cond, flow->dev_y, flow->batch,
                         flow->cond_dim, h, flow->cond_embed, e, ec) != 0)
        return -1;
    for (size_t t = 0; t < flow->num_time_conds; ++t) {
        if (cuda_flow_linear(st, flow->dev_time_embed +
                                 t * flow->freq_embed_dim,
                             flow->dev_time_hidden + t * h, 1u,
                             flow->freq_embed_dim, h, &flow->time_mlp_in[t], e,
                             ec) != 0) return -1;
        k_silu<<<((int)h + 255) / 256, 256, 0, st->stream>>>(
            flow->dev_time_hidden + t * h, (int)h);
        if (ce(cudaGetLastError(), e, ec) ||
            cuda_flow_linear(st, flow->dev_time_hidden + t * h,
                             flow->dev_time_output + t * h, 1u, h, h,
                             &flow->time_mlp_out[t], e, ec) != 0) return -1;
        const float *alpha = flow->time_alpha[t];
        float *d_alpha = nullptr;
        if (alpha == nullptr || cached_weight(st, alpha, h * sizeof(float),
                                              &d_alpha, e, ec)) return -1;
        k_flow_var_rms_norm<<<1, 256, 0, st->stream>>>(
            flow->dev_time_output + t * h, d_alpha,
            flow->dev_time_output + t * h, 1, (int)h, flow->rmsnorm_eps);
        if (ce(cudaGetLastError(), e, ec)) return -1;
        if (t == 0u) {
            k_copy<<<((int)h + 255) / 256, 256, 0, st->stream>>>(
                flow->dev_time_sum, flow->dev_time_output, (int)h);
        } else {
            k_residual_add<<<((int)h + 255) / 256, 256, 0, st->stream>>>(
                flow->dev_time_sum, flow->dev_time_output + t * h, (int)h);
        }
        if (ce(cudaGetLastError(), e, ec)) return -1;
    }
    k_scale<<<((int)h + 255) / 256, 256, 0, st->stream>>>(
        flow->dev_time_sum, 1.0f / (float)flow->num_time_conds, (int)h);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    k_bias_add<<<((int)batch_hidden + 255) / 256, 256, 0, st->stream>>>(
        flow->dev_y, flow->dev_time_sum, (int)flow->batch, (int)h);
    if (ce(cudaGetLastError(), e, ec) ||
        cuda_flow_linear(st, flow->dev_noise, flow->dev_x, flow->batch,
                         flow->latent_dim, h, flow->input_proj, e, ec) != 0)
        return -1;
    k_copy<<<((int)batch_hidden + 255) / 256, 256, 0, st->stream>>>(
        flow->dev_silu, flow->dev_y, (int)batch_hidden);
    k_silu<<<((int)batch_hidden + 255) / 256, 256, 0, st->stream>>>(
        flow->dev_silu, (int)batch_hidden);
    if (ce(cudaGetLastError(), e, ec)) return -1;

    for (size_t b = 0; b < flow->depth; ++b) {
        const mynah_backend_flow_block *block = &flow->blocks[b];
        if (cuda_flow_linear(st, flow->dev_silu, flow->dev_mod, flow->batch,
                             h, three_h, &block->adaln, e, ec) != 0 ||
            cuda_flow_layer_norm(st, flow->dev_x, flow->dev_norm,
                                 block->in_ln_weight, block->in_ln_bias,
                                 flow->batch, h, flow->layernorm_eps, e, ec) != 0)
            return -1;
        k_flow_modulate<<<((int)batch_hidden + 255) / 256, 256, 0, st->stream>>>(
            flow->dev_norm, flow->dev_mod, (int)flow->batch, (int)h,
            (int)three_h);
        if (ce(cudaGetLastError(), e, ec) ||
            cuda_flow_linear(st, flow->dev_norm, flow->dev_hidden, flow->batch,
                             h, h, &block->mlp_in, e, ec) != 0) return -1;
        k_silu<<<((int)batch_hidden + 255) / 256, 256, 0, st->stream>>>(
            flow->dev_hidden, (int)batch_hidden);
        if (ce(cudaGetLastError(), e, ec) ||
            cuda_flow_linear(st, flow->dev_hidden, flow->dev_scratch,
                             flow->batch, h, h, &block->mlp_out, e, ec) != 0)
            return -1;
        k_flow_gate_add<<<((int)batch_hidden + 255) / 256, 256, 0, st->stream>>>(
            flow->dev_x, flow->dev_mod + 2u * h, flow->dev_scratch,
            (int)flow->batch, (int)h, (int)three_h);
        if (ce(cudaGetLastError(), e, ec)) return -1;
    }

    if (cuda_flow_linear(st, flow->dev_silu, flow->dev_final_mod,
                         flow->batch, h, two_h, flow->final_adaln, e, ec) != 0 ||
        cuda_flow_layer_norm(st, flow->dev_x, flow->dev_norm, nullptr, nullptr,
                             flow->batch, h, flow->layernorm_eps, e, ec) != 0)
        return -1;
    k_flow_modulate<<<((int)batch_hidden + 255) / 256, 256, 0, st->stream>>>(
        flow->dev_norm, flow->dev_final_mod, (int)flow->batch, (int)h,
        (int)two_h);
    if (ce(cudaGetLastError(), e, ec) ||
        cuda_flow_linear(st, flow->dev_norm, flow->dev_out, flow->batch, h,
                         flow->latent_dim, flow->final_linear, e, ec) != 0)
        return -1;
    return 0;
}

/* im2col kernel for causal conv1d: builds the columns matrix on GPU.
 * columns[(i*kernel+k)*length + l] = input[i*length + l - shift] or 0. */
__global__ static void k_im2col_causal(const float *__restrict__ input,
                                       float *__restrict__ columns,
                                       int in_ch, int length, int kernel, int dilation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = in_ch * kernel * length;
    if (idx >= total) return;
    int l = idx % length;
    int ik = idx / length;
    int k = ik % kernel;
    int i = ik / kernel;
    int shift = (kernel - 1 - k) * dilation;
    int src = l - shift;
    columns[idx] = (src >= 0) ? input[i * length + src] : 0.0f;
}

extern "C" int mynah_cuda_im2col(void *opaque, const float *input, float *columns,
                       int in_ch, int length, int kernel, int dilation,
                       char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (input == nullptr || columns == nullptr || in_ch <= 0 || length <= 0 ||
        kernel <= 0 || dilation <= 0 ||
        (size_t)in_ch > (size_t)INT_MAX / (size_t)kernel ||
        (size_t)in_ch * (size_t)kernel > (size_t)INT_MAX / (size_t)length) {
        set_error(e, ec, "invalid CUDA im2col dimensions");
        return -1;
    }
    int total = in_ch * kernel * length;
    k_im2col_causal<<<(total+255)/256, 256, 0, st->stream>>>(
        input, columns, in_ch, length, kernel, dilation);
    return ce(cudaGetLastError(), e, ec);
}

/* Broadcast bias: out[o*length+t] = bias[o] for all t. */
__global__ static void k_broadcast_bias(float *out, const float *bias,
                                        int out_ch, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out_ch * length) return;
    out[i] = bias[i / length];
}

/* Full GPU conv1d causal: im2col on GPU + cuBLAS sgemm, single call.
 * Replaces CPU im2col pack + cuda_sgemm (eliminates intermediate copy).
 * weight is [out_ch, in_ch*kernel] row-major (already packed for sgemm). */
extern "C" int mynah_cuda_conv1d(void *opaque,
                       const float *input, float *output,
                       int in_ch, int out_ch, int length,
                       int kernel, int dilation,
                       const float *weight, const float *bias,
                       char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    const size_t inner = (size_t)in_ch * kernel;
    const size_t col_count = inner * length;
    const size_t in_count = (size_t)in_ch * length;
    const size_t out_count = (size_t)out_ch * length;

    /* Cache weight on device. */
    float *dw = nullptr;
    if (cached_weight(st, weight, inner * out_ch * sizeof(float), &dw, e, ec)) return -1;

    /* Scratch: FP32 columns + FP32 output. */
    size_t scratch_need = (col_count + out_count) * sizeof(float);
    if (ensure_scratch(st, scratch_need, e, ec)) return -1;
    float *d_cols = st->dev_scratch;
    float *d_out = st->dev_scratch + col_count;

    /* Upload input via mapped buffer. */
    if (ensure_host(st, in_count * sizeof(float), e, ec)) return -1;
    std::memcpy(st->host_buf, input, in_count * sizeof(float));

    /* GPU im2col. */
    int total = in_ch * kernel * length;
    k_im2col_causal<<<(total+255)/256, 256, 0, st->stream>>>(
        st->dev_buf, d_cols, in_ch, length, kernel, dilation);
    if (ce(cudaGetLastError(), e, ec)) return -1;

    /* Seed output with bias. */
    if (bias != nullptr) {
        float *db = nullptr;
        if (cached_weight(st, bias, out_ch * sizeof(float), &db, e, ec)) return -1;
        k_broadcast_bias<<<((int)out_count+255)/256, 256, 0, st->stream>>>(
            d_out, db, out_ch, length);
        if (ce(cudaGetLastError(), e, ec)) return -1;
    } else {
        cudaMemsetAsync(d_out, 0, out_count * sizeof(float), st->stream);
    }

    /* sgemm: output[out_ch, length] += weight[out_ch, inner] @ columns[inner, length] */
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b1 = 1.0f;
    if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         length, out_ch, (int)inner,
                         &a1,
                         d_cols, CUDA_R_32F, length,
                         dw, CUDA_R_32F, (int)inner,
                         &b1,
                         d_out, CUDA_R_32F, length,
                         cuda_compute_type(st), cuda_gemm_algo(st)), e, ec)) return -1;

    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;

    /* Download output via mapped buffer. */
    size_t out_mapped_need = out_count * sizeof(float);
    if (ensure_host(st, out_mapped_need, e, ec)) return -1;
    cudaMemcpy(st->host_buf, d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost);
    std::memcpy(output, st->host_buf, out_count * sizeof(float));
    return 0;
}

/* Resident causal conv1d.  Unlike mynah_cuda_conv1d(), this function never
 * interprets its activation pointers as host memory, never uses mapped host
 * staging, and never synchronizes.  It is the unit used by the resident
 * NanoCodec graph. */
extern "C" int mynah_cuda_conv1d_dev(void *opaque,
                                     const float *input, float *output,
                                     int in_ch, int out_ch, int length,
                                     int kernel, int dilation,
                                     const float *weight, const float *bias,
                                     char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (input == nullptr || output == nullptr || weight == nullptr ||
        in_ch <= 0 || out_ch <= 0 || length <= 0 || kernel <= 0 || dilation <= 0 ||
        (size_t)in_ch > (size_t)INT_MAX / (size_t)kernel ||
        (size_t)in_ch * (size_t)kernel > (size_t)INT_MAX / (size_t)length ||
        (size_t)out_ch > (size_t)INT_MAX / (size_t)length) {
        set_error(e, ec, "invalid CUDA resident conv1d dimensions");
        return -1;
    }
    const size_t inner = (size_t)in_ch * (size_t)kernel;
    const size_t columns_count = inner * (size_t)length;
    const size_t output_count = (size_t)out_ch * (size_t)length;
    size_t weight_count = 0;
    size_t weight_bytes = 0;
    if (columns_count > SIZE_MAX - output_count ||
        columns_count + output_count > SIZE_MAX / sizeof(float) ||
        !cuda_size_mul(inner, (size_t)out_ch, &weight_count) ||
        !cuda_size_mul(weight_count, sizeof(float), &weight_bytes)) {
        set_error(e, ec, "CUDA resident conv1d workspace overflow");
        return -1;
    }
    float *dw = nullptr;
    if (cached_weight(st, weight, weight_bytes, &dw, e, ec)) return -1;
    float *db = nullptr;
    if (bias != nullptr && cached_weight(st, bias, (size_t)out_ch * sizeof(float),
                                         &db, e, ec)) return -1;
    if (ensure_scratch(st, columns_count * sizeof(float), e, ec)) return -1;
    float *columns = st->dev_scratch;
    const int total = in_ch * kernel * length;
    k_im2col_causal<<<(total + 255) / 256, 256, 0, st->stream>>>(
        input, columns, in_ch, length, kernel, dilation);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (db != nullptr) {
        k_broadcast_bias<<<((int)output_count + 255) / 256, 256, 0, st->stream>>>(
            output, db, out_ch, length);
    } else {
        if (ce(cudaMemsetAsync(output, 0, output_count * sizeof(float), st->stream), e, ec)) return -1;
    }
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (cbe(cublasSetStream(st->cublas, st->stream), e, ec)) return -1;
    const float alpha = 1.0f;
    const float beta = 1.0f;
    if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                        length, out_ch, (int)inner,
                        &alpha, columns, CUDA_R_32F, length,
                        dw, CUDA_R_32F, (int)inner,
                        &beta, output, CUDA_R_32F, length,
                        cuda_compute_type(st), cuda_gemm_algo(st)), e, ec)) return -1;
    return 0;
}

__global__ static void k_conv_transpose(const float *input, const float *weight,
                                        const float *bias, float *output,
                                        int in_ch, int out_ch, int length,
                                        int output_length, int kernel, int stride,
                                        int groups) {
    const int index = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int total = out_ch * output_length;
    if (index >= total) return;
    const int t = index % output_length;
    const int o = index / output_length;
    const int in_per_group = in_ch / groups;
    const int out_per_group = out_ch / groups;
    const int group = o / out_per_group;
    const int out_local = o % out_per_group;
    float value = bias == nullptr ? 0.0f : bias[o];
    for (int k = 0; k < kernel; ++k) {
        if (t < k || ((t - k) % stride) != 0) continue;
        const int input_t = (t - k) / stride;
        if (input_t >= length) continue;
        for (int input_local = 0; input_local < in_per_group; ++input_local) {
            const int input_channel = group * in_per_group + input_local;
            const size_t weight_index =
                ((size_t)input_channel * (size_t)out_per_group + (size_t)out_local) *
                (size_t)kernel + (size_t)k;
            value += input[(size_t)input_channel * (size_t)length + (size_t)input_t] *
                     weight[weight_index];
        }
    }
    output[index] = value;
}

extern "C" int mynah_cuda_conv_transpose_dev(
    void *opaque, const float *input, float *output,
    int in_ch, int out_ch, int length, int output_length,
    int kernel, int stride, int groups,
    const float *weight, const float *bias, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (input == nullptr || output == nullptr || weight == nullptr ||
        in_ch <= 0 || out_ch <= 0 || length <= 0 || output_length <= 0 ||
        kernel <= 0 || stride <= 0 || groups <= 0 || in_ch % groups != 0 ||
        out_ch % groups != 0 || (size_t)out_ch > (size_t)INT_MAX / (size_t)output_length) {
        set_error(e, ec, "invalid CUDA conv-transpose dimensions");
        return -1;
    }
    const size_t out_per_group = (size_t)out_ch / (size_t)groups;
    size_t weight_count = 0;
    size_t weight_bytes = 0;
    if (!cuda_size_mul((size_t)in_ch, out_per_group, &weight_count) ||
        !cuda_size_mul(weight_count, (size_t)kernel, &weight_count) ||
        !cuda_size_mul(weight_count, sizeof(float), &weight_bytes)) {
        set_error(e, ec, "CUDA conv-transpose weight size overflow");
        return -1;
    }
    float *dw = nullptr;
    if (cached_weight(st, weight, weight_bytes, &dw, e, ec)) return -1;
    float *db = nullptr;
    if (bias != nullptr && cached_weight(st, bias, (size_t)out_ch * sizeof(float),
                                         &db, e, ec)) return -1;
    const int total = out_ch * output_length;
    k_conv_transpose<<<(total + 255) / 256, 256, 0, st->stream>>>(
        input, dw, db, output, in_ch, out_ch, length, output_length,
        kernel, stride, groups);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_conv_transpose_causal_step_dev(
    void *opaque, const float *input, float *output, float *partial,
    int channels, int kernel, int stride, const float *weight,
    const float *bias, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t total = 0u;
    size_t weight_count = 0u;
    size_t weight_bytes = 0u;
    size_t bias_bytes = 0u;
    if (st == nullptr || input == nullptr || output == nullptr ||
        weight == nullptr || channels <= 0 ||
        kernel <= 0 || stride <= 0 || kernel < stride ||
        (size_t)channels > (size_t)INT_MAX / (size_t)kernel ||
        !cuda_size_mul((size_t)channels, (size_t)kernel, &total) ||
        total > (size_t)INT_MAX ||
        !cuda_size_mul((size_t)channels, (size_t)kernel, &weight_count) ||
        !cuda_size_mul(weight_count, sizeof(float), &weight_bytes) ||
        !cuda_size_mul((size_t)channels, sizeof(float), &bias_bytes)) {
        set_error(e, ec, "invalid CUDA causal transpose dimensions");
        return -1;
    }
    if (kernel > stride && partial == nullptr) {
        set_error(e, ec, "missing CUDA causal transpose tail");
        return -1;
    }
    float *dw = nullptr;
    if (cached_weight(st, weight, weight_bytes, &dw, e, ec)) return -1;
    float *db = nullptr;
    if (bias != nullptr && cached_weight(st, bias, bias_bytes, &db, e, ec))
        return -1;
    k_conv_transpose_causal_depthwise_step<<<
        (int)((total + 255u) / 256u), 256, 0, st->stream>>>(
        input, output, partial, dw, db, channels, kernel, stride,
        kernel - stride);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_scatter_row_to_channels_dev(
    void *opaque, const float *row, float *output, size_t width,
    size_t length, size_t position, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || row == nullptr || output == nullptr || width == 0u ||
        length == 0u || position >= length || width > (size_t)INT_MAX ||
        length > (size_t)INT_MAX || position > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA row-to-channel scatter dimensions");
        return -1;
    }
    k_scatter_row_to_channels<<<((int)width + 255) / 256, 256, 0, st->stream>>>(
        row, output, (int)width, (int)length, (int)position);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_scatter_rows_to_channels_dev(
    void *opaque, const float *rows, float *const *outputs, size_t batch,
    size_t width, size_t length, size_t position, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t total = 0u;
    if (st == nullptr || rows == nullptr || outputs == nullptr || batch == 0u ||
        batch > st->batch_meta_cap || width == 0u || length == 0u ||
        position >= length || batch > (size_t)INT_MAX || width > (size_t)INT_MAX ||
        length > (size_t)INT_MAX || position > (size_t)INT_MAX ||
        !cuda_size_mul(batch, width, &total) || total > (size_t)INT_MAX) {
        set_error(e, ec, "invalid CUDA batched row-to-channel scatter dimensions");
        return -1;
    }
    for (size_t i = 0; i < batch; ++i) {
        if (outputs[i] == nullptr) {
            set_error(e, ec, "invalid CUDA batched scatter output");
            return -1;
        }
    }
    if (ce(cudaMemcpyAsync(st->dev_batch_k_cache, outputs,
                           batch * sizeof(*outputs), cudaMemcpyHostToDevice,
                           st->stream), e, ec)) return -1;
    k_scatter_rows_to_channels<<<((int)total + 255) / 256, 256, 0, st->stream>>>(
        rows, st->dev_batch_k_cache, (int)batch, (int)width, (int)length,
        (int)position);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_gather_rows_to_batch_dev(
    void *opaque, float *const *inputs, float *rows, size_t batch,
    size_t width, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t total = 0u;
    size_t pointer_bytes = 0u;
    if (st == nullptr || inputs == nullptr || rows == nullptr || batch == 0u ||
        batch > st->batch_meta_cap || width == 0u ||
        batch > (size_t)INT_MAX || width > (size_t)INT_MAX ||
        !cuda_size_mul(batch, width, &total) || total > (size_t)INT_MAX ||
        !cuda_size_mul(batch, sizeof(*inputs), &pointer_bytes)) {
        set_error(e, ec, "invalid CUDA batched row-gather dimensions");
        return -1;
    }
    if (ce(cudaMemcpyAsync(st->dev_batch_k_cache, inputs, pointer_bytes,
                           cudaMemcpyHostToDevice, st->stream), e, ec))
        return -1;
    k_gather_rows_to_batch<<<(int)((total + 255u) / 256u), 256, 0,
                             st->stream>>>(st->dev_batch_k_cache, rows,
                                           (int)batch, (int)width);
    return ce(cudaGetLastError(), e, ec);
}

/* ------------------------------------------------------------------ */
/*  Cross-request Pocket codec gang: quantizer + upsample, PCM collect  */
/* ------------------------------------------------------------------ */

static void codec_gang_release(cuda_backend_state *st) {
    cuda_codec_gang_workspace &w = st->codec_gang;
    cudaFree(w.up_meta_dev);
    if (w.up_meta_host != nullptr) cudaFreeHost(w.up_meta_host);
    if (w.up_meta_event != nullptr) cudaEventDestroy(w.up_meta_event);
    cudaFree(w.up_proj);
    cudaFree(w.pcm_meta_dev);
    if (w.pcm_meta_host != nullptr) cudaFreeHost(w.pcm_meta_host);
    if (w.pcm_meta_event != nullptr) cudaEventDestroy(w.pcm_meta_event);
    cudaFree(w.pcm_dev);
    if (w.pcm_host != nullptr) cudaFreeHost(w.pcm_host);
    w = cuda_codec_gang_workspace();
}

/* Grow one pinned-host + device pair.  Growth drains the stream first: an
 * earlier async copy may still read the old block, and the old device block
 * may still be referenced by queued kernels.  It happens a handful of times
 * per process (the gang only widens), never per frame. */
static int codec_gang_grow_pair(cuda_backend_state *st, void **dev, void **host,
                                size_t *cap, size_t bytes, char *e, size_t ec) {
    if (bytes <= *cap) return 0;
    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    cudaFree(*dev);
    if (*host != nullptr) cudaFreeHost(*host);
    *dev = nullptr;
    *host = nullptr;
    *cap = 0u;
    if (ce(cudaMalloc(dev, bytes), e, ec)) return -1;
    if (host != nullptr &&
        ce(cudaHostAlloc(host, bytes, cudaHostAllocDefault), e, ec)) {
        cudaFree(*dev);
        *dev = nullptr;
        return -1;
    }
    *cap = bytes;
    return 0;
}

extern "C" int mynah_cuda_codec_upsample_batch_dev(
    void *opaque, const mynah_backend_upsample_batch_desc *d, char *e,
    size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || d == nullptr || d->host_input == nullptr ||
        d->output == nullptr || d->proj_weight == nullptr ||
        d->up_weight == nullptr || d->rows == 0u || d->rows > 4096u ||
        d->in_dim == 0u || d->channels == 0u || d->stride == 0u ||
        d->kernel < d->stride || d->in_dim > 65536u ||
        d->channels > 65536u || d->kernel > 4096u) {
        set_error(e, ec, "invalid CUDA codec gang upsample description");
        return -1;
    }
    const size_t tail = d->kernel - d->stride;
    if (tail > 0u && d->partial == nullptr) {
        set_error(e, ec, "missing CUDA codec gang upsample tails");
        return -1;
    }
    for (size_t r = 0; r < d->rows; ++r) {
        if (d->host_input[r] == nullptr || d->output[r] == nullptr ||
            (tail > 0u && d->partial[r] == nullptr)) {
            set_error(e, ec, "invalid CUDA codec gang upsample row");
            return -1;
        }
    }
    size_t input_floats = 0u, input_bytes = 0u, proj_floats = 0u,
           total = 0u, meta_bytes = 0u;
    if (!cuda_size_mul(d->rows, d->in_dim, &input_floats) ||
        !cuda_size_mul(input_floats, sizeof(float), &input_bytes) ||
        !cuda_size_mul(d->rows, d->channels, &proj_floats) ||
        proj_floats > (size_t)INT_MAX ||
        !cuda_size_mul(d->channels, d->kernel, &total) ||
        total > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA codec gang upsample size overflow");
        return -1;
    }
    total = proj_floats;
    meta_bytes = 2u * d->rows * sizeof(void *) + input_bytes;

    float *dpw = nullptr, *dpb = nullptr, *duw = nullptr, *dub = nullptr;
    if (cached_weight(st, d->proj_weight,
                      d->channels * d->in_dim * sizeof(float), &dpw, e, ec) ||
        (d->proj_bias != nullptr &&
         cached_weight(st, d->proj_bias, d->channels * sizeof(float), &dpb, e,
                       ec)) ||
        cached_weight(st, d->up_weight, d->channels * d->kernel * sizeof(float),
                      &duw, e, ec) ||
        (d->up_bias != nullptr &&
         cached_weight(st, d->up_bias, d->channels * sizeof(float), &dub, e,
                       ec)))
        return -1;

    cuda_codec_gang_workspace &w = st->codec_gang;
    if (w.up_meta_event == nullptr &&
        ce(cudaEventCreateWithFlags(&w.up_meta_event, cudaEventDisableTiming),
           e, ec))
        return -1;
    if (codec_gang_grow_pair(st, &w.up_meta_dev, &w.up_meta_host,
                             &w.up_meta_cap, meta_bytes, e, ec))
        return -1;
    if (proj_floats > w.up_proj_cap) {
        if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
        cudaFree(w.up_proj);
        w.up_proj = nullptr;
        w.up_proj_cap = 0u;
        if (ce(cudaMalloc(&w.up_proj, proj_floats * sizeof(float)), e, ec))
            return -1;
        w.up_proj_cap = proj_floats;
    }

    /* One pinned block, one H2D: [outputs][partials][rows x in_dim]. */
    if (ce(cudaEventSynchronize(w.up_meta_event), e, ec)) return -1;
    char *host = static_cast<char *>(w.up_meta_host);
    float **host_out = reinterpret_cast<float **>(host);
    float **host_partial =
        reinterpret_cast<float **>(host + d->rows * sizeof(void *));
    float *host_input =
        reinterpret_cast<float *>(host + 2u * d->rows * sizeof(void *));
    for (size_t r = 0; r < d->rows; ++r) {
        host_out[r] = d->output[r];
        host_partial[r] = tail > 0u ? d->partial[r] : nullptr;
        memcpy(host_input + r * d->in_dim, d->host_input[r],
               d->in_dim * sizeof(float));
    }
    if (ce(cudaMemcpyAsync(w.up_meta_dev, w.up_meta_host, meta_bytes,
                           cudaMemcpyHostToDevice, st->stream), e, ec) ||
        ce(cudaEventRecord(w.up_meta_event, st->stream), e, ec))
        return -1;
    st->h2d_bytes.fetch_add((unsigned long long)meta_bytes,
                            std::memory_order_relaxed);
    st->h2d_calls.fetch_add(1ull, std::memory_order_relaxed);
    char *dev = static_cast<char *>(w.up_meta_dev);
    float *const *d_out = reinterpret_cast<float *const *>(dev);
    float *const *d_partial =
        reinterpret_cast<float *const *>(dev + d->rows * sizeof(void *));
    const float *d_input =
        reinterpret_cast<const float *>(dev + 2u * d->rows * sizeof(void *));

    const dim3 grid((unsigned)((d->channels + 255u) / 256u), (unsigned)d->rows);
    k_matvec_rows<<<grid, 256, 0, st->stream>>>(d_input, dpw, dpb, w.up_proj,
                                                (int)d->in_dim,
                                                (int)d->channels);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    k_conv_transpose_causal_depthwise_step_rows<<<
        (int)((total + 255u) / 256u), 256, 0, st->stream>>>(
        w.up_proj, d_out, d_partial, duw, dub, (int)d->rows, (int)d->channels,
        (int)d->kernel, (int)d->stride, (int)tail);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_gather_rows_d2h(void *opaque,
                                          const float *const *dev_rows,
                                          size_t rows, size_t width,
                                          const float **host_out, char *e,
                                          size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (host_out != nullptr) *host_out = nullptr;
    size_t total = 0u, bytes = 0u;
    if (st == nullptr || dev_rows == nullptr || host_out == nullptr ||
        rows == 0u || rows > 4096u || width == 0u ||
        !cuda_size_mul(rows, width, &total) || total > (size_t)INT_MAX ||
        !cuda_size_mul(total, sizeof(float), &bytes)) {
        set_error(e, ec, "invalid CUDA batched row collect");
        return -1;
    }
    for (size_t r = 0; r < rows; ++r) {
        if (dev_rows[r] == nullptr) {
            set_error(e, ec, "invalid CUDA batched row collect source");
            return -1;
        }
    }
    cuda_codec_gang_workspace &w = st->codec_gang;
    if (w.pcm_meta_event == nullptr &&
        ce(cudaEventCreateWithFlags(&w.pcm_meta_event, cudaEventDisableTiming),
           e, ec))
        return -1;
    if (codec_gang_grow_pair(st, &w.pcm_meta_dev, &w.pcm_meta_host,
                             &w.pcm_meta_cap, rows * sizeof(void *), e, ec))
        return -1;
    if (total > w.pcm_cap) {
        void *dev = w.pcm_dev;
        void *host = w.pcm_host;
        size_t cap_bytes = w.pcm_cap * sizeof(float);
        const int rc = codec_gang_grow_pair(st, &dev, &host, &cap_bytes, bytes,
                                            e, ec);
        w.pcm_dev = static_cast<float *>(dev);
        w.pcm_host = static_cast<float *>(host);
        w.pcm_cap = cap_bytes / sizeof(float);
        if (rc) return -1;
    }
    if (ce(cudaEventSynchronize(w.pcm_meta_event), e, ec)) return -1;
    memcpy(w.pcm_meta_host, dev_rows, rows * sizeof(void *));
    if (ce(cudaMemcpyAsync(w.pcm_meta_dev, w.pcm_meta_host,
                           rows * sizeof(void *), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaEventRecord(w.pcm_meta_event, st->stream), e, ec))
        return -1;
    k_gather_rows_to_batch<<<(int)((total + 255u) / 256u), 256, 0,
                             st->stream>>>(
        static_cast<float *const *>(w.pcm_meta_dev), w.pcm_dev, (int)rows,
        (int)width);
    if (ce(cudaGetLastError(), e, ec) ||
        ce(cudaMemcpyAsync(w.pcm_host, w.pcm_dev, bytes, cudaMemcpyDeviceToHost,
                           st->stream), e, ec))
        return -1;
    st->d2h_bytes.fetch_add((unsigned long long)bytes, std::memory_order_relaxed);
    st->d2h_calls.fetch_add(1ull, std::memory_order_relaxed);
    *host_out = w.pcm_host;
    return 0;
}

/* Model-free parity probe for the codec gang: three requests with distinct
 * inputs and tails, two consecutive frames, through the gang path and through
 * the single-request matvec + causal-step path. The outputs and carried tails
 * must be bit-identical, and the batched collect must return exactly the
 * device rows. */
static int cuda_codec_gang_self_test(cuda_backend_state *st, char *e,
                                     size_t ec) {
    enum { ROWS = 3, IN = 5, CH = 6, KERNEL = 4, STRIDE = 2, TAIL = 2 };
    static float proj_w[CH * IN];
    static float proj_b[CH];
    static float up_w[CH * KERNEL];
    static float up_b[CH];
    for (int i = 0; i < CH * IN; ++i) proj_w[i] = 0.037f * (float)((i * 7) % 11) - 0.19f;
    for (int i = 0; i < CH; ++i) proj_b[i] = 0.013f * (float)i - 0.021f;
    for (int i = 0; i < CH * KERNEL; ++i) up_w[i] = 0.29f - 0.041f * (float)((i * 5) % 9);
    for (int i = 0; i < CH; ++i) up_b[i] = 0.007f * (float)(i + 1);
    float input[2][ROWS][IN];
    for (int f = 0; f < 2; ++f)
        for (int r = 0; r < ROWS; ++r)
            for (int k = 0; k < IN; ++k)
                input[f][r][k] = 0.11f * (float)(k + 1) - 0.07f * (float)r +
                                 0.53f * (float)f - 0.3f;
    const size_t out_n = (size_t)STRIDE * CH;
    const size_t tail_n = (size_t)CH * TAIL;
    /* [gang out x3][gang tail x3][solo out x3][solo tail x3][solo in][solo proj] */
    const size_t floats = 2u * ROWS * (out_n + tail_n) + IN + CH;
    float *base = nullptr;
    if (ce(cudaMalloc(&base, floats * sizeof(float)), e, ec)) return -1;
    int result = -1;
    do {
        float *gang_out[ROWS], *gang_tail[ROWS], *solo_out[ROWS], *solo_tail[ROWS];
        for (int r = 0; r < ROWS; ++r) {
            gang_out[r] = base + (size_t)r * out_n;
            gang_tail[r] = base + ROWS * out_n + (size_t)r * tail_n;
            solo_out[r] = base + ROWS * (out_n + tail_n) + (size_t)r * out_n;
            solo_tail[r] = base + ROWS * (2u * out_n + tail_n) + (size_t)r * tail_n;
        }
        float *solo_in = base + 2u * ROWS * (out_n + tail_n);
        float *solo_proj = solo_in + IN;
        if (ce(cudaMemset(base, 0, floats * sizeof(float)), e, ec)) break;
        /* Give each request a different carried tail before the first frame. */
        float seed_tail[ROWS][CH * TAIL];
        for (int r = 0; r < ROWS; ++r)
            for (int i = 0; i < CH * TAIL; ++i)
                seed_tail[r][i] = 0.05f * (float)(r + 1) - 0.01f * (float)i;
        bool failed = false;
        for (int r = 0; r < ROWS && !failed; ++r)
            failed = ce(cudaMemcpy(gang_tail[r], seed_tail[r], sizeof(seed_tail[r]),
                                   cudaMemcpyHostToDevice), e, ec) ||
                     ce(cudaMemcpy(solo_tail[r], seed_tail[r], sizeof(seed_tail[r]),
                                   cudaMemcpyHostToDevice), e, ec);
        if (failed) break;
        for (int f = 0; f < 2 && !failed; ++f) {
            const float *rows_in[ROWS];
            for (int r = 0; r < ROWS; ++r) rows_in[r] = input[f][r];
            mynah_backend_upsample_batch_desc desc;
            memset(&desc, 0, sizeof(desc));
            desc.rows = ROWS;
            desc.in_dim = IN;
            desc.channels = CH;
            desc.kernel = KERNEL;
            desc.stride = STRIDE;
            desc.host_input = rows_in;
            desc.output = gang_out;
            desc.partial = gang_tail;
            desc.proj_weight = proj_w;
            desc.proj_bias = proj_b;
            desc.up_weight = up_w;
            desc.up_bias = up_b;
            if (mynah_cuda_codec_upsample_batch_dev(st, &desc, e, ec) != 0) {
                failed = true;
                break;
            }
            for (int r = 0; r < ROWS && !failed; ++r) {
                failed = ce(cudaMemcpyAsync(solo_in, input[f][r], sizeof(input[f][r]),
                                            cudaMemcpyHostToDevice, st->stream),
                            e, ec) ||
                         mynah_cuda_matvec_dev(st, solo_in, solo_proj, IN, CH,
                                               proj_w, proj_b, e, ec) != 0 ||
                         mynah_cuda_conv_transpose_causal_step_dev(
                             st, solo_proj, solo_out[r], solo_tail[r], CH, KERNEL,
                             STRIDE, up_w, up_b, e, ec) != 0 ||
                         ce(cudaStreamSynchronize(st->stream), e, ec);
            }
        }
        if (failed || ce(cudaStreamSynchronize(st->stream), e, ec)) break;
        float got[ROWS * (STRIDE * CH + CH * TAIL)];
        float want[ROWS * (STRIDE * CH + CH * TAIL)];
        if (ce(cudaMemcpy(got, base, sizeof(got), cudaMemcpyDeviceToHost), e, ec) ||
            ce(cudaMemcpy(want, base + ROWS * (out_n + tail_n), sizeof(want),
                          cudaMemcpyDeviceToHost), e, ec))
            break;
        if (memcmp(got, want, sizeof(got)) != 0) {
            set_error(e, ec, "CUDA codec gang upsample differs from the single-request path");
            break;
        }
        const float *collected = nullptr;
        const float *collect_rows[ROWS] = {gang_out[2], gang_out[0], gang_out[1]};
        if (mynah_cuda_gather_rows_d2h(st, collect_rows, ROWS, out_n, &collected,
                                       e, ec) != 0 ||
            ce(cudaStreamSynchronize(st->stream), e, ec))
            break;
        static const int order[ROWS] = {2, 0, 1};
        bool same = true;
        for (int r = 0; r < ROWS && same; ++r)
            same = memcmp(collected + (size_t)r * out_n, got + (size_t)order[r] * out_n,
                          out_n * sizeof(float)) == 0;
        if (!same) {
            set_error(e, ec, "CUDA batched row collect self-test mismatch");
            break;
        }
        result = 0;
    } while (false);
    cudaFree(base);
    return result;
}

extern "C" int mynah_cuda_zero_dev(void *opaque, float *data, size_t n,
                                    char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t bytes = 0u;
    if (st == nullptr || data == nullptr || n == 0u ||
        !cuda_size_mul(n, sizeof(float), &bytes)) {
        set_error(e, ec, "invalid CUDA device-zero request");
        return -1;
    }
    return ce(cudaMemsetAsync(data, 0, bytes, st->stream), e, ec);
}

/* ------------------------------------------------------------------ */
/*  Resident Pocket SEANet decoder                                    */
/* ------------------------------------------------------------------ */

static bool decoder_mul(size_t a, size_t b, size_t *out) {
    if (a != 0u && b > SIZE_MAX / a) return false;
    *out = a * b;
    return true;
}

static bool decoder_add(size_t a, size_t b, size_t *out) {
    if (b > SIZE_MAX - a) return false;
    *out = a + b;
    return true;
}

static bool decoder_int(size_t value, int *out) {
    if (value > (size_t)INT_MAX) return false;
    *out = (int)value;
    return true;
}

static void decoder_free_op(cuda_decoder_op *op) {
    if (op == nullptr) return;
    if (op->previous != nullptr) cudaFree(op->previous);
    if (op->window != nullptr) cudaFree(op->window);
    if (op->partial != nullptr) cudaFree(op->partial);
    if (op->full != nullptr) cudaFree(op->full);
    op->previous = nullptr;
    op->window = nullptr;
    op->partial = nullptr;
    op->full = nullptr;
}

static void decoder_destroy(mynah_backend_decoder *decoder) {
    if (decoder == nullptr) return;
    for (auto &op : decoder->ops) decoder_free_op(&op);
    if (decoder->lean) {
        /* The solo scratch is the backend's shared set: not ours to free. */
        if (decoder->gang_a != nullptr) cudaFree(decoder->gang_a);
        if (decoder->gang_b != nullptr) cudaFree(decoder->gang_b);
        if (decoder->gang_columns != nullptr) cudaFree(decoder->gang_columns);
        decoder->gang_a = nullptr;
        decoder->gang_b = nullptr;
        decoder->gang_columns = nullptr;
    } else {
        if (decoder->work_a != nullptr) cudaFree(decoder->work_a);
        if (decoder->work_b != nullptr) cudaFree(decoder->work_b);
        if (decoder->work_c != nullptr) cudaFree(decoder->work_c);
        if (decoder->columns != nullptr) cudaFree(decoder->columns);
    }
    decoder->work_a = nullptr;
    decoder->work_b = nullptr;
    decoder->work_c = nullptr;
    decoder->columns = nullptr;
    delete decoder;
}

static bool decoder_add_op(mynah_backend_decoder *decoder,
                           cuda_decoder_op_kind kind, int pre_elu,
                           size_t in_channels, size_t out_channels,
                           size_t kernel, size_t stride, size_t dilation,
                           size_t groups, size_t max_in_len,
                           const mynah_backend_decoder_weight *weights,
                           size_t *max_work, size_t *max_columns,
                           char *e, size_t ec) {
    cuda_decoder_op op{};
    int launch_range = 0;
    op.kind = kind;
    op.pre_elu = pre_elu;
    op.max_in_len = max_in_len;
    if (!decoder_int(in_channels, &op.in_channels) ||
        !decoder_int(out_channels, &op.out_channels) ||
        !decoder_int(kernel, &op.kernel) || !decoder_int(stride, &op.stride) ||
        !decoder_int(dilation, &op.dilation) || !decoder_int(groups, &op.groups) ||
        weights == nullptr || weights->weight == nullptr || in_channels == 0u ||
        out_channels == 0u || kernel == 0u || stride == 0u || groups == 0u ||
        in_channels % groups != 0u || out_channels % groups != 0u ||
        max_in_len == 0u || max_in_len > (size_t)INT_MAX) {
        set_error(e, ec, "invalid resident decoder operation descriptor");
        return false;
    }
    size_t span = 0u;
    if (!decoder_mul(kernel - 1u, dilation, &span) ||
        !decoder_add(span, 1u, &span) || span < stride) {
        set_error(e, ec, "resident decoder causal span is invalid");
        return false;
    }
    op.tail = span - stride;
    if (!decoder_int(op.tail, &launch_range) ||
        !decoder_int(max_in_len, &launch_range)) {
        /* The operation fields are already validated above; this pair is only
         * a compact launch-range check for the causal state length. */
        set_error(e, ec, "resident decoder state length exceeds CUDA range");
        return false;
    }
    if (kind == CUDA_DECODER_CONVTR) {
        if (kernel < stride || !decoder_mul(max_in_len, stride, &op.max_full_len) ||
            !decoder_add(op.max_full_len, kernel - stride, &op.max_full_len)) {
            set_error(e, ec, "resident decoder transpose shape overflow");
            return false;
        }
        size_t full_elems = 0u;
        if (!decoder_mul(out_channels, op.max_full_len, &full_elems) ||
            full_elems > (size_t)INT_MAX ||
            !decoder_int(op.max_full_len, &launch_range)) {
            set_error(e, ec, "resident decoder transpose launch range overflow");
            return false;
        }
    }

    size_t work = 0u;
    const size_t work_len = kind == CUDA_DECODER_CONVTR
        ? op.max_full_len - op.tail : max_in_len;
    if (!decoder_mul(out_channels, work_len, &work)) {
        set_error(e, ec, "resident decoder work shape overflow");
        return false;
    }
    if (work > (size_t)INT_MAX) {
        set_error(e, ec, "resident decoder work launch range overflow");
        return false;
    }
    if (work > *max_work) *max_work = work;
    if (kind != CUDA_DECODER_CONVTR) {
        if (stride != 1u) {
            set_error(e, ec, "resident decoder only supports stride-one conv1d");
            return false;
        }
        size_t kernel_width = 0u;
        size_t inner = 0u;
        if (!decoder_mul(in_channels, kernel, &kernel_width) ||
            !decoder_int(kernel_width, &launch_range) ||
            !decoder_mul(kernel_width, max_in_len, &inner)) {
            set_error(e, ec, "resident decoder column shape overflow");
            return false;
        }
        if (inner > (size_t)INT_MAX) {
            set_error(e, ec, "resident decoder column launch range overflow");
            return false;
        }
        if (inner > *max_columns) *max_columns = inner;
    }

    size_t weight_count = 0u;
    if (kind == CUDA_DECODER_CONVTR) {
        size_t out_per_group = out_channels / groups;
        if (!decoder_mul(in_channels, out_per_group, &weight_count) ||
            !decoder_mul(weight_count, kernel, &weight_count)) {
            set_error(e, ec, "resident decoder transpose weight overflow");
            return false;
        }
    } else if (!decoder_mul(out_channels, in_channels, &weight_count) ||
               !decoder_mul(weight_count, kernel, &weight_count)) {
        set_error(e, ec, "resident decoder weight overflow");
        return false;
    }
    size_t weight_bytes = 0u;
    if (!decoder_mul(weight_count, sizeof(float), &weight_bytes) ||
        cached_weight(decoder->backend, weights->weight, weight_bytes,
                      &op.weight, e, ec)) return false;
    size_t bias_bytes = 0u;
    if (weights->bias != nullptr &&
        (!decoder_mul(out_channels, sizeof(float), &bias_bytes) ||
         cached_weight(decoder->backend, weights->bias, bias_bytes, &op.bias,
                       e, ec))) return false;
    /* Every conv1d is an im2col GEMM; a transposed conv has a GEMM form only
     * with groups == 1 (the grouped one stays on its FP32 SIMT kernel).  The
     * copy is made here, at decoder open, never during a graph capture: the
     * conversion synchronises the stream once per weight, and every later
     * decoder hits the cache. */
    if (cuda_seanet_bf16_enabled() &&
        ((kind == CUDA_DECODER_CONV && (cuda_seanet_bf16_mask() & 1)) ||
         (kind == CUDA_DECODER_CONVTR && groups == 1u &&
          (cuda_seanet_bf16_mask() & 2))) &&
        cached_weight_bf16(decoder->backend, weights->weight, weight_count,
                           &op.weight_bf16, e, ec) != 0)
        return false;

    size_t n = 0u;
    size_t bytes = 0u;
    if (kind == CUDA_DECODER_CONVTR) {
        if (op.tail > 0u) {
            if (!decoder_mul(out_channels, op.tail, &n) ||
                !decoder_mul(n, sizeof(float), &bytes) ||
                ce(cudaMalloc(&op.partial, bytes), e, ec) ||
                ce(cudaMemset(op.partial, 0, bytes), e, ec)) {
                decoder_free_op(&op);
                return false;
            }
        }
        /* MYNAH_CUDA_ROW_MEM_DIET: a lean decoder takes `full` from the
         * shared solo scratch (sized from max_full_floats). */
        size_t state_n = 0u;
        if (!decoder_mul(out_channels, op.tail, &state_n) ||
            !decoder_add(decoder->state_floats, state_n,
                         &decoder->state_floats) ||
            !decoder_mul(out_channels, op.max_full_len, &n) ||
            !decoder_add(decoder->sum_full_floats, n,
                         &decoder->sum_full_floats) ||
            !decoder_mul(n, sizeof(float), &bytes) ||
            (!decoder->lean &&
             (ce(cudaMalloc(&op.full, bytes), e, ec) ||
              ce(cudaMemset(op.full, 0, bytes), e, ec)))) {
            decoder_free_op(&op);
            return false;
        }
        if (n > decoder->max_full_floats) decoder->max_full_floats = n;
    } else if (op.tail > 0u) {
        size_t window_len = 0u;
        if (!decoder_mul(in_channels, op.tail, &n) ||
            !decoder_add(decoder->state_floats, n, &decoder->state_floats) ||
            !decoder_mul(n, sizeof(float), &bytes) ||
            ce(cudaMalloc(&op.previous, bytes), e, ec) ||
            ce(cudaMemset(op.previous, 0, bytes), e, ec) ||
            !decoder_add(op.tail, max_in_len, &window_len) ||
            !decoder_mul(in_channels, window_len, &n) ||
            !decoder_add(decoder->sum_window_floats, n,
                         &decoder->sum_window_floats) ||
            !decoder_mul(n, sizeof(float), &bytes) ||
            /* Lean: the causal window comes from the shared solo scratch. */
            (!decoder->lean &&
             (ce(cudaMalloc(&op.window, bytes), e, ec) ||
              ce(cudaMemset(op.window, 0, bytes), e, ec)))) {
            decoder_free_op(&op);
            return false;
        }
        if (n > decoder->max_window_floats) decoder->max_window_floats = n;
    }
    try {
        decoder->ops.push_back(op);
    } catch (const std::bad_alloc &) {
        decoder_free_op(&op);
        throw;
    }
    return true;
}

static int decoder_build(mynah_backend_decoder *decoder,
                         const mynah_backend_decoder_desc *desc,
                         char *e, size_t ec) {
    if (desc->channels == 0u || desc->dimension == 0u ||
        desc->n_filters == 0u || desc->n_ratios == 0u || desc->ratios == nullptr ||
        desc->kernel_size == 0u || desc->last_kernel_size == 0u ||
        desc->compress == 0u || desc->first.weight == nullptr ||
        desc->last.weight == nullptr || desc->convtr == nullptr ||
        (desc->n_residual_layers > 0u && desc->blocks == nullptr)) {
        set_error(e, ec, "incomplete resident decoder descriptor");
        return -1;
    }
    size_t stage_ops = 0u;
    size_t block_ops = 0u;
    size_t op_capacity = 0u;
    if (!decoder_mul(desc->n_residual_layers, 3u, &block_ops) ||
        !decoder_add(block_ops, 1u, &stage_ops) ||
        !decoder_mul(desc->n_ratios, stage_ops, &op_capacity) ||
        !decoder_add(op_capacity, 2u, &op_capacity)) {
        set_error(e, ec, "resident decoder topology size overflow");
        return -1;
    }
    try {
        decoder->ops.reserve(op_capacity);
    } catch (const std::bad_alloc &) {
        set_error(e, ec, "out of memory reserving resident decoder topology");
        return -1;
    }
    size_t mult = 1u;
    for (size_t i = 0; i < desc->n_ratios; ++i) {
        if (desc->ratios[i] == 0u || !decoder_mul(mult, 2u, &mult)) {
            set_error(e, ec, "invalid resident decoder ratio");
            return -1;
        }
    }
    size_t channels_here = 0u;
    if (!decoder_mul(mult, desc->n_filters, &channels_here)) {
        set_error(e, ec, "resident decoder channel overflow");
        return -1;
    }
    size_t length = decoder->max_encoder_frames;
    size_t max_work = 0u;
    size_t max_columns = 0u;
    if (!decoder_add_op(decoder, CUDA_DECODER_CONV, 0, desc->dimension,
                        channels_here, desc->kernel_size, 1u, 1u, 1u, length,
                        &desc->first, &max_work, &max_columns, e, ec)) return -1;
    for (size_t stage = 0; stage < desc->n_ratios; ++stage) {
        const size_t ratio = desc->ratios[stage];
        if (channels_here < 2u || channels_here % 2u != 0u) {
            set_error(e, ec, "resident decoder stage channel count is invalid");
            return -1;
        }
        const size_t out_channels = channels_here / 2u;
        size_t convtr_kernel = 0u;
        if (!decoder_mul(ratio, 2u, &convtr_kernel)) {
            set_error(e, ec, "resident decoder kernel shape overflow");
            return -1;
        }
        if (!decoder_add_op(decoder, CUDA_DECODER_CONVTR, 1,
                            channels_here, out_channels, convtr_kernel, ratio,
                            1u, 1u, length, &desc->convtr[stage], &max_work,
                            &max_columns, e, ec)) return -1;
        if (!decoder_mul(length, ratio, &length)) {
            set_error(e, ec, "resident decoder length overflow");
            return -1;
        }
        for (size_t layer = 0; layer < desc->n_residual_layers; ++layer) {
            if (out_channels % desc->compress != 0u) {
                set_error(e, ec, "resident decoder compress does not divide channels");
                return -1;
            }
            const size_t hidden = out_channels / desc->compress;
            size_t dilation = 1u;
            for (size_t d = 0; d < layer; ++d) {
                if (!decoder_mul(dilation, desc->dilation_base, &dilation)) {
                    set_error(e, ec, "resident decoder dilation overflow");
                    return -1;
                }
            }
            const size_t block = stage * desc->n_residual_layers + layer;
            if (!decoder_add_op(decoder, CUDA_DECODER_CONV, 0,
                                out_channels, hidden, desc->residual_kernel_size,
                                1u, dilation, 1u, length,
                                &desc->blocks[block].conv1, &max_work,
                                &max_columns, e, ec) ||
                !decoder_add_op(decoder, CUDA_DECODER_CONV, 0,
                                hidden, out_channels, 1u, 1u, 1u, 1u, length,
                                &desc->blocks[block].conv2, &max_work,
                                &max_columns, e, ec)) return -1;
            /* Replace the two conv ops with one explicit residual op marker.
             * The two stateful conv objects remain the final two entries; the
             * execution loop consumes this topology through the marker below. */
            cuda_decoder_op rb1 = decoder->ops[decoder->ops.size() - 2u];
            cuda_decoder_op rb2 = decoder->ops[decoder->ops.size() - 1u];
            decoder->ops.resize(decoder->ops.size() - 2u);
            cuda_decoder_op marker{};
            marker.kind = CUDA_DECODER_RESBLOCK;
            marker.pre_elu = 0;
            marker.in_channels = rb1.in_channels;
            marker.out_channels = rb1.out_channels;
            marker.kernel = rb1.kernel;
            marker.stride = rb1.stride;
            marker.dilation = rb1.dilation;
            marker.groups = 1;
            marker.max_in_len = length;
            /* The marker owns rb1/rb2 by storing them in adjacent vector
             * entries immediately after it. This keeps the execution order
             * explicit without allocating a second topology structure. */
            decoder->ops.push_back(marker);
            decoder->ops.push_back(rb1);
            decoder->ops.push_back(rb2);
        }
        channels_here = out_channels;
    }
    if (channels_here != desc->n_filters) {
        set_error(e, ec, "resident decoder final channel count mismatch");
        return -1;
    }
    if (!decoder_add_op(decoder, CUDA_DECODER_CONV, 1, desc->n_filters,
                        desc->channels, desc->last_kernel_size, 1u, 1u, 1u,
                        length, &desc->last, &max_work, &max_columns, e, ec))
        return -1;
    decoder->work_floats = max_work;
    decoder->columns_floats = max_columns;
    return 0;
}

static int decoder_alloc_workspace(mynah_backend_decoder *decoder,
                                   char *e, size_t ec) {
    if (decoder->work_floats == 0u || decoder->columns_floats == 0u) {
        set_error(e, ec, "resident decoder workspace is empty");
        return -1;
    }
    size_t work_bytes = 0u;
    size_t columns_bytes = 0u;
    if (!decoder_mul(decoder->work_floats, sizeof(float), &work_bytes) ||
        !decoder_mul(decoder->columns_floats, sizeof(float), &columns_bytes)) {
        set_error(e, ec, "resident decoder workspace size overflow");
        return -1;
    }
    if (ce(cudaMalloc(&decoder->work_a, work_bytes), e, ec) ||
        ce(cudaMalloc(&decoder->work_b, work_bytes), e, ec) ||
        ce(cudaMalloc(&decoder->work_c, work_bytes), e, ec) ||
        ce(cudaMalloc(&decoder->columns, columns_bytes), e, ec)) {
        return -1;
    }
    return 0;
}

/* The one-GEMM decoder shares two backend buffers sized for the widest gang
 * the batch metadata allows. They are reserved when a decoder opens, because
 * a batched call may be captured into a graph, where nothing can allocate. */
static bool decoder_convtr_gemm_enabled(void) {
    static const bool on = cuda_env_enabled("MYNAH_CUDA_DECODER_CONVTR_GEMM", true);
    return on;
}

/* MYNAH_CUDA_DECODER_FUSE (default on; =0 is the rollback to the unfused
 * kernel sequence): in the cross-request decoder, ELU and the causal
 * window are folded into the im2col / gather kernels that read them, the
 * residual add and (see below) the conv bias into the next reader, and the
 * transposed conv's overlap, fold and prefix copy into one pass (same float
 * operations in the same order, so the audio is bit-identical). */
static bool decoder_fuse_enabled(void) {
    static const bool on = cuda_env_enabled("MYNAH_CUDA_DECODER_FUSE", true);
    return on;
}

/* Part of MYNAH_CUDA_DECODER_FUSE (default on with it): a conv1d whose output
 * is read by a fused kernel runs its GEMM with beta = 0 and leaves the bias to
 * that reader instead of k_decoder_bias_batch + beta = 1.  The only part of
 * the fusion that leans on cuBLAS: the same algorithm must be picked for both
 * beta values (no serial split-K that folds C in before the last partial), so
 * it can be switched off alone (MYNAH_CUDA_DECODER_FUSE_BIAS=0). */
static bool decoder_fuse_bias_enabled(void) {
    static const bool on = decoder_fuse_enabled() &&
        cuda_env_enabled("MYNAH_CUDA_DECODER_FUSE_BIAS", true);
    return on;
}

static bool decoder_one_gemm_enabled(void) {
    static const bool on = cuda_env_enabled("MYNAH_CUDA_DECODER_ONEGEMM", false);
    return on;
}

/* MYNAH_CUDA_ROW_MEM_DIET (default 0): per-request device memory that is not
 * request state moves out of the request.  For the SEANet decoder that is
 * everything but the carried causal state (`previous`, `partial`): the
 * single-request scratch becomes one backend-wide set, and the per-row gang
 * scratch keeps only what the fused gang reads and writes (two activation
 * buffers and the im2col columns, at BF16 size when every conv is BF16).
 * Only placement changes, never an operation or its order. */
static bool cuda_row_mem_diet_enabled(void) {
    static const bool on = cuda_env_enabled("MYNAH_CUDA_ROW_MEM_DIET", true);
    return on;
}

/* A lean decoder relies on the gang taking the fused path everywhere (no
 * per-row window, `full` or ELU scratch); decoder_gang_lean_ok re-checks it
 * per call before anything is queued. */
static bool decoder_lean_planned(void) {
    return cuda_row_mem_diet_enabled() && decoder_fuse_enabled() &&
           decoder_convtr_gemm_enabled() && !decoder_one_gemm_enabled();
}

/* Device bytes a decoder of this topology owns without the diet (the old
 * layout): three work buffers, FP32 columns, every op's window and `full`,
 * and the carried state. */
static size_t decoder_legacy_bytes(const mynah_backend_decoder *d) {
    return (3u * d->work_floats + d->columns_floats + d->sum_window_floats +
            d->sum_full_floats + d->state_floats) * sizeof(float);
}

static size_t decoder_owned_bytes(const mynah_backend_decoder *d) {
    if (!d->lean) return decoder_legacy_bytes(d);
    return (2u * d->work_floats + d->state_floats) * sizeof(float) +
           d->gang_columns_bytes;
}

/* Whether the gang of this topology takes the fused path at its own frame
 * count (the shapes the engine submits): every conv1d with a carried tail
 * no longer than its input, the resblock's first conv included, and every
 * transposed conv in the GEMM form with tail <= output length.  The tiny
 * self-test topology (one encoder frame, tail 2) fails this and keeps the
 * private layout.  decoder_gang_lean_ok re-checks per call with the real
 * width, length and buffer caps. */
static bool decoder_topology_lean_ok(const mynah_backend_decoder *d) {
    size_t length = d->max_encoder_frames;
    for (const auto &op : d->ops) {
        if (op.kind == CUDA_DECODER_CONV) {
            if (op.tail > length) return false;
        } else if (op.kind == CUDA_DECODER_CONVTR) {
            size_t output_len = 0u;
            if (op.groups != 1 ||
                !decoder_mul(length, (size_t)op.stride, &output_len) ||
                op.tail > output_len)
                return false;
            length = output_len;
        }
    }
    return true;
}

/* A planned lean decoder whose topology cannot use it (see above): give its
 * ops the private window and `full` decoder_add_op skipped, exactly as
 * without the diet. */
static int decoder_alloc_private_op_scratch(mynah_backend_decoder *decoder,
                                            char *e, size_t ec) {
    for (auto &op : decoder->ops) {
        size_t n = 0u;
        if (op.kind == CUDA_DECODER_CONVTR && op.full == nullptr) {
            n = (size_t)op.out_channels * op.max_full_len;
        } else if (op.kind == CUDA_DECODER_CONV && op.tail > 0u &&
                   op.window == nullptr) {
            n = (size_t)op.in_channels * (op.tail + op.max_in_len);
        } else {
            continue;
        }
        float **dst = op.kind == CUDA_DECODER_CONVTR ? &op.full : &op.window;
        if (ce(cudaMalloc(dst, n * sizeof(float)), e, ec) ||
            ce(cudaMemset(*dst, 0, n * sizeof(float)), e, ec))
            return -1;
    }
    return 0;
}

/* Bind a lean decoder's single-request scratch to the backend's shared set,
 * creating it on first use.  A set too small for this topology (e.g. one
 * made by the tiny self-test decoder) is never resized in place, because
 * captured single-request graphs and the decoders bound to it hold its
 * addresses: a larger set replaces it for new decoders and the old one is
 * retired (kept until cuda_close). */
static int decoder_bind_solo_scratch(mynah_backend_decoder *decoder, char *e,
                                     size_t ec) {
    cuda_backend_state *st = decoder->backend;
    if (st->solo_work_a == nullptr ||
        decoder->work_floats > st->solo_work_cap ||
        decoder->columns_floats > st->solo_columns_cap ||
        decoder->max_window_floats > st->solo_window_cap ||
        decoder->max_full_floats > st->solo_full_cap) {
        const size_t work = std::max(decoder->work_floats, st->solo_work_cap);
        const size_t cols = std::max(decoder->columns_floats, st->solo_columns_cap);
        const size_t win = std::max(decoder->max_window_floats, st->solo_window_cap);
        const size_t full = std::max(decoder->max_full_floats, st->solo_full_cap);
        float *p[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
        const size_t n[6] = {work, work, work, cols, win, full};
        bool ok = true;
        for (size_t i = 0; i < 6u && ok; ++i) {
            if (n[i] == 0u) continue;
            ok = ce(cudaMalloc(&p[i], n[i] * sizeof(float)), e, ec) == 0 &&
                 ce(cudaMemset(p[i], 0, n[i] * sizeof(float)), e, ec) == 0;
        }
        if (ok) {
            try {
                float *old[6] = {st->solo_work_a, st->solo_work_b, st->solo_work_c,
                                 st->solo_columns, st->solo_window, st->solo_full};
                for (float *q : old)
                    if (q != nullptr) st->solo_retired.push_back(q);
            } catch (const std::bad_alloc &) {
                set_error(e, ec, "out of memory retiring the decoder scratch");
                ok = false;
            }
        }
        if (!ok) {
            for (float *q : p) cudaFree(q);
            return -1;
        }
        st->solo_work_a = p[0];
        st->solo_work_b = p[1];
        st->solo_work_c = p[2];
        st->solo_columns = p[3];
        st->solo_window = p[4];
        st->solo_full = p[5];
        st->solo_work_cap = work;
        st->solo_columns_cap = cols;
        st->solo_window_cap = win;
        st->solo_full_cap = full;
    }
    decoder->work_a = st->solo_work_a;
    decoder->work_b = st->solo_work_b;
    decoder->work_c = st->solo_work_c;
    decoder->columns = st->solo_columns;
    decoder->solo_window = st->solo_window;
    decoder->solo_full = st->solo_full;
    return 0;
}

/* Workspace of a lean decoder: the shared solo set plus the private gang set
 * (gang_a, gang_b, gang_columns).  The columns are stored BF16 by every
 * conv whose weight has a BF16 copy, so when all of them do the private
 * columns are allocated at BF16 size (the old buffer was sized in floats and
 * half of it used).  A topology whose gang cannot be fully fused falls back
 * to the private layout (lean = false). */
static int decoder_alloc_lean(mynah_backend_decoder *decoder, char *e,
                              size_t ec) {
    if (decoder->work_floats == 0u || decoder->columns_floats == 0u) {
        set_error(e, ec, "resident decoder workspace is empty");
        return -1;
    }
    if (!decoder_topology_lean_ok(decoder)) {
        decoder->lean = false;
        if (decoder_alloc_private_op_scratch(decoder, e, ec) != 0) return -1;
        return decoder_alloc_workspace(decoder, e, ec);
    }
    if (decoder_bind_solo_scratch(decoder, e, ec) != 0) return -1;
    bool all_bf16 = true;
    for (const auto &op : decoder->ops)
        if (op.kind == CUDA_DECODER_CONV && op.weight_bf16 == nullptr)
            all_bf16 = false;
    decoder->gang_columns_bytes =
        decoder->columns_floats * (all_bf16 ? sizeof(uint16_t) : sizeof(float));
    const size_t work = decoder->work_floats * sizeof(float);
    if (ce(cudaMalloc(&decoder->gang_a, work), e, ec) ||
        ce(cudaMalloc(&decoder->gang_b, work), e, ec) ||
        ce(cudaMalloc(&decoder->gang_columns, decoder->gang_columns_bytes), e,
           ec))
        return -1;
    return 0;
}

/* The per-row buffers of the cross-request gang (the work/columns entries
 * of the pointer tables).  Without the diet they are the decoder's own
 * work/columns buffers, as always. */
static float *decoder_gang_a(const mynah_backend_decoder *d) {
    return d->lean ? d->gang_a : d->work_a;
}

static float *decoder_gang_b(const mynah_backend_decoder *d) {
    return d->lean ? d->gang_b : d->work_b;
}

static float *decoder_gang_c(const mynah_backend_decoder *d) {
    /* Lean: none; only the unfused paths read it, and those are refused
     * for a lean gang by decoder_gang_lean_ok before anything is queued. */
    return d->lean ? nullptr : d->work_c;
}

static float *decoder_gang_columns(const mynah_backend_decoder *d) {
    return d->lean ? static_cast<float *>(d->gang_columns) : d->columns;
}

static int decoder_reserve_one_gemm(cuda_backend_state *st,
                                    const mynah_backend_decoder *decoder,
                                    char *e, size_t ec) {
    if (decoder_convtr_gemm_enabled()) {
        size_t x_need = 0u, y_need = 0u;
        for (const auto &op : decoder->ops) {
            if (op.kind != CUDA_DECODER_CONVTR || op.groups != 1) continue;
            const size_t n = st->batch_meta_cap * op.max_in_len;
            const size_t x = (size_t)op.in_channels * n;
            const size_t y = (size_t)op.out_channels * (size_t)op.kernel * n;
            if (x > x_need) x_need = x;
            if (y > y_need) y_need = y;
        }
        if (x_need > st->dec_tr_x_cap) {
            cudaFree(st->dec_tr_x);
            st->dec_tr_x = nullptr;
            st->dec_tr_x_cap = 0u;
            if (ce(cudaMalloc(&st->dec_tr_x, x_need * sizeof(float)), e, ec)) return -1;
            st->dec_tr_x_cap = x_need;
        }
        if (y_need > st->dec_tr_y_cap) {
            cudaFree(st->dec_tr_y);
            st->dec_tr_y = nullptr;
            st->dec_tr_y_cap = 0u;
            if (ce(cudaMalloc(&st->dec_tr_y, y_need * sizeof(float)), e, ec)) return -1;
            st->dec_tr_y_cap = y_need;
        }
    }
    if (!cuda_env_enabled("MYNAH_CUDA_DECODER_ONEGEMM", false)) return 0;
    size_t cols = 0u, out = 0u;
    if (!decoder_mul(decoder->columns_floats, st->batch_meta_cap, &cols) ||
        !decoder_mul(decoder->work_floats, st->batch_meta_cap, &out)) {
        set_error(e, ec, "decoder one-GEMM reservation overflow");
        return -1;
    }
    if (cols > st->dec_cols_cap) {
        cudaFree(st->dec_cols);
        st->dec_cols = nullptr;
        st->dec_cols_cap = 0u;
        if (ce(cudaMalloc(&st->dec_cols, cols * sizeof(float)), e, ec)) return -1;
        st->dec_cols_cap = cols;
    }
    if (out > st->dec_out_cap) {
        cudaFree(st->dec_out);
        st->dec_out = nullptr;
        st->dec_out_cap = 0u;
        if (ce(cudaMalloc(&st->dec_out, out * sizeof(float)), e, ec)) return -1;
        st->dec_out_cap = out;
    }
    return 0;
}

static int decoder_conv1d(mynah_backend_decoder *decoder, cuda_decoder_op *op,
                          const float *input, float *output, size_t length,
                          char *e, size_t ec) {
    if (length == 0u || length > op->max_in_len || op->stride != 1 ||
        length % (size_t)op->stride != 0u)
        return -1;
    const size_t out_len = length / (size_t)op->stride;
    const size_t window_len = op->tail + length;
    const float *source = input;
    /* MYNAH_CUDA_ROW_MEM_DIET: a lean decoder's window is the shared one. */
    float *window = op->window != nullptr ? op->window : decoder->solo_window;
    if (op->tail > 0u) {
        const size_t total = (size_t)op->in_channels * window_len;
        k_decoder_causal_window<<<((int)total + 255) / 256, 256,
                                  0, decoder->backend->stream>>>(
            op->previous, input, window, op->in_channels, (int)length,
            (int)op->tail);
        if (ce(cudaGetLastError(), e, ec)) return -1;
        source = window;
    }
    const size_t inner = (size_t)op->in_channels * (size_t)op->kernel;
    const size_t columns = inner * out_len;
    /* BF16 columns live in the same per-decoder buffer (sized in floats, so
     * half of it is used). */
    uint16_t *columns_bf16 = op->weight_bf16 != nullptr
        ? reinterpret_cast<uint16_t *>(decoder->columns) : nullptr;
    if (columns_bf16 != nullptr) {
        k_decoder_causal_columns<<<((int)columns + 255) / 256, 256,
                                   0, decoder->backend->stream>>>(
            source, columns_bf16, op->in_channels, (int)out_len, op->kernel,
            op->dilation, op->stride, (int)op->tail);
    } else {
        k_decoder_causal_columns<<<((int)columns + 255) / 256, 256,
                                   0, decoder->backend->stream>>>(
            source, decoder->columns, op->in_channels, (int)out_len, op->kernel,
            op->dilation, op->stride, (int)op->tail);
    }
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (op->bias != nullptr) {
        k_broadcast_bias<<<((int)((size_t)op->out_channels * out_len) + 255) / 256,
                           256, 0, decoder->backend->stream>>>(
            output, op->bias, op->out_channels, (int)out_len);
    } else if (ce(cudaMemsetAsync(output, 0,
                                  (size_t)op->out_channels * out_len * sizeof(float),
                                  decoder->backend->stream), e, ec)) return -1;
    if (ce(cudaGetLastError(), e, ec)) return -1;
    const float alpha = 1.0f;
    const float beta = 1.0f;
    if (columns_bf16 != nullptr) {
        cuda_bf16_math_scope math(decoder->backend->cublas);
        if (cbe(cublasGemmEx(decoder->backend->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)out_len, op->out_channels, (int)inner,
                             &alpha, columns_bf16, CUDA_R_16BF, (int)out_len,
                             op->weight_bf16, CUDA_R_16BF, (int)inner, &beta,
                             output, CUDA_R_32F, (int)out_len,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), e, ec))
            return -1;
    } else if (cbe(cublasGemmEx(decoder->backend->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                         (int)out_len, op->out_channels, (int)inner,
                         &alpha, decoder->columns, CUDA_R_32F, (int)out_len,
                         op->weight, CUDA_R_32F, (int)inner, &beta, output,
                         CUDA_R_32F, (int)out_len,
                         cuda_compute_type(decoder->backend),
                         cuda_gemm_algo(decoder->backend)), e, ec)) return -1;
    if (op->tail > 0u) {
        const size_t total = (size_t)op->in_channels * op->tail;
        k_decoder_copy_tail<<<((int)total + 255) / 256, 256,
                              0, decoder->backend->stream>>>(
            window, op->previous, op->in_channels, (int)length,
            (int)op->tail);
        if (ce(cudaGetLastError(), e, ec)) return -1;
    }
    return 0;
}

static int decoder_convtr(mynah_backend_decoder *decoder, cuda_decoder_op *op,
                          const float *input, float *output, size_t length,
                          char *e, size_t ec) {
    if (length == 0u || length > op->max_in_len) return -1;
    const size_t output_len = length * (size_t)op->stride;
    const size_t full_len = output_len + op->tail;
    const size_t total = (size_t)op->out_channels * full_len;
    cuda_backend_state *backend = decoder->backend;
    /* MYNAH_CUDA_ROW_MEM_DIET: a lean decoder's `full` is the shared one. */
    float *full = op->full != nullptr ? op->full : decoder->solo_full;
    if (op->weight_bf16 != nullptr && decoder_convtr_gemm_enabled() &&
        op->groups == 1 &&
        (size_t)op->in_channels * length <= backend->dec_tr_x_cap &&
        (size_t)op->out_channels * (size_t)op->kernel * length <=
            backend->dec_tr_y_cap) {
        /* BF16 SEANet: the gang's GEMM form at batch 1, so a solo decode
         * rounds exactly like its gang (the gather is the identity layout
         * here, only the BF16 rounding remains). */
        const int m = op->out_channels * op->kernel;
        const int n = (int)length;
        const int x_total = op->in_channels * n;
        uint16_t *x = reinterpret_cast<uint16_t *>(backend->dec_tr_x);
        k_f32_to_bf16<<<(x_total + 255) / 256, 256, 0, backend->stream>>>(
            input, x, x_total);
        if (ce(cudaGetLastError(), e, ec)) return -1;
        const float alpha = 1.0f, beta = 0.0f;
        {
            cuda_bf16_math_scope math(backend->cublas);
            if (cbe(cublasGemmEx(backend->cublas, CUBLAS_OP_N, CUBLAS_OP_T, m,
                                 n, op->in_channels, &alpha, op->weight_bf16,
                                 CUDA_R_16BF, m, x, CUDA_R_16BF, n, &beta,
                                 backend->dec_tr_y, CUDA_R_32F, m,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                    e, ec))
                return -1;
        }
        k_decoder_convtr_overlap_one<<<((int)total + 255) / 256, 256, 0,
                                       backend->stream>>>(
            backend->dec_tr_y, full, op->bias, op->out_channels,
            (int)length, (int)full_len, op->kernel, op->stride);
    } else {
        k_conv_transpose<<<((int)total + 255) / 256, 256, 0, decoder->backend->stream>>>(
            input, op->weight, op->bias, full, op->in_channels,
            op->out_channels, (int)length, (int)full_len, op->kernel, op->stride,
            op->groups);
    }
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (op->tail > 0u) {
        const size_t state = (size_t)op->out_channels * op->tail;
        k_decoder_convtr_fold_save<<<((int)state + 255) / 256, 256,
                                     0, decoder->backend->stream>>>(
            full, op->partial, op->bias, op->out_channels, (int)full_len,
            (int)op->tail);
        if (ce(cudaGetLastError(), e, ec)) return -1;
    }
    k_decoder_copy_prefix<<<((int)((size_t)op->out_channels * output_len) + 255) / 256,
                            256, 0, decoder->backend->stream>>>(
        full, output, op->out_channels, (int)full_len, (int)output_len);
    return ce(cudaGetLastError(), e, ec);
}

/* These helpers are defined with the graph-cache utilities below, but the
 * pointer-table upload path is the first place that needs them.  Keep the
 * declarations here so the CUDA translation unit remains valid with both the
 * older and newer toolkit front ends. */
static float **decoder_graph_table(cuda_decoder_batch_graph_entry *entry,
                                   size_t slot, size_t channel);
static float **decoder_graph_host_table(cuda_decoder_batch_graph_entry *entry,
                                        size_t slot, size_t channel);
static size_t decoder_table_channel(const cuda_backend_state *backend,
                                    float **device);

static int decoder_upload_ptrs(cuda_backend_state *backend, float **device,
                               float *const *host, size_t batch, char *e,
                               size_t ec) {
    if (backend == nullptr || device == nullptr || host == nullptr ||
        batch == 0u || batch > backend->batch_meta_cap) {
        set_error(e, ec, "invalid resident decoder pointer table");
        return -1;
    }
    cuda_decoder_batch_graph_entry *graph =
        backend->active_decoder_batch_graph;
    if (graph != nullptr) {
        const size_t channel = decoder_table_channel(backend, device);
        if (channel >= 4u || batch != graph->batch ||
            backend->active_decoder_upload_slot >= graph->upload_slots) {
            set_error(e, ec, "resident decoder graph pointer-table overflow");
            return -1;
        }
        const size_t slot = backend->active_decoder_upload_slot++;
        if (graph->slot_channels != nullptr) {
            if (graph->layout_recording)
                graph->slot_channels[slot] = (uint8_t)channel;
            else if (graph->slot_channels[slot] != (uint8_t)channel)
                graph->layout_drift = true;
        }
        float **host_table = decoder_graph_host_table(graph, slot, channel);
        std::memcpy(host_table, host, batch * sizeof(*host));
        float **device_table = decoder_graph_table(graph, slot, channel);
        backend->active_decoder_tables[channel] = device_table;
        return ce(cudaMemcpyAsync(device_table, host_table,
                                  batch * sizeof(*host),
                                  cudaMemcpyHostToDevice, backend->stream),
                  e, ec);
    }
    return ce(cudaMemcpyAsync(device, host, batch * sizeof(*host),
                              cudaMemcpyHostToDevice, backend->stream),
              e, ec);
}

static bool decoder_batch_launch_range(size_t elements, int *blocks) {
    if (elements == 0u || elements > (size_t)INT_MAX) return false;
    size_t rounded = 0u;
    if (!decoder_add(elements, 255u, &rounded)) return false;
    const size_t count = rounded / 256u;
    if (count == 0u || count > (size_t)INT_MAX) return false;
    *blocks = (int)count;
    return true;
}

static bool decoder_ops_compatible(const cuda_decoder_op *a,
                                   const cuda_decoder_op *b) {
    return a != nullptr && b != nullptr && a->kind == b->kind &&
           a->pre_elu == b->pre_elu && a->in_channels == b->in_channels &&
           a->out_channels == b->out_channels && a->kernel == b->kernel &&
           a->stride == b->stride && a->dilation == b->dilation &&
           a->groups == b->groups && a->max_in_len == b->max_in_len &&
           a->tail == b->tail && a->max_full_len == b->max_full_len &&
           a->weight == b->weight && a->bias == b->bias &&
           a->weight_bf16 == b->weight_bf16;
}

static bool cuda_decoder_graphs_enabled(void) {
    return cuda_env_enabled("MYNAH_CUDA_DECODER_GRAPHS", true);
}

static bool cuda_decoder_graph_reuse_enabled(void) {
    static const bool on = cuda_env_enabled("MYNAH_CUDA_DECODER_GRAPH_REUSE", true);
    return on;
}

/* MYNAH_CUDA_DECODER_TABLE_PATCH=1 (default off): a same-width gang change
 * rewrites the reused graph's pinned pointer tables from per-decoder columns
 * cached at an earlier real recording, instead of re-recording the whole
 * decoder on the host to produce the same tables.  Needs graph reuse. */
static bool cuda_decoder_table_patch_enabled(void) {
    static const bool on =
        cuda_env_enabled("MYNAH_CUDA_DECODER_TABLE_PATCH", false) &&
        cuda_decoder_graph_reuse_enabled();
    return on;
}

/* Set once if a real recording ever disagrees with a cached column: the
 * "column depends only on the decoder" premise is then wrong for this
 * build, and every later gang change re-records as before. */
static std::atomic<bool> g_decoder_table_patch_broken{false};

/* MYNAH_CUDA_DECODER_VALIDATE_ONCE=1 (default off): a gang row whose
 * compatibility class matches the first row's skips the op-by-op
 * decoder_ops_compatible scan (see mynah_backend_decoder::compat_class). */
static bool cuda_decoder_validate_once_enabled(void) {
    static const bool on =
        cuda_env_enabled("MYNAH_CUDA_DECODER_VALIDATE_ONCE", false);
    return on;
}

/* Every row of a cross-request gang must run the first row's topology on its
 * own buffers: the batched kernels take shapes and weights from the first. */
static bool decoder_gang_rows_ok(cuda_backend_state *backend,
                                 mynah_backend_decoder *const *decoders,
                                 const float *const *inputs,
                                 float *const *outputs, size_t batch) {
    mynah_backend_decoder *first = decoders[0];
    const size_t op_count = first->ops.size();
    const bool once = cuda_decoder_validate_once_enabled();
    for (size_t i = 0; i < batch; ++i) {
        mynah_backend_decoder *decoder = decoders[i];
        if (decoder == nullptr || decoder->backend != backend ||
            decoder->channels != first->channels ||
            decoder->dimension != first->dimension ||
            decoder->n_filters != first->n_filters ||
            decoder->elu_alpha != first->elu_alpha ||
            decoder->max_encoder_frames != first->max_encoder_frames ||
            decoder->ops.size() != op_count || inputs[i] == nullptr ||
            outputs[i] == nullptr)
            return false;
        if (once && decoder->compat_class != 0ull &&
            decoder->compat_class == first->compat_class)
            continue;
        for (size_t op = 0; op < op_count; ++op) {
            if (!decoder_ops_compatible(&first->ops[op], &decoder->ops[op]))
                return false;
        }
        if (once) {
            /* Field equality is an equivalence, so joining the first row's
             * class keeps "same class => compatible" true. */
            if (first->compat_class == 0ull)
                first->compat_class =
                    backend->decoder_compat_next.fetch_add(
                        1ull, std::memory_order_relaxed) + 1ull;
            decoder->compat_class = first->compat_class;
        }
    }
    return true;
}

static size_t decoder_conv1d_upload_count(const cuda_decoder_op *op) {
    /* causal-window pointers, columns pointers, output/weight pointers and
     * the tail copy pointers; the tail-free case has no window or tail copy.
     * The fused path (MYNAH_CUDA_DECODER_FUSE) uploads state, columns, input,
     * residual, output and weight tables: 6, which the tail-free count must
     * cover as well. */
    if (op == nullptr) return 0u;
    return op->tail > 0u ? 9u : decoder_fuse_enabled() ? 6u : 4u;
}

static size_t decoder_convtr_upload_count(const cuda_decoder_op *op) {
    /* full-input/output, optional partial-tail fold, final prefix output. */
    return op != nullptr ? 4u + (op->tail > 0u ? 2u : 0u) : 0u;
}

/* Count the pointer-table uploads made by decoder_step_batch_impl.  The count
 * is used before capture to reserve stable metadata storage; allocating during
 * CUDA stream capture is illegal. */
static bool decoder_batch_upload_bound(const mynah_backend_decoder *decoder,
                                       size_t *out) {
    if (decoder == nullptr || out == nullptr || decoder->ops.empty()) return false;
    size_t count = 0u;
    for (size_t index = 0u; index < decoder->ops.size();) {
        const cuda_decoder_op *op = &decoder->ops[index++];
        size_t add = 0u;
        if (op->kind == CUDA_DECODER_RESBLOCK) {
            if (index + 1u >= decoder->ops.size()) return false;
            const cuda_decoder_op *rb1 = &decoder->ops[index++];
            const cuda_decoder_op *rb2 = &decoder->ops[index++];
            add = 2u + decoder_conv1d_upload_count(rb1) + 2u +
                  decoder_conv1d_upload_count(rb2) + 2u;
        } else {
            add = (op->pre_elu != 0 ? 2u : 0u) +
                  (op->kind == CUDA_DECODER_CONV
                       ? decoder_conv1d_upload_count(op)
                       : op->kind == CUDA_DECODER_CONVTR
                             ? decoder_convtr_upload_count(op)
                             : 0u);
        }
        if (!decoder_add(count, add, &count)) return false;
    }
    *out = count;
    return count != 0u;
}

static void decoder_batch_graph_free(cuda_decoder_batch_graph_entry *entry) {
    if (entry == nullptr) return;
    if (entry->exec != nullptr) cudaGraphExecDestroy(entry->exec);
    if (entry->graph != nullptr) cudaGraphDestroy(entry->graph);
    if (entry->device_tables != nullptr) cudaFree(entry->device_tables);
    if (entry->host_tables != nullptr) cudaFreeHost(entry->host_tables);
    if (entry->done != nullptr) cudaEventDestroy(entry->done);
    delete[] entry->slot_channels;
    entry->slot_channels = nullptr;
    entry->done = nullptr;
    entry->exec = nullptr;
    entry->graph = nullptr;
    entry->device_tables = nullptr;
    entry->host_tables = nullptr;
    delete entry;
}

static void decoder_batch_graph_remove(cuda_backend_state *backend,
                                       cuda_decoder_batch_graph_entry *entry) {
    if (backend == nullptr || entry == nullptr) return;
    for (size_t i = 0u; i < backend->decoder_batch_graphs.size(); ++i) {
        if (backend->decoder_batch_graphs[i] != entry) continue;
        backend->decoder_batch_graphs.erase(
            backend->decoder_batch_graphs.begin() + i);
        decoder_batch_graph_free(entry);
        return;
    }
}

static void destroy_decoder_batch_graphs(cuda_backend_state *backend) {
    if (backend == nullptr) return;
    for (cuda_decoder_batch_graph_entry *entry : backend->decoder_batch_graphs)
        decoder_batch_graph_free(entry);
    backend->decoder_batch_graphs.clear();
    backend->active_decoder_batch_graph = nullptr;
    backend->active_decoder_upload_slot = 0u;
    for (size_t i = 0u; i < 4u; ++i) backend->active_decoder_tables[i] = nullptr;
}

static void destroy_decoder_batch_graphs_for(
    cuda_backend_state *backend, const mynah_backend_decoder *decoder) {
    if (backend == nullptr || decoder == nullptr) return;
    for (size_t i = 0u; i < backend->decoder_batch_graphs.size();) {
        cuda_decoder_batch_graph_entry *entry =
            backend->decoder_batch_graphs[i];
        bool hit = false;
        for (mynah_backend_decoder *row : entry->decoders) {
            if (row == decoder) {
                hit = true;
                break;
            }
        }
        if (!hit) {
            ++i;
            continue;
        }
        if (cuda_decoder_graph_reuse_enabled() && entry->valid &&
            entry->done != nullptr) {
            /* Keep the executable graph; only its rows go stale.  The next
             * gang of this width rewrites the tables before any launch. */
            entry->decoders.clear();
            entry->inputs.clear();
            entry->outputs.clear();
            ++i;
            continue;
        }
        backend->decoder_batch_graphs.erase(
            backend->decoder_batch_graphs.begin() + i);
        decoder_batch_graph_free(entry);
    }
}

static cuda_decoder_batch_graph_entry *find_decoder_batch_graph(
    cuda_backend_state *backend, mynah_backend_decoder *const *decoders,
    const float *const *inputs, float *const *outputs, size_t batch,
    size_t encoder_frames) {
    if (backend == nullptr || decoders == nullptr || inputs == nullptr ||
        outputs == nullptr) return nullptr;
    for (cuda_decoder_batch_graph_entry *entry : backend->decoder_batch_graphs) {
        if (!entry->valid || entry->batch != batch ||
            entry->encoder_frames != encoder_frames ||
            entry->decoders.size() != batch)
            continue;
        bool same = true;
        for (size_t i = 0u; i < batch; ++i) {
            if (entry->decoders[i] != decoders[i] ||
                entry->inputs[i] != inputs[i] ||
                entry->outputs[i] != outputs[i]) {
                same = false;
                break;
            }
        }
        if (same) return entry;
    }
    return nullptr;
}

static cuda_decoder_batch_graph_entry *find_decoder_batch_graph_shape(
    cuda_backend_state *backend, const mynah_backend_decoder *first,
    size_t batch, size_t encoder_frames) {
    if (backend == nullptr || first == nullptr || first->ops.empty())
        return nullptr;
    for (cuda_decoder_batch_graph_entry *entry : backend->decoder_batch_graphs) {
        if (entry->valid && entry->done != nullptr && entry->batch == batch &&
            entry->encoder_frames == encoder_frames &&
            entry->sig_ops == first->ops.size() &&
            entry->sig_weight == first->ops[0].weight)
            return entry;
    }
    return nullptr;
}

static cuda_decoder_batch_graph_entry *decoder_batch_graph_create(
    cuda_backend_state *backend, mynah_backend_decoder *const *decoders,
    const float *const *inputs, float *const *outputs, size_t batch,
    size_t encoder_frames, char *e, size_t ec) {
    if (backend == nullptr || decoders == nullptr || inputs == nullptr ||
        outputs == nullptr || batch < 2u || batch > backend->batch_meta_cap ||
        backend->decoder_batch_graphs.size() >= CUDA_DECODER_GRAPH_CAP)
        return nullptr;
    size_t upload_slots = 0u;
    if (!decoder_batch_upload_bound(decoders[0], &upload_slots) ||
        upload_slots > 4096u) return nullptr;
    size_t table_count = 0u;
    size_t table_bytes = 0u;
    if (!decoder_mul(upload_slots, 4u, &table_count) ||
        !decoder_mul(table_count, batch, &table_count) ||
        !decoder_mul(table_count, sizeof(float *), &table_bytes)) {
        set_error(e, ec, "resident decoder graph metadata size overflow");
        return nullptr;
    }
    auto *entry = new (std::nothrow) cuda_decoder_batch_graph_entry();
    if (entry == nullptr) {
        set_error(e, ec, "out of memory creating resident decoder graph");
        return nullptr;
    }
    entry->batch = batch;
    entry->encoder_frames = encoder_frames;
    entry->graph = nullptr;
    entry->exec = nullptr;
    entry->device_tables = nullptr;
    entry->host_tables = nullptr;
    entry->upload_slots = upload_slots;
    entry->used_slots = 0u;
    entry->sig_ops = decoders[0]->ops.size();
    entry->sig_weight = decoders[0]->ops[0].weight;
    entry->done = nullptr;
    entry->valid = false;
    entry->slot_channels = nullptr;
    entry->layout_key = 0ull;
    entry->layout_recording = false;
    entry->layout_drift = false;
    entry->patch_verified = false;
    try {
        entry->decoders.assign(decoders, decoders + batch);
        entry->inputs.assign(inputs, inputs + batch);
        entry->outputs.assign(outputs, outputs + batch);
    } catch (const std::bad_alloc &) {
        decoder_batch_graph_free(entry);
        set_error(e, ec, "out of memory storing resident decoder graph rows");
        return nullptr;
    }
    if (ce(cudaMalloc((void **)&entry->device_tables, table_bytes), e, ec) != 0 ||
        ce(cudaHostAlloc((void **)&entry->host_tables, table_bytes,
                         cudaHostAllocPortable), e, ec) != 0) {
        decoder_batch_graph_free(entry);
        return nullptr;
    }
    std::memset(entry->host_tables, 0, table_bytes);
    /* Without the channel record the graph simply never patches. */
    if (cuda_decoder_table_patch_enabled())
        entry->slot_channels = new (std::nothrow) uint8_t[upload_slots]();
    try {
        backend->decoder_batch_graphs.push_back(entry);
    } catch (const std::bad_alloc &) {
        decoder_batch_graph_free(entry);
        set_error(e, ec, "out of memory indexing resident decoder graphs");
        return nullptr;
    }
    return entry;
}

static float **decoder_graph_table(cuda_decoder_batch_graph_entry *entry,
                                   size_t slot, size_t channel) {
    return entry->device_tables +
           (slot * 4u + channel) * entry->batch;
}

static float **decoder_graph_host_table(cuda_decoder_batch_graph_entry *entry,
                                        size_t slot, size_t channel) {
    return entry->host_tables +
           (slot * 4u + channel) * entry->batch;
}

static size_t decoder_table_channel(const cuda_backend_state *backend,
                                    float **device) {
    if (backend == nullptr || device == nullptr) return SIZE_MAX;
    if (device == backend->dev_decoder_ptr0) return 0u;
    if (device == backend->dev_decoder_ptr1) return 1u;
    if (device == backend->dev_decoder_ptr2) return 2u;
    if (device == backend->dev_decoder_ptr3) return 3u;
    return SIZE_MAX;
}

/* MYNAH_CUDA_DECODER_TABLE_PATCH.  Every pointer a decoder batch graph uploads
 * for row i is built from that row alone: decoders[i]'s own buffers (work,
 * columns, causal rings, window, `full`), inputs[i], outputs[i], and weights
 * that every compatible row shares.  So once the graph's upload sequence is
 * fixed (slot_channels), row i of every table is a function of
 * (decoder, input, output): that decoder's column.  A same-width gang change
 * then only has to scatter the new rows' columns into the pinned tables the
 * graph's memcpy nodes read; the executable graph is the one already
 * instantiated, as on the re-record path. */
static unsigned long long decoder_table_layout_key(
    const cuda_decoder_batch_graph_entry *entry) {
    unsigned long long h = 1469598103934665603ull; /* FNV-1a */
    auto mix = [&h](unsigned long long v) {
        for (int b = 0; b < 8; ++b) {
            h ^= (v >> (8 * b)) & 0xffull;
            h *= 1099511628211ull;
        }
    };
    mix((unsigned long long)entry->encoder_frames);
    mix((unsigned long long)entry->used_slots);
    for (size_t s = 0u; s < entry->used_slots; ++s)
        mix((unsigned long long)entry->slot_channels[s]);
    return h != 0ull ? h : 1ull;
}

static float *decoder_table_cell(const cuda_decoder_batch_graph_entry *entry,
                                 size_t slot, size_t row) {
    return entry->host_tables[(slot * 4u + entry->slot_channels[slot]) *
                                  entry->batch + row];
}

static bool decoder_table_patch_usable(
    const cuda_decoder_batch_graph_entry *entry) {
    return cuda_decoder_table_patch_enabled() &&
           !g_decoder_table_patch_broken.load(std::memory_order_relaxed) &&
           entry != nullptr && entry->slot_channels != nullptr &&
           entry->layout_key != 0ull && !entry->layout_drift;
}

static bool decoder_table_column_fits(const mynah_backend_decoder *decoder,
                                      const cuda_decoder_batch_graph_entry *entry,
                                      const float *input, const float *output) {
    return decoder != nullptr && decoder->table_layout == entry->layout_key &&
           decoder->table_column.size() == entry->used_slots &&
           decoder->table_input == input && decoder->table_output == output;
}

/* After a real recording (first capture or re-record) of `entry` for this
 * gang: compare the cells it wrote with every row's cached column, then cache
 * the fresh columns.  A row whose column came from another graph or another
 * row position and matches proves the premise for this graph; one mismatch
 * anywhere disproves it and turns patching off for the process. */
static void decoder_table_harvest(cuda_decoder_batch_graph_entry *entry,
                                  mynah_backend_decoder *const *decoders,
                                  const float *const *inputs,
                                  float *const *outputs) {
    if (!decoder_table_patch_usable(entry)) return;
    const size_t slots = entry->used_slots;
    bool informative = false;
    for (size_t i = 0u; i < entry->batch; ++i) {
        const mynah_backend_decoder *decoder = decoders[i];
        if (!decoder_table_column_fits(decoder, entry, inputs[i], outputs[i]))
            continue;
        for (size_t s = 0u; s < slots; ++s) {
            if (decoder->table_column[s] == decoder_table_cell(entry, s, i))
                continue;
            if (!g_decoder_table_patch_broken.exchange(true))
                std::fprintf(stderr,
                             "mynah-tts: warning: MYNAH_CUDA_DECODER_TABLE_PATCH "
                             "disabled: a recorded decoder pointer table differs "
                             "from the cached column (row %zu, slot %zu); gang "
                             "changes re-record the graph as before\n",
                             i, s);
            return;
        }
        if (decoder->table_source != entry || decoder->table_row != i)
            informative = true;
    }
    if (informative) entry->patch_verified = true;
    for (size_t i = 0u; i < entry->batch; ++i) {
        mynah_backend_decoder *decoder = decoders[i];
        try {
            decoder->table_column.resize(slots);
        } catch (const std::bad_alloc &) {
            decoder->table_layout = 0ull;
            continue;
        }
        for (size_t s = 0u; s < slots; ++s)
            decoder->table_column[s] = decoder_table_cell(entry, s, i);
        decoder->table_layout = entry->layout_key;
        decoder->table_input = inputs[i];
        decoder->table_output = outputs[i];
        decoder->table_source = entry;
        decoder->table_row = i;
    }
}

/* Whether `entry` can serve this gang by a table patch: verified, and every
 * row that differs from the gang the tables currently hold has a column. */
static bool decoder_table_patch_ready(
    const cuda_decoder_batch_graph_entry *entry,
    mynah_backend_decoder *const *decoders, const float *const *inputs,
    float *const *outputs) {
    if (!decoder_table_patch_usable(entry) || !entry->patch_verified)
        return false;
    const bool held = entry->decoders.size() == entry->batch &&
                      entry->inputs.size() == entry->batch &&
                      entry->outputs.size() == entry->batch;
    for (size_t i = 0u; i < entry->batch; ++i) {
        if (held && entry->decoders[i] == decoders[i] &&
            entry->inputs[i] == inputs[i] && entry->outputs[i] == outputs[i])
            continue;
        if (!decoder_table_column_fits(decoders[i], entry, inputs[i],
                                       outputs[i]))
            return false;
    }
    return true;
}

/* Scatter the changed rows' columns.  The caller has waited for the graph's
 * previous launch (entry->done): its memcpy nodes read these pinned cells. */
static void decoder_table_patch(cuda_decoder_batch_graph_entry *entry,
                                mynah_backend_decoder *const *decoders,
                                const float *const *inputs,
                                float *const *outputs) {
    const size_t slots = entry->used_slots;
    const bool held = entry->decoders.size() == entry->batch &&
                      entry->inputs.size() == entry->batch &&
                      entry->outputs.size() == entry->batch;
    for (size_t i = 0u; i < entry->batch; ++i) {
        if (held && entry->decoders[i] == decoders[i] &&
            entry->inputs[i] == inputs[i] && entry->outputs[i] == outputs[i])
            continue;
        const mynah_backend_decoder *decoder = decoders[i];
        for (size_t s = 0u; s < slots; ++s)
            entry->host_tables[(s * 4u + entry->slot_channels[s]) * entry->batch +
                               i] = decoder->table_column[s];
    }
}

static float **decoder_current_table(const cuda_backend_state *backend,
                                     float **ordinary) {
    const size_t channel = decoder_table_channel(backend, ordinary);
    if (backend != nullptr && channel < 4u &&
        backend->active_decoder_batch_graph != nullptr &&
        backend->active_decoder_tables[channel] != nullptr)
        return backend->active_decoder_tables[channel];
    return ordinary;
}

static int decoder_elu_batch(cuda_backend_state *backend,
                             float *const *inputs, float *const *outputs,
                             size_t batch, size_t elements, float alpha,
                             char *e, size_t ec) {
    size_t total = 0u;
    int blocks = 0;
    if (!decoder_mul(batch, elements, &total) ||
        !decoder_batch_launch_range(total, &blocks) ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, inputs, batch,
                            e, ec) != 0 ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, outputs, batch,
                            e, ec) != 0)
        return -1;
    k_decoder_elu_batch<<<blocks, 256, 0, backend->stream>>>(
        decoder_current_table(backend, backend->dev_decoder_ptr0),
        decoder_current_table(backend, backend->dev_decoder_ptr1),
        (int)batch,
        (int)elements, alpha);
    return ce(cudaGetLastError(), e, ec);
}

/* add_bias: `add` is a bare GEMM accumulator (beta = 0, fused bias); the
 * bias of the op that produced it is added first, as its epilogue would have
 * (MYNAH_CUDA_DECODER_FUSE). */
static int decoder_residual_batch(cuda_backend_state *backend,
                                  float *const *base, float *const *add,
                                  size_t batch, size_t channels, size_t length,
                                  int add_bias, const float *bias, char *e,
                                  size_t ec) {
    size_t elements = 0u;
    size_t total = 0u;
    int blocks = 0;
    if (!decoder_mul(channels, length, &elements) ||
        !decoder_mul(batch, elements, &total) ||
        !decoder_batch_launch_range(total, &blocks) ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, base, batch,
                            e, ec) != 0 ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, add, batch,
                            e, ec) != 0)
        return -1;
    if (add_bias) {
        k_decoder_residual_bias_batch<<<blocks, 256, 0, backend->stream>>>(
            decoder_current_table(backend, backend->dev_decoder_ptr0),
            decoder_current_table(backend, backend->dev_decoder_ptr1), bias,
            (int)batch, (int)channels, (int)length);
        return ce(cudaGetLastError(), e, ec);
    }
    k_decoder_residual_batch<<<blocks, 256, 0, backend->stream>>>(
        decoder_current_table(backend, backend->dev_decoder_ptr0),
        decoder_current_table(backend, backend->dev_decoder_ptr1),
        (int)batch,
        (int)elements);
    return ce(cudaGetLastError(), e, ec);
}

/* Whether a decoder op of the gang reads its input through the fused
 * kernels (MYNAH_CUDA_DECODER_FUSE), so a deferred bias or residual sum can
 * be resolved there.  Must match the path decoder_conv1d_batch and
 * decoder_convtr_batch take for the same arguments. */
static bool decoder_conv_fused_path(const cuda_decoder_op *op, size_t batch,
                                    size_t length) {
    return op != nullptr && op->kind == CUDA_DECODER_CONV &&
           decoder_fuse_enabled() && !(decoder_one_gemm_enabled() && batch > 1u) &&
           op->tail <= length;
}

static bool decoder_convtr_gemm_path(const cuda_backend_state *backend,
                                     const cuda_decoder_op *op, size_t batch,
                                     size_t length) {
    const size_t n = batch * length;
    return op != nullptr && op->kind == CUDA_DECODER_CONVTR &&
           decoder_convtr_gemm_enabled() && op->groups == 1 &&
           (size_t)op->in_channels * n <= backend->dec_tr_x_cap &&
           (size_t)op->out_channels * (size_t)op->kernel * n <=
               backend->dec_tr_y_cap;
}

static bool decoder_lazy_reader(const cuda_backend_state *backend,
                                const cuda_decoder_op *op, size_t batch,
                                size_t length) {
    if (op == nullptr || !decoder_fuse_enabled()) return false;
    if (op->kind == CUDA_DECODER_CONV)
        return decoder_conv_fused_path(op, batch, length);
    if (op->kind == CUDA_DECODER_CONVTR)
        return decoder_convtr_gemm_path(backend, op, batch, length) &&
               op->tail <= length * (size_t)op->stride;
    return false;
}

/* elu_input: the op reads ELU(input). Fused, the ELU happens inside the
 * column kernel; otherwise ELU(input) is first written to elu_out (which may
 * be `inputs` itself for an in-place ELU) exactly as before the fusion.
 * MYNAH_CUDA_DECODER_FUSE only (fused path, else the call is refused):
 *   resid / in_bias_on / in_bias: the input is resolved on the fly as
 *     resid + (input + in_bias) before the ELU (decoder_lazy_value);
 *   defer_bias: the GEMM runs with beta = 0 and this op's bias is left to the
 *     reader of `outputs`, which must then be told (in_bias_on). */
static int decoder_conv1d_batch(cuda_backend_state *backend,
                                mynah_backend_decoder *const *decoders,
                                cuda_decoder_op *const *ops,
                                float *const *inputs, float *const *outputs,
                                size_t batch, size_t length, int elu_input,
                                float alpha, float *const *elu_out,
                                float *const *resid, int in_bias_on,
                                const float *in_bias, int defer_bias, char *e,
                                size_t ec) {
    if (backend == nullptr || decoders == nullptr || ops == nullptr ||
        inputs == nullptr ||
        outputs == nullptr || batch == 0u || batch > backend->batch_meta_cap ||
        decoders[0] == nullptr || ops[0] == nullptr || length == 0u ||
        length > ops[0]->max_in_len || ops[0]->stride != 1 ||
        length % (size_t)ops[0]->stride != 0u)
        return -1;
    cuda_decoder_op *op = ops[0];
    const bool bf16 = op->weight_bf16 != nullptr;
    const size_t out_len = length / (size_t)op->stride;
    size_t window_len = 0u;
    size_t window_elements = 0u;
    size_t inner = 0u;
    size_t columns = 0u;
    size_t output_elements = 0u;
    if (!decoder_add(op->tail, length, &window_len) ||
        !decoder_mul((size_t)op->in_channels, (size_t)op->kernel, &inner) ||
        !decoder_mul(inner, out_len, &columns) ||
        !decoder_mul((size_t)op->out_channels, out_len, &output_elements) ||
        !decoder_mul(window_len, (size_t)op->in_channels, &window_elements) ||
        !decoder_mul(batch, window_elements, &window_elements) ||
        !decoder_mul(batch, columns, &columns) ||
        !decoder_mul(batch, output_elements, &output_elements)) {
        set_error(e, ec, "resident decoder batched conv shape overflow");
        return -1;
    }
    if (window_elements > (size_t)INT_MAX || columns > (size_t)INT_MAX ||
        output_elements > (size_t)INT_MAX)
        return -1;

    const bool one_gemm = decoder_one_gemm_enabled();
    const bool fused = decoder_conv_fused_path(op, batch, length);
    if (((resid != nullptr || in_bias_on) && !fused) ||
        (defer_bias && (one_gemm && batch > 1u))) {
        set_error(e, ec, "resident decoder fused input on an unfused conv");
        return -1;
    }
    if (elu_input && !fused) {
        size_t elu_elements = 0u;
        if (elu_out == nullptr ||
            !decoder_mul((size_t)op->in_channels, length, &elu_elements) ||
            elu_elements > (size_t)INT_MAX ||
            decoder_elu_batch(backend, inputs, elu_out, batch, elu_elements,
                              alpha, e, ec) != 0)
            return -1;
        inputs = elu_out;
        elu_input = 0;
    }

    float *p0[CUDA_BATCH_META_CAP];
    float *p1[CUDA_BATCH_META_CAP];
    float *p2[CUDA_BATCH_META_CAP];
    float *p3[CUDA_BATCH_META_CAP];
    if (fused) {
        for (size_t i = 0; i < batch; ++i) {
            if (ops[i] == nullptr || (op->tail > 0u && ops[i]->previous == nullptr))
                return -1;
            p0[i] = op->tail > 0u ? ops[i]->previous : inputs[i];
        }
    } else if (op->tail > 0u) {
        for (size_t i = 0; i < batch; ++i) {
            if (ops[i] == nullptr || ops[i]->window == nullptr ||
                ops[i]->previous == nullptr)
                return -1;
            p0[i] = ops[i]->previous;
            p1[i] = inputs[i];
            p2[i] = ops[i]->window;
        }
        int blocks = 0;
        if (!decoder_batch_launch_range(window_elements, &blocks) ||
            decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, p0, batch,
                                e, ec) != 0 ||
            decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, p1, batch,
                                e, ec) != 0 ||
            decoder_upload_ptrs(backend, backend->dev_decoder_ptr2, p2, batch,
                                e, ec) != 0)
            return -1;
        k_decoder_causal_window_batch<<<blocks, 256, 0, backend->stream>>>(
            decoder_current_table(backend, backend->dev_decoder_ptr0),
            decoder_current_table(backend, backend->dev_decoder_ptr1),
            decoder_current_table(backend, backend->dev_decoder_ptr2),
            (int)batch, op->in_channels, (int)length,
            (int)op->tail);
        if (ce(cudaGetLastError(), e, ec) != 0) return -1;
        for (size_t i = 0; i < batch; ++i) p0[i] = ops[i]->window;
    } else {
        for (size_t i = 0; i < batch; ++i) p0[i] = inputs[i];
    }
    if (one_gemm && batch > 1u) {
        size_t cols_need = 0u, out_need = 0u;
        if (!decoder_mul(inner, batch * out_len, &cols_need) ||
            !decoder_mul((size_t)op->out_channels, batch * out_len, &out_need))
            return -1;
        if (cols_need > backend->dec_cols_cap || out_need > backend->dec_out_cap) {
            cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(backend->stream, &capture) != cudaSuccess ||
                capture != cudaStreamCaptureStatusNone) {
                set_error(e, ec, "decoder one-GEMM buffers cannot grow in a capture");
                return -1;
            }
            if (cols_need > backend->dec_cols_cap) {
                cudaFree(backend->dec_cols);
                backend->dec_cols = nullptr;
                backend->dec_cols_cap = 0u;
                if (ce(cudaMalloc(&backend->dec_cols, cols_need * sizeof(float)), e, ec))
                    return -1;
                backend->dec_cols_cap = cols_need;
            }
            if (out_need > backend->dec_out_cap) {
                cudaFree(backend->dec_out);
                backend->dec_out = nullptr;
                backend->dec_out_cap = 0u;
                if (ce(cudaMalloc(&backend->dec_out, out_need * sizeof(float)), e, ec))
                    return -1;
                backend->dec_out_cap = out_need;
            }
        }
        for (size_t i = 0; i < batch; ++i) p2[i] = outputs[i];
        int blocks = 0;
        if (!decoder_batch_launch_range(columns, &blocks) ||
            decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, p0, batch, e,
                                ec) != 0)
            return -1;
        uint16_t *cols_bf16 = op->weight_bf16 != nullptr
            ? reinterpret_cast<uint16_t *>(backend->dec_cols) : nullptr;
        if (cols_bf16 != nullptr) {
            k_decoder_causal_columns_shared<<<blocks, 256, 0, backend->stream>>>(
                decoder_current_table(backend, backend->dev_decoder_ptr0),
                cols_bf16, (int)batch, op->in_channels, (int)out_len,
                op->kernel, op->dilation, op->stride, (int)op->tail);
        } else {
            k_decoder_causal_columns_shared<<<blocks, 256, 0, backend->stream>>>(
                decoder_current_table(backend, backend->dev_decoder_ptr0),
                backend->dec_cols, (int)batch, op->in_channels, (int)out_len,
                op->kernel, op->dilation, op->stride, (int)op->tail);
        }
        if (ce(cudaGetLastError(), e, ec) != 0 ||
            cbe(cublasSetStream(backend->cublas, backend->stream), e, ec) != 0)
            return -1;
        const float alpha = 1.0f, beta = 0.0f;
        const int m = (int)(batch * out_len);
        if (cols_bf16 != nullptr) {
            cuda_bf16_math_scope math(backend->cublas);
            if (cbe(cublasGemmEx(backend->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m,
                                 op->out_channels, (int)inner, &alpha,
                                 cols_bf16, CUDA_R_16BF, m, op->weight_bf16,
                                 CUDA_R_16BF, (int)inner, &beta,
                                 backend->dec_out, CUDA_R_32F, m,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                    e, ec) != 0)
                return -1;
        } else if (cbe(cublasSgemm(backend->cublas, CUBLAS_OP_N, CUBLAS_OP_N, m,
                            op->out_channels, (int)inner, &alpha,
                            backend->dec_cols, m, op->weight, (int)inner, &beta,
                            backend->dec_out, m),
                e, ec) != 0)
            return -1;
        if (!decoder_batch_launch_range(output_elements, &blocks) ||
            decoder_upload_ptrs(backend, backend->dev_decoder_ptr2, p2, batch, e,
                                ec) != 0)
            return -1;
        k_decoder_scatter_bias<<<blocks, 256, 0, backend->stream>>>(
            backend->dec_out,
            decoder_current_table(backend, backend->dev_decoder_ptr2), op->bias,
            (int)batch, op->out_channels, (int)out_len);
        if (ce(cudaGetLastError(), e, ec) != 0) return -1;
    } else {
    for (size_t i = 0; i < batch; ++i) {
        if (decoders[i] == nullptr || decoder_gang_columns(decoders[i]) == nullptr ||
            ops[i] == nullptr)
            return -1;
        p1[i] = decoder_gang_columns(decoders[i]);
        p2[i] = outputs[i];
        /* The pointer tables are float * on the device; with BF16 SEANet the
         * columns and weight entries point at BF16 data (the per-decoder
         * columns buffer is sized in floats, so BF16 uses half of it). */
        p3[i] = bf16 ? reinterpret_cast<float *>(op->weight_bf16) : op->weight;
    }
    int blocks = 0;
    if (!decoder_batch_launch_range(columns, &blocks) ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, p0, batch, e,
                            ec) != 0 ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, p1, batch, e,
                            ec) != 0)
        return -1;
    if (fused) {
        float *pin[CUDA_BATCH_META_CAP];
        for (size_t i = 0; i < batch; ++i) pin[i] = inputs[i];
        if (decoder_upload_ptrs(backend, backend->dev_decoder_ptr2, pin, batch,
                                e, ec) != 0)
            return -1;
        /* The residual table rides on channel 3 until the weight table
         * replaces it below; both kernels that read it are launched first. */
        float *const *resid_table = nullptr;
        if (resid != nullptr) {
            if (decoder_upload_ptrs(backend, backend->dev_decoder_ptr3, resid,
                                    batch, e, ec) != 0)
                return -1;
            resid_table = decoder_current_table(backend, backend->dev_decoder_ptr3);
        }
        if (bf16) {
            k_decoder_causal_columns_fused<<<blocks, 256, 0, backend->stream>>>(
                decoder_current_table(backend, backend->dev_decoder_ptr0),
                decoder_current_table(backend, backend->dev_decoder_ptr2),
                reinterpret_cast<uint16_t *const *>(
                    decoder_current_table(backend, backend->dev_decoder_ptr1)),
                (int)batch, op->in_channels, (int)out_len, op->kernel,
                op->dilation, (int)op->tail, elu_input, alpha, resid_table,
                in_bias, in_bias_on);
        } else {
            k_decoder_causal_columns_fused<<<blocks, 256, 0, backend->stream>>>(
                decoder_current_table(backend, backend->dev_decoder_ptr0),
                decoder_current_table(backend, backend->dev_decoder_ptr2),
                decoder_current_table(backend, backend->dev_decoder_ptr1),
                (int)batch, op->in_channels, (int)out_len, op->kernel,
                op->dilation, (int)op->tail, elu_input, alpha, resid_table,
                in_bias, in_bias_on);
        }
        if (ce(cudaGetLastError(), e, ec) != 0) return -1;
        /* The carried state moves now, before the bias and the GEMM write
         * the output, which may be the input buffer itself. */
        if (op->tail > 0u) {
            size_t tail_elements = 0u;
            int tail_blocks = 0;
            if (!decoder_mul(batch, (size_t)op->in_channels, &tail_elements) ||
                !decoder_mul(tail_elements, op->tail, &tail_elements) ||
                !decoder_batch_launch_range(tail_elements, &tail_blocks))
                return -1;
            k_decoder_copy_tail_fused<<<tail_blocks, 256, 0, backend->stream>>>(
                decoder_current_table(backend, backend->dev_decoder_ptr2),
                decoder_current_table(backend, backend->dev_decoder_ptr0),
                (int)batch, op->in_channels, (int)length, (int)op->tail,
                elu_input, alpha, resid_table, in_bias, in_bias_on);
            if (ce(cudaGetLastError(), e, ec) != 0) return -1;
        }
    } else if (bf16) {
        k_decoder_causal_columns_batch<<<blocks, 256, 0, backend->stream>>>(
            decoder_current_table(backend, backend->dev_decoder_ptr0),
            reinterpret_cast<uint16_t *const *>(
                decoder_current_table(backend, backend->dev_decoder_ptr1)),
            (int)batch,
            op->in_channels, (int)out_len, op->kernel, op->dilation, op->stride,
            (int)op->tail);
    } else {
    k_decoder_causal_columns_batch<<<blocks, 256, 0, backend->stream>>>(
        decoder_current_table(backend, backend->dev_decoder_ptr0),
        decoder_current_table(backend, backend->dev_decoder_ptr1),
        (int)batch,
        op->in_channels, (int)out_len, op->kernel, op->dilation, op->stride,
        (int)op->tail);
    }
    if (ce(cudaGetLastError(), e, ec) != 0 ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr2, p2, batch, e,
                            ec) != 0)
        return -1;
    if (!defer_bias) {
        if (!decoder_batch_launch_range(output_elements, &blocks)) return -1;
        k_decoder_bias_batch<<<blocks, 256, 0, backend->stream>>>(
            decoder_current_table(backend, backend->dev_decoder_ptr2), op->bias,
            (int)batch, op->out_channels,
            (int)out_len);
        if (ce(cudaGetLastError(), e, ec) != 0) return -1;
    }
    if (decoder_upload_ptrs(backend, backend->dev_decoder_ptr3, p3, batch, e,
                            ec) != 0 ||
        cbe(cublasSetStream(backend->cublas, backend->stream), e, ec) != 0)
        return -1;
    const float alpha = 1.0f;
    /* beta = 0 with a deferred bias: the bare accumulator, the bias is added
     * by the reader (decoder_lazy_value / k_decoder_residual_bias_batch). */
    const float beta = defer_bias ? 0.0f : 1.0f;
    if (bf16) {
        /* Same batched GEMM, BF16 operands on tensor cores, FP32 accumulate,
         * FP32 output on top of the bias written above (beta = 1). */
        cuda_bf16_math_scope math(backend->cublas);
        if (cbe(cublasGemmBatchedEx(
                    backend->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)out_len, op->out_channels, (int)inner, &alpha,
                    (const void *const *)decoder_current_table(
                        backend, backend->dev_decoder_ptr1), CUDA_R_16BF,
                    (int)out_len,
                    (const void *const *)decoder_current_table(
                        backend, backend->dev_decoder_ptr3), CUDA_R_16BF,
                    (int)inner, &beta,
                    (void *const *)decoder_current_table(
                        backend, backend->dev_decoder_ptr2), CUDA_R_32F,
                    (int)out_len, (int)batch, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT),
                e, ec) != 0)
            return -1;
    } else if (cbe(cublasSgemmBatched(
                backend->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)out_len, op->out_channels, (int)inner, &alpha,
                (const float *const *)decoder_current_table(
                    backend, backend->dev_decoder_ptr1), (int)out_len,
                (const float *const *)decoder_current_table(
                    backend, backend->dev_decoder_ptr3), (int)inner,
                &beta, decoder_current_table(backend, backend->dev_decoder_ptr2),
                (int)out_len,
                (int)batch),
            e, ec) != 0)
        return -1;
    }
    if (op->tail > 0u && !fused) {
        int blocks = 0;
        for (size_t i = 0; i < batch; ++i) {
            p0[i] = ops[i]->window;
            p1[i] = ops[i]->previous;
        }
        if (decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, p0, batch,
                                e, ec) != 0 ||
            decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, p1, batch,
                                e, ec) != 0)
            return -1;
        size_t tail_elements = 0u;
        if (!decoder_mul(batch, (size_t)op->in_channels, &tail_elements) ||
            !decoder_mul(tail_elements, op->tail, &tail_elements))
            return -1;
        if (!decoder_batch_launch_range(tail_elements, &blocks)) return -1;
        k_decoder_copy_tail_batch<<<blocks, 256, 0, backend->stream>>>(
            decoder_current_table(backend, backend->dev_decoder_ptr0),
            decoder_current_table(backend, backend->dev_decoder_ptr1),
            (int)batch,
            op->in_channels, (int)length, (int)op->tail);
        if (ce(cudaGetLastError(), e, ec) != 0) return -1;
    }
    return 0;
}

/* elu_input: the op reads ELU(input), applied in place first unless the
 * GEMM gather can apply it on the fly (MYNAH_CUDA_DECODER_FUSE).  With the
 * fusion on the GEMM path the gather also resolves a deferred residual / bias
 * of the producer (resid, in_bias_on, in_bias; refused elsewhere), and one
 * kernel writes `outputs` and folds the carried partial without `full`. */
static int decoder_convtr_batch(cuda_backend_state *backend,
                                cuda_decoder_op *const *ops,
                                float *const *inputs, float *const *outputs,
                                size_t batch, size_t length, int elu_input,
                                float alpha, float *const *resid,
                                int in_bias_on, const float *in_bias, char *e,
                                size_t ec) {
    if (backend == nullptr || ops == nullptr || inputs == nullptr ||
        outputs == nullptr || batch == 0u || batch > backend->batch_meta_cap ||
        ops[0] == nullptr || length == 0u || length > ops[0]->max_in_len)
        return -1;
    cuda_decoder_op *op = ops[0];
    size_t output_len = 0u;
    size_t full_len = 0u;
    if (!decoder_mul(length, (size_t)op->stride, &output_len) ||
        !decoder_add(output_len, op->tail, &full_len)) {
        set_error(e, ec, "resident decoder batched transpose shape overflow");
        return -1;
    }
    size_t full_elements = 0u;
    if (!decoder_mul((size_t)op->out_channels, full_len, &full_elements) ||
        !decoder_mul(batch, full_elements, &full_elements) ||
        full_elements > (size_t)INT_MAX)
        return -1;
    const size_t gemm_n = batch * length;
    const bool gemm_path = decoder_convtr_gemm_path(backend, op, batch, length);
    const bool fused = gemm_path && decoder_fuse_enabled();
    /* One pass for overlap + fold + prefix copy (see
     * k_decoder_convtr_overlap_out); needs tail <= output_len. */
    const bool fused_out = fused && op->tail <= output_len;
    if ((resid != nullptr || in_bias_on) && !fused_out) {
        set_error(e, ec, "resident decoder fused input on an unfused transpose");
        return -1;
    }
    const int gather_elu = elu_input && fused;
    if (elu_input && !gather_elu) {
        size_t elu_elements = 0u;
        if (!decoder_mul((size_t)op->in_channels, length, &elu_elements) ||
            elu_elements > (size_t)INT_MAX ||
            decoder_elu_batch(backend, inputs, inputs, batch, elu_elements,
                              alpha, e, ec) != 0)
            return -1;
    }
    float *p0[CUDA_BATCH_META_CAP];
    float *p1[CUDA_BATCH_META_CAP];
    if (fused_out) {
        /* No `full`: gather input (+ residual), GEMM, then one kernel writes
         * the destination and carries the partial tail. */
        size_t out_elements = 0u;
        int out_blocks = 0;
        for (size_t i = 0; i < batch; ++i) {
            if (ops[i] == nullptr || (op->tail > 0u && ops[i]->partial == nullptr))
                return -1;
            p0[i] = inputs[i];
            p1[i] = outputs[i];
        }
        if (!decoder_mul(batch, (size_t)op->out_channels, &out_elements) ||
            !decoder_mul(out_elements, output_len, &out_elements) ||
            out_elements > (size_t)INT_MAX ||
            !decoder_batch_launch_range(out_elements, &out_blocks) ||
            decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, p0, batch, e,
                                ec) != 0)
            return -1;
        float *const *resid_table = nullptr;
        if (resid != nullptr) {
            if (decoder_upload_ptrs(backend, backend->dev_decoder_ptr3, resid,
                                    batch, e, ec) != 0)
                return -1;
            resid_table = decoder_current_table(backend, backend->dev_decoder_ptr3);
        }
        const int m = op->out_channels * op->kernel;
        const int n = (int)gemm_n;
        const int x_total = op->in_channels * n;
        uint16_t *x_bf16 = op->weight_bf16 != nullptr
            ? reinterpret_cast<uint16_t *>(backend->dec_tr_x) : nullptr;
        if (x_bf16 != nullptr) {
            k_decoder_convtr_gather<<<(x_total + 255) / 256, 256, 0, backend->stream>>>(
                decoder_current_table(backend, backend->dev_decoder_ptr0),
                x_bf16, (int)batch, op->in_channels, (int)length, gather_elu,
                alpha, resid_table, in_bias, in_bias_on);
        } else {
            k_decoder_convtr_gather<<<(x_total + 255) / 256, 256, 0, backend->stream>>>(
                decoder_current_table(backend, backend->dev_decoder_ptr0),
                backend->dec_tr_x, (int)batch, op->in_channels, (int)length,
                gather_elu, alpha, resid_table, in_bias, in_bias_on);
        }
        if (ce(cudaGetLastError(), e, ec) != 0 ||
            cbe(cublasSetStream(backend->cublas, backend->stream), e, ec) != 0)
            return -1;
        const float one = 1.0f, zero = 0.0f;
        if (x_bf16 != nullptr) {
            cuda_bf16_math_scope math(backend->cublas);
            if (cbe(cublasGemmEx(backend->cublas, CUBLAS_OP_N, CUBLAS_OP_T, m,
                                 n, op->in_channels, &one, op->weight_bf16,
                                 CUDA_R_16BF, m, x_bf16, CUDA_R_16BF, n, &zero,
                                 backend->dec_tr_y, CUDA_R_32F, m,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                    e, ec) != 0)
                return -1;
        } else if (cbe(cublasGemmEx(backend->cublas, CUBLAS_OP_N, CUBLAS_OP_T, m, n,
                             op->in_channels, &one, op->weight, CUDA_R_32F, m,
                             backend->dec_tr_x, CUDA_R_32F, n, &zero,
                             backend->dec_tr_y, CUDA_R_32F, m,
                             cuda_compute_type(backend), cuda_gemm_algo(backend)),
                e, ec) != 0)
            return -1;
        float *const *partial_table = nullptr;
        if (op->tail > 0u) {
            float *pp[CUDA_BATCH_META_CAP];
            for (size_t i = 0; i < batch; ++i) pp[i] = ops[i]->partial;
            if (decoder_upload_ptrs(backend, backend->dev_decoder_ptr2, pp,
                                    batch, e, ec) != 0)
                return -1;
            partial_table = decoder_current_table(backend, backend->dev_decoder_ptr2);
        }
        if (decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, p1, batch, e,
                                ec) != 0)
            return -1;
        k_decoder_convtr_overlap_out<<<out_blocks, 256, 0, backend->stream>>>(
            backend->dec_tr_y,
            decoder_current_table(backend, backend->dev_decoder_ptr1),
            partial_table, op->bias, (int)batch, op->out_channels, (int)length,
            (int)output_len, op->kernel, op->stride, (int)op->tail);
        return ce(cudaGetLastError(), e, ec);
    }
    for (size_t i = 0; i < batch; ++i) {
        if (ops[i] == nullptr || ops[i]->full == nullptr) return -1;
        p0[i] = inputs[i];
        p1[i] = ops[i]->full;
    }
    int blocks = 0;
    if (!decoder_batch_launch_range(full_elements, &blocks) ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, p0, batch, e,
                            ec) != 0 ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, p1, batch, e,
                            ec) != 0)
        return -1;
    if (gemm_path) {
        const int m = op->out_channels * op->kernel;
        const int n = (int)gemm_n;
        const int x_total = op->in_channels * n;
        /* BF16 SEANet: the gather rounds the input to BF16 into the same
         * buffer (sized in floats, half of it used). */
        uint16_t *x_bf16 = op->weight_bf16 != nullptr
            ? reinterpret_cast<uint16_t *>(backend->dec_tr_x) : nullptr;
        if (x_bf16 != nullptr) {
            k_decoder_convtr_gather<<<(x_total + 255) / 256, 256, 0, backend->stream>>>(
                decoder_current_table(backend, backend->dev_decoder_ptr0),
                x_bf16, (int)batch, op->in_channels, (int)length, gather_elu,
                alpha);
        } else {
            k_decoder_convtr_gather<<<(x_total + 255) / 256, 256, 0, backend->stream>>>(
                decoder_current_table(backend, backend->dev_decoder_ptr0),
                backend->dec_tr_x, (int)batch, op->in_channels, (int)length,
                gather_elu, alpha);
        }
        if (ce(cudaGetLastError(), e, ec) != 0 ||
            cbe(cublasSetStream(backend->cublas, backend->stream), e, ec) != 0)
            return -1;
        const float alpha = 1.0f, beta = 0.0f;
        if (x_bf16 != nullptr) {
            cuda_bf16_math_scope math(backend->cublas);
            if (cbe(cublasGemmEx(backend->cublas, CUBLAS_OP_N, CUBLAS_OP_T, m,
                                 n, op->in_channels, &alpha, op->weight_bf16,
                                 CUDA_R_16BF, m, x_bf16, CUDA_R_16BF, n, &beta,
                                 backend->dec_tr_y, CUDA_R_32F, m,
                                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                    e, ec) != 0)
                return -1;
        } else if (cbe(cublasGemmEx(backend->cublas, CUBLAS_OP_N, CUBLAS_OP_T, m, n,
                             op->in_channels, &alpha, op->weight, CUDA_R_32F, m,
                             backend->dec_tr_x, CUDA_R_32F, n, &beta,
                             backend->dec_tr_y, CUDA_R_32F, m,
                             cuda_compute_type(backend), cuda_gemm_algo(backend)),
                e, ec) != 0)
            return -1;
        k_decoder_convtr_overlap<<<blocks, 256, 0, backend->stream>>>(
            backend->dec_tr_y,
            decoder_current_table(backend, backend->dev_decoder_ptr1), op->bias,
            (int)batch, op->out_channels, (int)length, (int)full_len, op->kernel,
            op->stride);
        if (ce(cudaGetLastError(), e, ec) != 0) return -1;
    } else {
    k_decoder_convtr_batch<<<blocks, 256, 0, backend->stream>>>(
        decoder_current_table(backend, backend->dev_decoder_ptr0),
        decoder_current_table(backend, backend->dev_decoder_ptr1), op->weight,
        op->bias, (int)batch, op->in_channels, op->out_channels, (int)length,
        (int)full_len, op->kernel, op->stride, op->groups);
    if (ce(cudaGetLastError(), e, ec) != 0) return -1;
    }
    if (op->tail > 0u) {
        for (size_t i = 0; i < batch; ++i) {
            p0[i] = ops[i]->full;
            p1[i] = ops[i]->partial;
        }
        size_t tail_elements = 0u;
        if (!decoder_mul(batch, (size_t)op->out_channels, &tail_elements) ||
            !decoder_mul(tail_elements, op->tail, &tail_elements) ||
            !decoder_batch_launch_range(tail_elements, &blocks) ||
            decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, p0, batch,
                                e, ec) != 0 ||
            decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, p1, batch,
                                e, ec) != 0)
            return -1;
        k_decoder_convtr_fold_batch<<<blocks, 256, 0, backend->stream>>>(
            decoder_current_table(backend, backend->dev_decoder_ptr0),
            decoder_current_table(backend, backend->dev_decoder_ptr1), op->bias,
            (int)batch, op->out_channels, (int)full_len, (int)op->tail);
        if (ce(cudaGetLastError(), e, ec) != 0) return -1;
    }
    for (size_t i = 0; i < batch; ++i) {
        p0[i] = ops[i]->full;
        p1[i] = outputs[i];
    }
    size_t output_elements = 0u;
    if (!decoder_mul(batch, (size_t)op->out_channels, &output_elements) ||
        !decoder_mul(output_elements, output_len, &output_elements) ||
        !decoder_batch_launch_range(output_elements, &blocks) ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, p0, batch, e,
                            ec) != 0 ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, p1, batch, e,
                            ec) != 0)
        return -1;
    k_decoder_copy_prefix_batch<<<blocks, 256, 0, backend->stream>>>(
        decoder_current_table(backend, backend->dev_decoder_ptr0),
        decoder_current_table(backend, backend->dev_decoder_ptr1),
        (int)batch,
        op->out_channels, (int)full_len, (int)output_len);
    return ce(cudaGetLastError(), e, ec);
}

static int decoder_step_impl(mynah_backend_decoder *decoder,
                             const float *input, size_t encoder_frames,
                             float *output, char *e, size_t ec) {
    if (decoder == nullptr || input == nullptr || output == nullptr ||
        encoder_frames == 0u || encoder_frames > decoder->max_encoder_frames) {
        set_error(e, ec, "invalid resident decoder step");
        return -1;
    }
    if (cbe(cublasSetStream(decoder->backend->cublas, decoder->backend->stream),
            e, ec)) return -1;
    const float *current = input;
    size_t length = encoder_frames;
    size_t channels = decoder->dimension;
    for (size_t index = 0; index < decoder->ops.size();) {
        cuda_decoder_op *op = &decoder->ops[index++];
        if (op->kind == CUDA_DECODER_RESBLOCK) {
            if (index + 1u >= decoder->ops.size()) {
                set_error(e, ec, "resident decoder residual topology is truncated");
                return -1;
            }
            cuda_decoder_op *rb1 = &decoder->ops[index++];
            cuda_decoder_op *rb2 = &decoder->ops[index++];
            const size_t n = channels * length;
            float *other = current == decoder->work_a ? decoder->work_b
                         : current == decoder->work_b ? decoder->work_a
                         : decoder->work_a;
            k_decoder_elu<<<((int)n + 255) / 256, 256, 0, decoder->backend->stream>>>(
                current, decoder->work_c, decoder->elu_alpha, (int)n);
            if (ce(cudaGetLastError(), e, ec) ||
                decoder_conv1d(decoder, rb1, decoder->work_c, other, length, e, ec) != 0)
                return -1;
            k_decoder_elu<<<((int)n + 255) / 256, 256, 0, decoder->backend->stream>>>(
                other, other, decoder->elu_alpha, (int)(rb1->out_channels * length));
            if (ce(cudaGetLastError(), e, ec) ||
                decoder_conv1d(decoder, rb2, other, decoder->work_c, length, e, ec) != 0)
                return -1;
            k_residual_add<<<((int)n + 255) / 256, 256, 0, decoder->backend->stream>>>(
                const_cast<float *>(current), decoder->work_c, (int)n);
            if (ce(cudaGetLastError(), e, ec)) return -1;
            continue;
        }
        const bool last = index == decoder->ops.size();
        float *destination = last ? output
                                  : (current == decoder->work_a
                                         ? decoder->work_b : decoder->work_a);
        if (op->pre_elu) {
            const size_t n = channels * length;
            k_decoder_elu<<<((int)n + 255) / 256, 256, 0, decoder->backend->stream>>>(
                current, const_cast<float *>(current), decoder->elu_alpha, (int)n);
            if (ce(cudaGetLastError(), e, ec)) return -1;
        }
        if (op->kind == CUDA_DECODER_CONV) {
            if (decoder_conv1d(decoder, op, current, destination, length, e, ec) != 0)
                return -1;
            current = destination;
            channels = (size_t)op->out_channels;
        } else if (op->kind == CUDA_DECODER_CONVTR) {
            if (decoder_convtr(decoder, op, current, destination, length, e, ec) != 0)
                return -1;
            current = destination;
            channels = (size_t)op->out_channels;
            length *= (size_t)op->stride;
        } else {
            set_error(e, ec, "resident decoder operation kind is invalid");
            return -1;
        }
    }
    return 0;
}

/* MYNAH_CUDA_ROW_MEM_DIET: a lean decoder has no private causal window,
 * `full` or ELU scratch for the gang, so a gang with a lean row may only run
 * if every op takes the fused path: each conv1d fused or tail-free, the
 * resblock's first conv fused (its unfused ELU would go to the scratch), and
 * every transposed conv in the fused GEMM form that writes the output and
 * the carried partial directly.  Mirrors the choices decoder_step_batch_impl
 * makes for the same width and lengths; false means "refuse the gang before
 * queuing anything" (the caller then decodes the rows one by one). */
static bool decoder_gang_lean_ok(const cuda_backend_state *backend,
                                 const mynah_backend_decoder *first,
                                 size_t batch, size_t encoder_frames) {
    if (!decoder_fuse_enabled()) return false;
    size_t length = encoder_frames;
    const size_t op_count = first->ops.size();
    for (size_t index = 0u; index < op_count;) {
        const cuda_decoder_op *op = &first->ops[index++];
        if (op->kind == CUDA_DECODER_RESBLOCK) {
            if (index + 1u >= op_count) return false;
            const cuda_decoder_op *rb1 = &first->ops[index++];
            const cuda_decoder_op *rb2 = &first->ops[index++];
            if (!decoder_conv_fused_path(rb1, batch, length) ||
                !(decoder_conv_fused_path(rb2, batch, length) || rb2->tail == 0u))
                return false;
            continue;
        }
        if (op->kind == CUDA_DECODER_CONV) {
            if (!(decoder_conv_fused_path(op, batch, length) || op->tail == 0u))
                return false;
        } else if (op->kind == CUDA_DECODER_CONVTR) {
            size_t output_len = 0u;
            if (!decoder_mul(length, (size_t)op->stride, &output_len) ||
                !decoder_convtr_gemm_path(backend, op, batch, length) ||
                op->tail > output_len)
                return false;
            length = output_len;
        } else {
            return false;
        }
    }
    return true;
}

/* One causal decoder topology, many independent request states.  The
 * per-request work/tail buffers remain owned by each decoder object, while
 * elementwise kernels and the conv1d GEMM use one launch/batched cuBLAS call
 * for the whole gang. */
static int decoder_step_batch_impl(
    cuda_backend_state *backend, mynah_backend_decoder *const *decoders,
    const float *const *inputs, size_t batch, size_t encoder_frames,
    float *const *outputs, char *e, size_t ec) {
    if (backend == nullptr || decoders == nullptr || inputs == nullptr ||
        outputs == nullptr || batch < 2u || batch > backend->batch_meta_cap ||
        encoder_frames == 0u || decoders[0] == nullptr ||
        decoders[0]->backend != backend ||
        encoder_frames > decoders[0]->max_encoder_frames)
        return 1;
    if (!backend->decoder_batch_enabled) return 1;
    mynah_backend_decoder *first = decoders[0];
    const size_t op_count = first->ops.size();
    if (op_count == 0u) return -1;
    if (!decoder_gang_rows_ok(backend, decoders, inputs, outputs, batch))
        return 1;
    bool any_lean = false;
    for (size_t i = 0; i < batch; ++i) any_lean = any_lean || decoders[i]->lean;
    if (any_lean && !decoder_gang_lean_ok(backend, first, batch, encoder_frames))
        return 1;
    if (cbe(cublasSetStream(backend->cublas, backend->stream), e, ec) != 0)
        return -1;

    float *current[CUDA_BATCH_META_CAP];
    float *other[CUDA_BATCH_META_CAP];
    float *scratch[CUDA_BATCH_META_CAP];
    float *destination[CUDA_BATCH_META_CAP];
    float *source[CUDA_BATCH_META_CAP];
    float *pending_resid[CUDA_BATCH_META_CAP];
    cuda_decoder_op *op_rows[CUDA_BATCH_META_CAP];
    for (size_t i = 0; i < batch; ++i)
        current[i] = const_cast<float *>(inputs[i]);

    /* MYNAH_CUDA_DECODER_FUSE: work the producer of `current` left to its
     * reader.  pending_bias_on: `current` is a bare GEMM accumulator and
     * pending_bias (possibly null: +0.0f) is still to be added.  has_resid:
     * the residual-block sum was not written; the reader takes
     * pending_resid (the block input x) + (source + bias), where `source`
     * holds the 1x1 output and `current` still names x (so the destination
     * choice below is unchanged).  Only set when the next op reads through a
     * fused kernel (decoder_lazy_reader). */
    int pending_bias_on = 0;
    const float *pending_bias = nullptr;
    bool has_resid = false;
    const bool fuse_bias = decoder_fuse_bias_enabled();

    size_t length = encoder_frames;
    size_t channels = first->dimension;
    for (size_t index = 0; index < op_count;) {
        cuda_decoder_op *op = &first->ops[index++];
        if (op->kind == CUDA_DECODER_RESBLOCK) {
            if (index + 1u >= op_count) {
                set_error(e, ec, "resident decoder residual topology is truncated");
                return -1;
            }
            if (has_resid || pending_bias_on) {
                set_error(e, ec, "resident decoder residual input is unresolved");
                return -1;
            }
            index += 2u;
            const cuda_decoder_op *rb1 = &first->ops[index - 2u];
            const cuda_decoder_op *rb2 = &first->ops[index - 1u];
            const cuda_decoder_op *next = index < op_count ? &first->ops[index]
                                                           : nullptr;
            for (size_t i = 0; i < batch; ++i) {
                op_rows[i] = &decoders[i]->ops[index - 2u];
                if (current[i] == decoder_gang_a(decoders[i]))
                    other[i] = decoder_gang_b(decoders[i]);
                else if (current[i] == decoder_gang_b(decoders[i]))
                    other[i] = decoder_gang_a(decoders[i]);
                else
                    other[i] = decoder_gang_a(decoders[i]);
                scratch[i] = decoder_gang_c(decoders[i]);
            }
            size_t elements = 0u;
            size_t hidden_elements = 0u;
            /* conv1's bias goes to conv2's column kernel, conv2's to the
             * residual add (wherever that runs). */
            const int defer1 =
                fuse_bias && decoder_conv_fused_path(rb2, batch, length);
            const int defer2 =
                fuse_bias && decoder_conv_fused_path(rb2, batch, length);
            if (!decoder_mul(channels, length, &elements) ||
                !decoder_mul((size_t)rb1->out_channels, length,
                             &hidden_elements) ||
                elements > (size_t)INT_MAX ||
                hidden_elements > (size_t)INT_MAX ||
                decoder_conv1d_batch(backend, decoders, op_rows, current, other,
                                     batch, length, 1, first->elu_alpha,
                                     scratch, nullptr, 0, nullptr, defer1, e,
                                     ec) != 0)
                return -1;
            for (size_t i = 0; i < batch; ++i) {
                op_rows[i] = &decoders[i]->ops[index - 1u];
                scratch[i] = other[i];
            }
            if (decoder_conv1d_batch(backend, decoders, op_rows, other, scratch,
                                     batch, length, 1, first->elu_alpha, other,
                                     nullptr, defer1, rb1->bias, defer2, e,
                                     ec) != 0)
                return -1;
            if (decoder_fuse_enabled() &&
                decoder_lazy_reader(backend, next, batch, length)) {
                /* The next op reads x + (y + b) on the fly; nothing written. */
                for (size_t i = 0; i < batch; ++i) {
                    pending_resid[i] = current[i];
                    source[i] = scratch[i];
                }
                has_resid = true;
                pending_bias_on = defer2;
                pending_bias = rb2->bias;
            } else if (decoder_residual_batch(backend, current, scratch, batch,
                                              channels, length, defer2,
                                              rb2->bias, e, ec) != 0) {
                return -1;
            }
            continue;
        }

        for (size_t i = 0; i < batch; ++i) {
            op_rows[i] = &decoders[i]->ops[index - 1u];
            if (!has_resid) source[i] = current[i];
        }
        float *const *resid = has_resid ? pending_resid : nullptr;
        /* pre_elu: applied in place on `current` by the op (or fused into
         * the kernel that reads it, MYNAH_CUDA_DECODER_FUSE). */
        const bool last = index == op_count;
        for (size_t i = 0; i < batch; ++i) {
            if (last)
                destination[i] = outputs[i];
            else if (current[i] == decoder_gang_a(decoders[i]))
                destination[i] = decoder_gang_b(decoders[i]);
            else
                destination[i] = decoder_gang_a(decoders[i]);
        }
        if (op->kind == CUDA_DECODER_CONV) {
            /* Defer this conv's bias when the next op reads through a fused
             * kernel (never into a residual block, never for the last op). */
            const cuda_decoder_op *next = last ? nullptr : &first->ops[index];
            const int defer = fuse_bias &&
                              !(decoder_one_gemm_enabled() && batch > 1u) &&
                              decoder_lazy_reader(backend, next, batch, length);
            if (decoder_conv1d_batch(backend, decoders, op_rows, source,
                                     destination, batch, length, op->pre_elu,
                                     first->elu_alpha, current, resid,
                                     pending_bias_on, pending_bias, defer, e,
                                     ec) != 0)
                return -1;
            pending_bias_on = defer;
            pending_bias = op->bias;
            channels = (size_t)op->out_channels;
        } else if (op->kind == CUDA_DECODER_CONVTR) {
            if (decoder_convtr_batch(backend, op_rows, source, destination,
                                     batch, length, op->pre_elu,
                                     first->elu_alpha, resid, pending_bias_on,
                                     pending_bias, e, ec) != 0)
                return -1;
            pending_bias_on = 0;
            pending_bias = nullptr;
            channels = (size_t)op->out_channels;
            if (!decoder_mul(length, (size_t)op->stride, &length))
                return -1;
        } else {
            set_error(e, ec, "resident decoder operation kind is invalid");
            return -1;
        }
        has_resid = false;
        for (size_t i = 0; i < batch; ++i) current[i] = destination[i];
    }
    if (has_resid || pending_bias_on) {
        set_error(e, ec, "resident decoder output is unresolved");
        return -1;
    }
    return 0;
}

extern "C" int mynah_cuda_decoder_open(
    void *opaque, const mynah_backend_decoder_desc *desc,
    size_t max_encoder_frames, mynah_backend_decoder **out,
    char *e, size_t ec) {
    if (out != nullptr) *out = nullptr;
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || desc == nullptr || out == nullptr ||
        max_encoder_frames == 0u) {
        set_error(e, ec, "invalid resident decoder open");
        return -1;
    }
    auto *decoder = new (std::nothrow) mynah_backend_decoder();
    if (decoder == nullptr) {
        set_error(e, ec, "out of memory creating resident decoder");
        return -1;
    }
    decoder->backend = backend;
    decoder->channels = desc->channels;
    decoder->dimension = desc->dimension;
    decoder->n_filters = desc->n_filters;
    decoder->max_encoder_frames = max_encoder_frames;
    decoder->elu_alpha = desc->elu_alpha;
    /* MYNAH_CUDA_ROW_MEM_DIET: decided before the build, which then skips
     * the per-op window and `full` (the solo path takes them from the shared
     * set; decoder_alloc_lean gives them back if the topology cannot be
     * lean). Off: false, the old allocations in the old order. */
    decoder->lean = decoder_lean_planned();
    try {
        if (decoder_build(decoder, desc, e, ec) != 0 ||
            (decoder->lean ? decoder_alloc_lean(decoder, e, ec)
                           : decoder_alloc_workspace(decoder, e, ec)) != 0 ||
            decoder_reserve_one_gemm(backend, decoder, e, ec) != 0) {
            decoder_destroy(decoder);
            return -1;
        }
    } catch (const std::bad_alloc &) {
        decoder_destroy(decoder);
        set_error(e, ec, "out of memory building resident decoder topology");
        return -1;
    }
    if (cuda_seanet_bf16_enabled()) {
        /* One line per process, so an A/B log shows which arm it is. */
        static std::atomic<bool> announced{false};
        if (!announced.exchange(true))
            std::fprintf(stderr,
                         "mynah-tts: CUDA SEANet decoder convolutions in bf16 "
                         "(MYNAH_CUDA_SEANET_BF16=1; fp32 accumulate, states "
                         "and audio)\n");
    }
    /* A planned lean decoder that fell back for its topology (the tiny
     * self-test decoder) does not take the announcement. */
    if (cuda_row_mem_diet_enabled() &&
        (decoder->lean || !decoder_lean_planned())) {
        static std::atomic<bool> announced_diet{false};
        if (!announced_diet.exchange(true)) {
            if (decoder->lean)
                std::fprintf(stderr,
                             "mynah-tts: MYNAH_CUDA_ROW_MEM_DIET=1: SEANet "
                             "decoder device memory per request %zu KiB (was "
                             "%zu KiB); single-request scratch is one shared "
                             "set of %zu KiB\n",
                             decoder_owned_bytes(decoder) / 1024u,
                             decoder_legacy_bytes(decoder) / 1024u,
                             ((3u * backend->solo_work_cap +
                               backend->solo_columns_cap +
                               backend->solo_window_cap +
                               backend->solo_full_cap) * sizeof(float)) / 1024u);
            else
                std::fprintf(stderr,
                             "mynah-tts: warning: MYNAH_CUDA_ROW_MEM_DIET=1 "
                             "leaves the SEANet decoder buffers per request "
                             "(needs MYNAH_CUDA_DECODER_FUSE=1, the transposed-"
                             "conv GEMM and no MYNAH_CUDA_DECODER_ONEGEMM)\n");
        }
    }
    if (decoder_fuse_enabled()) {
        static std::atomic<bool> announced_fuse{false};
        if (!announced_fuse.exchange(true))
            std::fprintf(stderr,
                         "mynah-tts: CUDA decoder ELU, causal window, residual "
                         "and transposed-conv fold fused into the "
                         "im2col/gather/overlap kernels, conv bias %s "
                         "(MYNAH_CUDA_DECODER_FUSE=1)\n",
                         decoder_fuse_bias_enabled()
                             ? "deferred to the reader (beta = 0)"
                             : "kept in the GEMM (MYNAH_CUDA_DECODER_FUSE_BIAS=0)");
    }
    if (cuda_env_enabled("MYNAH_CUDA_DECODER_TABLE_PATCH", false)) {
        static std::atomic<bool> announced_patch{false};
        if (!announced_patch.exchange(true))
            std::fprintf(stderr,
                         cuda_decoder_table_patch_enabled()
                             ? "mynah-tts: CUDA decoder batch graph gang changes "
                               "patch per-decoder pointer-table columns instead "
                               "of re-recording (MYNAH_CUDA_DECODER_TABLE_PATCH=1)\n"
                             : "mynah-tts: warning: MYNAH_CUDA_DECODER_TABLE_PATCH=1 "
                               "needs MYNAH_CUDA_DECODER_GRAPH_REUSE; ignored\n");
    }
    if (cuda_decoder_validate_once_enabled()) {
        static std::atomic<bool> announced_once{false};
        if (!announced_once.exchange(true))
            std::fprintf(stderr,
                         "mynah-tts: CUDA decoder gang topology checked once per "
                         "decoder, then by compatibility class "
                         "(MYNAH_CUDA_DECODER_VALIDATE_ONCE=1)\n");
    }
    *out = decoder;
    return 0;
}

extern "C" void mynah_cuda_decoder_close(void *opaque,
                                           mynah_backend_decoder *decoder) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend != nullptr) {
        (void)cudaStreamSynchronize(backend->stream);
        destroy_decoder_batch_graphs_for(backend, decoder);
    }
    decoder_destroy(decoder);
}

/* Device bytes this decoder owns now and would own without
 * MYNAH_CUDA_ROW_MEM_DIET (equal when the diet is off or did not apply). */
extern "C" int mynah_cuda_decoder_device_bytes(void *opaque,
                                               const mynah_backend_decoder *decoder,
                                               size_t *owned, size_t *legacy) {
    (void)opaque;
    if (decoder == nullptr || owned == nullptr || legacy == nullptr) return -1;
    *owned = decoder_owned_bytes(decoder);
    *legacy = decoder_legacy_bytes(decoder);
    return 0;
}

extern "C" int mynah_cuda_decoder_reset(void *opaque,
                                          mynah_backend_decoder *decoder,
                                          char *e, size_t ec) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || decoder == nullptr || decoder->backend != backend) {
        set_error(e, ec, "invalid resident decoder reset");
        return -1;
    }
    for (auto &op : decoder->ops) {
        if (op.previous != nullptr &&
            ce(cudaMemsetAsync(op.previous, 0,
                               (size_t)op.in_channels * op.tail * sizeof(float),
                               backend->stream), e, ec)) return -1;
        if (op.partial != nullptr &&
            ce(cudaMemsetAsync(op.partial, 0,
                               (size_t)op.out_channels * op.tail * sizeof(float),
                               backend->stream), e, ec)) return -1;
    }
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_decoder_step(void *opaque,
                                         mynah_backend_decoder *decoder,
                                         const float *dev_input,
                                         size_t encoder_frames,
                                         float *dev_output,
                                         char *e, size_t ec) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || decoder == nullptr || decoder->backend != backend) {
        set_error(e, ec, "invalid resident decoder step");
        if (backend != nullptr)
            backend->decoder_failures.fetch_add(1ull, std::memory_order_relaxed);
        return -1;
    }
    /* During graph capture this call only records work; the graph launch is
     * the logical decoder step. Direct execution counts here, while the
     * engine calls decoder_note_step after a successful capture/replay. */
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool capturing =
        cudaStreamIsCapturing(backend->stream, &capture_status) == cudaSuccess &&
        capture_status != cudaStreamCaptureStatusNone;
    if (!capturing)
        backend->decoder_steps.fetch_add(1ull, std::memory_order_relaxed);
    const int result = decoder_step_impl(decoder, dev_input, encoder_frames,
                                         dev_output, e, ec);
    if (result != 0) {
        backend->decoder_failures.fetch_add(1ull, std::memory_order_relaxed);
        backend->resident_fallbacks.fetch_add(1ull, std::memory_order_relaxed);
    }
    return result;
}

extern "C" int mynah_cuda_decoder_step_batch(
    void *opaque, mynah_backend_decoder *const *decoders,
    const float *const *dev_inputs, size_t batch, size_t encoder_frames,
    float *const *dev_outputs, char *e, size_t ec) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || decoders == nullptr || dev_inputs == nullptr ||
        dev_outputs == nullptr || batch < 2u ||
        batch > backend->batch_meta_cap || encoder_frames == 0u)
        return 1;
    /* MYNAH_CUDA_ROW_MEM_DIET: refuse a gang a lean row cannot run in before
     * any graph bookkeeping (decoder_step_batch_impl checks it again). */
    for (size_t i = 0; i < batch; ++i) {
        if (decoders[i] != nullptr && decoders[i]->lean) {
            if (decoders[0] == nullptr ||
                !decoder_gang_lean_ok(backend, decoders[0], batch,
                                      encoder_frames))
                return 1;
            break;
        }
    }
    auto eager = [&]() -> int {
        const int result = decoder_step_batch_impl(
            backend, decoders, dev_inputs, batch, encoder_frames, dev_outputs,
            e, ec);
        if (result == 0) {
            backend->decoder_steps.fetch_add((unsigned long long)batch,
                                             std::memory_order_relaxed);
        } else if (result < 0) {
            backend->decoder_failures.fetch_add((unsigned long long)batch,
                                                std::memory_order_relaxed);
            backend->resident_fallbacks.fetch_add((unsigned long long)batch,
                                                   std::memory_order_relaxed);
        }
        return result;
    };

    /* The graph is an optimization of the already-correct arithmetic batch.
     * It is deliberately separate from the single-request graph: a changing
     * scheduler gang must never replay a graph containing another request's
     * causal rings. */
    if (!backend->graphs_enabled || !cuda_decoder_graphs_enabled()) {
        backend->decoder_graph_fallbacks.fetch_add(1ull,
                                                   std::memory_order_relaxed);
        return eager();
    }

    cuda_decoder_batch_graph_entry *entry = find_decoder_batch_graph(
        backend, decoders, dev_inputs, dev_outputs, batch, encoder_frames);
    if (entry != nullptr) {
        if (ce(cudaGraphLaunch(entry->exec, backend->stream), e, ec) != 0) {
            decoder_batch_graph_remove(backend, entry);
            backend->decoder_graph_fallbacks.fetch_add(
                1ull, std::memory_order_relaxed);
            backend->decoder_failures.fetch_add((unsigned long long)batch,
                                                std::memory_order_relaxed);
            backend->resident_fallbacks.fetch_add((unsigned long long)batch,
                                                   std::memory_order_relaxed);
            return -1;
        }
        if (entry->done != nullptr)
            (void)cudaEventRecord(entry->done, backend->stream);
        backend->decoder_graph_replays.fetch_add(1ull,
                                                 std::memory_order_relaxed);
        backend->decoder_steps.fetch_add((unsigned long long)batch,
                                         std::memory_order_relaxed);
        return 0;
    }

    entry = cuda_decoder_graph_reuse_enabled()
                ? find_decoder_batch_graph_shape(backend, decoders[0], batch,
                                                 encoder_frames)
                : nullptr;
    if (entry != nullptr) {
        /* Same width, different gang: re-record only to rewrite the pinned
         * pointer tables that the instantiated graph's memcpy nodes read at
         * launch, then drop the recording.  No instantiate on this path. */
        if (ce(cudaEventSynchronize(entry->done), e, ec) != 0) return -1;
        /* MYNAH_CUDA_DECODER_TABLE_PATCH: the same tables without the
         * re-record, from columns a real recording wrote.  The checks are the
         * ones decoder_step_batch_impl would refuse the gang with; anything
         * not ready takes the re-record below. */
        if (decoder_table_patch_ready(entry, decoders, dev_inputs,
                                      dev_outputs) &&
            backend->decoder_batch_enabled &&
            decoders[0]->backend == backend &&
            encoder_frames <= decoders[0]->max_encoder_frames &&
            decoder_gang_rows_ok(backend, decoders, dev_inputs, dev_outputs,
                                 batch)) {
            decoder_table_patch(entry, decoders, dev_inputs, dev_outputs);
            try {
                entry->decoders.assign(decoders, decoders + batch);
                entry->inputs.assign(dev_inputs, dev_inputs + batch);
                entry->outputs.assign(dev_outputs, dev_outputs + batch);
            } catch (const std::bad_alloc &) {
                entry->decoders.clear();
                entry->inputs.clear();
                entry->outputs.clear();
            }
            if (ce(cudaGraphLaunch(entry->exec, backend->stream), e, ec) != 0) {
                decoder_batch_graph_remove(backend, entry);
                backend->decoder_graph_fallbacks.fetch_add(
                    1ull, std::memory_order_relaxed);
                backend->decoder_failures.fetch_add((unsigned long long)batch,
                                                    std::memory_order_relaxed);
                backend->resident_fallbacks.fetch_add((unsigned long long)batch,
                                                       std::memory_order_relaxed);
                return -1;
            }
            (void)cudaEventRecord(entry->done, backend->stream);
            backend->decoder_graph_table_patches.fetch_add(
                1ull, std::memory_order_relaxed);
            backend->decoder_graph_replays.fetch_add(1ull,
                                                     std::memory_order_relaxed);
            backend->decoder_steps.fetch_add((unsigned long long)batch,
                                             std::memory_order_relaxed);
            return 0;
        }
        cudaError_t begin = cudaStreamBeginCapture(backend->stream,
                                                    cudaStreamCaptureModeRelaxed);
        if (begin != cudaSuccess) {
            ce(begin, e, ec);
            decoder_batch_graph_remove(backend, entry);
            backend->decoder_graph_fallbacks.fetch_add(1ull,
                                                       std::memory_order_relaxed);
            return eager();
        }
        backend->active_decoder_batch_graph = entry;
        backend->active_decoder_upload_slot = 0u;
        const int build = decoder_step_batch_impl(
            backend, decoders, dev_inputs, batch, encoder_frames, dev_outputs,
            e, ec);
        const size_t used = backend->active_decoder_upload_slot;
        backend->active_decoder_batch_graph = nullptr;
        for (size_t i = 0u; i < 4u; ++i) backend->active_decoder_tables[i] = nullptr;
        cudaGraph_t scratch = nullptr;
        const cudaError_t end = cudaStreamEndCapture(backend->stream, &scratch);
        if (scratch != nullptr) cudaGraphDestroy(scratch);
        if (build != 0 || end != cudaSuccess || used != entry->used_slots) {
            decoder_batch_graph_remove(backend, entry);
            backend->decoder_graph_fallbacks.fetch_add(1ull,
                                                       std::memory_order_relaxed);
            if (build < 0) {
                backend->decoder_failures.fetch_add((unsigned long long)batch,
                                                    std::memory_order_relaxed);
                backend->resident_fallbacks.fetch_add((unsigned long long)batch,
                                                       std::memory_order_relaxed);
                return -1;
            }
            if (end != cudaSuccess) ce(end, e, ec);
            return eager();
        }
        decoder_table_harvest(entry, decoders, dev_inputs, dev_outputs);
        backend->decoder_graph_rerecords.fetch_add(1ull,
                                                   std::memory_order_relaxed);
        try {
            entry->decoders.assign(decoders, decoders + batch);
            entry->inputs.assign(dev_inputs, dev_inputs + batch);
            entry->outputs.assign(dev_outputs, dev_outputs + batch);
        } catch (const std::bad_alloc &) {
            entry->decoders.clear();
            entry->inputs.clear();
            entry->outputs.clear();
        }
        if (ce(cudaGraphLaunch(entry->exec, backend->stream), e, ec) != 0) {
            decoder_batch_graph_remove(backend, entry);
            backend->decoder_graph_fallbacks.fetch_add(
                1ull, std::memory_order_relaxed);
            backend->decoder_failures.fetch_add((unsigned long long)batch,
                                                std::memory_order_relaxed);
            backend->resident_fallbacks.fetch_add((unsigned long long)batch,
                                                   std::memory_order_relaxed);
            return -1;
        }
        (void)cudaEventRecord(entry->done, backend->stream);
        backend->decoder_graph_replays.fetch_add(1ull,
                                                 std::memory_order_relaxed);
        backend->decoder_steps.fetch_add((unsigned long long)batch,
                                         std::memory_order_relaxed);
        return 0;
    }

    entry = decoder_batch_graph_create(backend, decoders, dev_inputs,
                                       dev_outputs, batch, encoder_frames, e,
                                       ec);
    if (entry == nullptr) {
        backend->decoder_graph_fallbacks.fetch_add(1ull,
                                                   std::memory_order_relaxed);
        return eager();
    }

    /* cuBLAS may request its workspace on the first batched convolution GEMM.
     * Reserve/configure it before capture so a lazy allocator cannot turn a
     * valid decoder graph into a capture-time failure.  The same workspace is
     * shared with the other resident graph families on this backend. */
    if (backend->cublas_workspace == nullptr) {
        backend->cublas_workspace_cap = 8u * 1024u * 1024u;
        if (ce(cudaMalloc(&backend->cublas_workspace,
                          backend->cublas_workspace_cap), e, ec) != 0) {
            decoder_batch_graph_remove(backend, entry);
            backend->decoder_graph_fallbacks.fetch_add(
                1ull, std::memory_order_relaxed);
            return eager();
        }
    }
    if (cbe(cublasSetWorkspace(backend->cublas, backend->cublas_workspace,
                               backend->cublas_workspace_cap), e, ec) != 0 ||
        cbe(cublasSetStream(backend->cublas, backend->stream), e, ec) != 0) {
        decoder_batch_graph_remove(backend, entry);
        backend->decoder_graph_fallbacks.fetch_add(
            1ull, std::memory_order_relaxed);
        return eager();
    }

    cudaError_t begin = cudaStreamBeginCapture(backend->stream,
                                                cudaStreamCaptureModeRelaxed);
    if (begin != cudaSuccess) {
        ce(begin, e, ec);
        decoder_batch_graph_remove(backend, entry);
        backend->decoder_graph_fallbacks.fetch_add(1ull,
                                                   std::memory_order_relaxed);
        return eager();
    }
    backend->active_decoder_batch_graph = entry;
    backend->active_decoder_upload_slot = 0u;
    for (size_t i = 0u; i < 4u; ++i)
        backend->active_decoder_tables[i] =
            i == 0u ? backend->dev_decoder_ptr0
            : i == 1u ? backend->dev_decoder_ptr1
            : i == 2u ? backend->dev_decoder_ptr2
                      : backend->dev_decoder_ptr3;
    entry->layout_recording = true;
    const int build = decoder_step_batch_impl(
        backend, decoders, dev_inputs, batch, encoder_frames, dev_outputs, e,
        ec);
    entry->layout_recording = false;
    entry->used_slots = backend->active_decoder_upload_slot;
    backend->active_decoder_batch_graph = nullptr;
    for (size_t i = 0u; i < 4u; ++i) backend->active_decoder_tables[i] = nullptr;

    cudaGraph_t graph = nullptr;
    const cudaError_t end = cudaStreamEndCapture(backend->stream, &graph);
    if (build != 0 || end != cudaSuccess || graph == nullptr) {
        if (graph != nullptr) cudaGraphDestroy(graph);
        decoder_batch_graph_remove(backend, entry);
        backend->decoder_graph_fallbacks.fetch_add(1ull,
                                                   std::memory_order_relaxed);
        /* A validation failure is the normal optional-path result.  A launch
         * failure must stay fatal to the CUDA batch; the caller's state was
         * not allowed to advance through an unlaunched graph. */
        if (build < 0) {
            backend->decoder_failures.fetch_add((unsigned long long)batch,
                                                std::memory_order_relaxed);
            backend->resident_fallbacks.fetch_add((unsigned long long)batch,
                                                   std::memory_order_relaxed);
            return -1;
        }
        if (end != cudaSuccess) ce(end, e, ec);
        return 1;
    }

    cudaGraphExec_t exec = nullptr;
    const cudaError_t instantiate = cudaGraphInstantiate(&exec, graph, 0);
    if (instantiate != cudaSuccess || exec == nullptr) {
        if (exec != nullptr) cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        entry->graph = nullptr;
        entry->exec = nullptr;
        decoder_batch_graph_remove(backend, entry);
        if (instantiate != cudaSuccess) ce(instantiate, e, ec);
        backend->decoder_graph_fallbacks.fetch_add(1ull,
                                                   std::memory_order_relaxed);
        /* Capture does not execute the recorded decoder.  Instantiation is
         * therefore safe to fall back from to one ordinary arithmetic batch. */
        return eager();
    }
    entry->graph = graph;
    entry->exec = exec;
    entry->valid = true;
    if (cuda_decoder_graph_reuse_enabled() &&
        cudaEventCreateWithFlags(&entry->done, cudaEventDisableTiming) != cudaSuccess)
        entry->done = nullptr;
    if (entry->slot_channels != nullptr) {
        entry->layout_key = decoder_table_layout_key(entry);
        decoder_table_harvest(entry, decoders, dev_inputs, dev_outputs);
    }
    backend->decoder_graph_captures.fetch_add(1ull,
                                             std::memory_order_relaxed);
    if (ce(cudaGraphLaunch(entry->exec, backend->stream), e, ec) != 0) {
        decoder_batch_graph_remove(backend, entry);
        backend->decoder_graph_fallbacks.fetch_add(1ull,
                                                   std::memory_order_relaxed);
        backend->decoder_failures.fetch_add((unsigned long long)batch,
                                            std::memory_order_relaxed);
        backend->resident_fallbacks.fetch_add((unsigned long long)batch,
                                               std::memory_order_relaxed);
        return -1;
    }
    if (entry->done != nullptr) (void)cudaEventRecord(entry->done, backend->stream);
    backend->decoder_steps.fetch_add((unsigned long long)batch,
                                     std::memory_order_relaxed);
    return 0;
}

extern "C" int mynah_cuda_decoder_note_step(
    void *opaque, mynah_backend_decoder *decoder) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || decoder == nullptr || decoder->backend != backend)
        return -1;
    backend->decoder_steps.fetch_add(1ull, std::memory_order_relaxed);
    return 0;
}

static void note_width(cuda_backend_state *st, int stage, size_t width) {
    int bucket = 0;
    size_t limit = 1u;
    while (bucket < 7 && width > limit) {
        ++bucket;
        limit <<= 1u;
    }
    st->width_hist[stage][bucket].fetch_add(1ull, std::memory_order_relaxed);
}

extern "C" int mynah_cuda_decoder_note_batch(void *opaque, size_t items,
                                               size_t frames) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || items == 0u || frames == 0u) return -1;
    if (!backend->decoder_batch_enabled) return 0;
    note_width(backend, 2, items);
    backend->decoder_batch_calls.fetch_add(1ull, std::memory_order_relaxed);
    backend->decoder_batch_items.fetch_add((unsigned long long)items,
                                           std::memory_order_relaxed);
    unsigned long long observed =
        backend->decoder_batch_max_width.load(std::memory_order_relaxed);
    const unsigned long long width = (unsigned long long)items;
    while (observed < width &&
           !backend->decoder_batch_max_width.compare_exchange_weak(
               observed, width, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    backend->decoder_batch_frames.fetch_add((unsigned long long)frames,
                                            std::memory_order_relaxed);
    return 0;
}

extern "C" int mynah_cuda_note_backbone_batch(void *opaque, size_t items) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || items == 0u) return -1;
    note_width(backend, 0, items);
    backend->backbone_batch_calls.fetch_add(1ull, std::memory_order_relaxed);
    backend->backbone_batch_items.fetch_add((unsigned long long)items,
                                            std::memory_order_relaxed);
    unsigned long long observed =
        backend->backbone_batch_max_width.load(std::memory_order_relaxed);
    const unsigned long long width = (unsigned long long)items;
    while (observed < width &&
           !backend->backbone_batch_max_width.compare_exchange_weak(
               observed, width, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    return 0;
}

extern "C" int mynah_cuda_note_codec_transformer_batch(void *opaque,
                                                         size_t items,
                                                         size_t width) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || items == 0u || width == 0u) return -1;
    note_width(backend, 1, items);
    backend->codec_transformer_batch_calls.fetch_add(
        1ull, std::memory_order_relaxed);
    backend->codec_transformer_batch_items.fetch_add(
        (unsigned long long)items, std::memory_order_relaxed);
    unsigned long long observed =
        backend->codec_transformer_batch_max_width.load(
            std::memory_order_relaxed);
    const unsigned long long batch_width = (unsigned long long)items;
    while (observed < batch_width &&
           !backend->codec_transformer_batch_max_width.compare_exchange_weak(
               observed, batch_width, std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
    return 0;
}

extern "C" void mynah_cuda_note_codec_gang(void *opaque, int stage,
                                           size_t rows) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || stage < 0 || stage > 1 || rows == 0u) return;
    int bucket = 0;
    size_t limit = 1u;
    while (bucket < 7 && rows > limit) {
        ++bucket;
        limit <<= 1u;
    }
    backend->codec_gang_hist[stage][bucket].fetch_add(
        1ull, std::memory_order_relaxed);
    backend->codec_gang_calls[stage].fetch_add(1ull, std::memory_order_relaxed);
    backend->codec_gang_rows[stage].fetch_add((unsigned long long)rows,
                                              std::memory_order_relaxed);
}

extern "C" void mynah_cuda_note_codec_upsample(void *opaque, int fallback) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr) return;
    if (fallback != 0) {
        backend->codec_upsample_fallbacks.fetch_add(
            1ull, std::memory_order_relaxed);
    } else {
        backend->codec_upsample_steps.fetch_add(
            1ull, std::memory_order_relaxed);
    }
}

/* Model-free end-to-end check for the resident causal decoder.  The generic
 * conv tests above cannot catch a stale causal ring or a transposed-conv tail
 * folded into the wrong frame, so this deliberately runs two one-frame calls
 * against the scalar SEANet state with the same tiny topology and weights. */
static int cuda_decoder_self_test(void *opaque, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr) {
        set_error(e, ec, "CUDA decoder self-test has no backend");
        return -1;
    }
    const size_t ratios[1] = {2u};
    float first_weight[24];
    float first_bias[4];
    float convtr_weight[32];
    float convtr_bias[2];
    float rb1_weight[6];
    float rb1_bias[1];
    float rb2_weight[2];
    float rb2_bias[2];
    float last_weight[6];
    float last_bias[1];
    for (size_t i = 0; i < 24u; ++i)
        first_weight[i] = ((int)(i % 9u) - 4) * 0.03125f;
    for (size_t i = 0; i < 4u; ++i) first_bias[i] = ((int)i - 1) * 0.05f;
    for (size_t i = 0; i < 32u; ++i)
        convtr_weight[i] = ((int)(i % 11u) - 5) * 0.021f;
    convtr_bias[0] = 0.07f; convtr_bias[1] = -0.04f;
    for (size_t i = 0; i < 6u; ++i)
        rb1_weight[i] = ((int)(i % 5u) - 2) * 0.027f;
    rb1_bias[0] = 0.03f;
    rb2_weight[0] = 0.11f; rb2_weight[1] = -0.08f;
    rb2_bias[0] = 0.02f; rb2_bias[1] = -0.01f;
    for (size_t i = 0; i < 6u; ++i)
        last_weight[i] = ((int)(i % 7u) - 3) * 0.019f;
    last_bias[0] = -0.02f;

    mynah_conv_weights first = {first_weight, first_bias};
    mynah_conv_weights convtr = {convtr_weight, convtr_bias};
    mynah_seanet_resblock_weights block;
    block.conv1.weight = rb1_weight;
    block.conv1.bias = rb1_bias;
    block.conv2.weight = rb2_weight;
    block.conv2.bias = rb2_bias;
    mynah_conv_weights last = {last_weight, last_bias};
    mynah_seanet_decoder_weights cpu_weights;
    cpu_weights.first = first;
    cpu_weights.convtr = &convtr;
    cpu_weights.blocks = &block;
    cpu_weights.last = last;

    mynah_backend_decoder_desc desc;
    std::memset(&desc, 0, sizeof(desc));
    desc.channels = 1u;
    desc.dimension = 2u;
    desc.n_filters = 2u;
    desc.n_residual_layers = 1u;
    desc.ratios = ratios;
    desc.n_ratios = 1u;
    desc.kernel_size = 3u;
    desc.residual_kernel_size = 3u;
    desc.last_kernel_size = 3u;
    desc.dilation_base = 2u;
    desc.compress = 2u;
    desc.elu_alpha = 1.0f;
    desc.first = first;
    desc.convtr = &convtr;
    desc.blocks = &block;
    desc.last = last;

    mynah_seanet_config cpu_config;
    std::memset(&cpu_config, 0, sizeof(cpu_config));
    cpu_config.channels = 1u;
    cpu_config.dimension = 2u;
    cpu_config.n_filters = 2u;
    cpu_config.n_residual_layers = 1u;
    cpu_config.ratios = ratios;
    cpu_config.n_ratios = 1u;
    cpu_config.kernel_size = 3u;
    cpu_config.residual_kernel_size = 3u;
    cpu_config.last_kernel_size = 3u;
    cpu_config.dilation_base = 2u;
    cpu_config.compress = 2u;
    cpu_config.elu_alpha = 1.0f;

    const float inputs[2][2] = {{0.20f, -0.15f}, {-0.31f, 0.27f}};
    float cpu_output[2];
    float gpu_output[2];
    char local[256];
    local[0] = '\0';
    mynah_backend_decoder *decoder = nullptr;
    mynah_seanet_state *cpu_state = nullptr;
    float *dev_input = nullptr;
    float *dev_output = nullptr;
    int result = -1;

    do {
        cpu_state = mynah_seanet_state_create(&cpu_config, nullptr, 2u, local,
                                              sizeof(local));
        if (cpu_state == nullptr) break;
        if (mynah_cuda_decoder_open(st, &desc, 1u, &decoder, local,
                                     sizeof(local)) != 0 || decoder == nullptr)
            break;
        if (mynah_cuda_decoder_reset(st, decoder, local, sizeof(local)) != 0 ||
            ce(cudaMalloc(&dev_input, 2u * sizeof(float)), local, sizeof(local)) ||
            ce(cudaMalloc(&dev_output, 2u * sizeof(float)), local, sizeof(local)))
            break;
        for (size_t step = 0; step < 2u; ++step) {
            if (ce(cudaMemcpyAsync(dev_input, inputs[step], 2u * sizeof(float),
                                   cudaMemcpyHostToDevice, st->stream),
                   local, sizeof(local)) != 0 ||
                mynah_cuda_decoder_step(st, decoder, dev_input, 1u, dev_output,
                                        local, sizeof(local)) != 0 ||
                mynah_cuda_sync(st, local, sizeof(local)) != 0 ||
                ce(cudaMemcpy(gpu_output, dev_output, 2u * sizeof(float),
                              cudaMemcpyDeviceToHost), local, sizeof(local)) != 0 ||
                mynah_seanet_decode(cpu_state, &cpu_weights, inputs[step], 1u,
                                    cpu_output) != 0) {
                break;
            }
            int mismatch = 0;
            for (size_t i = 0; i < 2u; ++i) {
                if (fabsf(cpu_output[i] - gpu_output[i]) > 3.0e-3f) {
                    mismatch = 1;
                    break;
                }
            }
            if (mismatch) {
                std::snprintf(local, sizeof(local),
                              "CUDA resident decoder mismatch at step %zu",
                              step);
                break;
            }
            if (step == 1u) result = 0;
        }
    } while (false);

    if (result != 0 && e != nullptr && ec > 0u)
        std::snprintf(e, ec, "%s", local[0] != '\0'
                          ? local : "CUDA resident decoder self-test failed");
    if (dev_input != nullptr) cudaFree(dev_input);
    if (dev_output != nullptr) cudaFree(dev_output);
    if (decoder != nullptr) mynah_cuda_decoder_close(st, decoder);
    if (cpu_state != nullptr) mynah_seanet_state_destroy(cpu_state);

    /* Exercise the same topology with two independent causal states.  This
     * is deliberately separate from the single-request check above: a batch
     * kernel can be correct for request zero while indexing request one with
     * the wrong window/tail stride.  The environment switch is a runtime
     * feature gate, so a deliberately disabled decoder batch remains a valid
     * self-test configuration. */
    if (result == 0 && st->decoder_batch_enabled) {
        mynah_backend_decoder *batch_decoders[2] = {nullptr, nullptr};
        mynah_seanet_state *batch_cpu[2] = {nullptr, nullptr};
        float *batch_inputs_dev[2] = {nullptr, nullptr};
        float *batch_outputs_dev[2] = {nullptr, nullptr};
        const float batch_inputs[2][2][2] = {
            {{0.17f, -0.22f}, {-0.28f, 0.31f}},
            {{-0.09f, 0.14f}, {0.36f, -0.19f}}
        };
        float batch_cpu_output[2][2];
        float batch_gpu_output[2][2];
        local[0] = '\0';
        int batch_ok = 1;
        for (size_t i = 0; i < 2u && batch_ok; ++i) {
            batch_cpu[i] = mynah_seanet_state_create(
                &cpu_config, nullptr, 2u, local, sizeof(local));
            batch_ok = batch_cpu[i] != nullptr &&
                       mynah_cuda_decoder_open(
                           st, &desc, 1u, &batch_decoders[i], local,
                           sizeof(local)) == 0 && batch_decoders[i] != nullptr;
            if (batch_ok)
                batch_ok = mynah_cuda_decoder_reset(
                               st, batch_decoders[i], local, sizeof(local)) == 0 &&
                           ce(cudaMalloc((void **)&batch_inputs_dev[i],
                                         2u * sizeof(float)),
                              local, sizeof(local)) == 0 &&
                           ce(cudaMalloc((void **)&batch_outputs_dev[i],
                                         2u * sizeof(float)),
                              local, sizeof(local)) == 0;
        }
        mynah_backend_decoder *decoder_rows[2] = {
            batch_decoders[0], batch_decoders[1]};
        const float *input_rows[2] = {
            batch_inputs_dev[0], batch_inputs_dev[1]};
        float *output_rows[2] = {
            batch_outputs_dev[0], batch_outputs_dev[1]};
        for (size_t step = 0; step < 2u && batch_ok; ++step) {
            for (size_t request = 0; request < 2u; ++request) {
                if (ce(cudaMemcpyAsync(
                           batch_inputs_dev[request],
                           batch_inputs[request][step], 2u * sizeof(float),
                           cudaMemcpyHostToDevice, st->stream),
                       local, sizeof(local)) != 0) {
                    batch_ok = 0;
                    break;
                }
            }
            if (!batch_ok ||
                mynah_cuda_decoder_step_batch(
                    st, decoder_rows, input_rows, 2u, 1u, output_rows, local,
                    sizeof(local)) != 0 ||
                mynah_cuda_sync(st, local, sizeof(local)) != 0) {
                batch_ok = 0;
                break;
            }
            for (size_t request = 0; request < 2u; ++request) {
                if (ce(cudaMemcpy(batch_gpu_output[request],
                                  batch_outputs_dev[request],
                                  2u * sizeof(float), cudaMemcpyDeviceToHost),
                       local, sizeof(local)) != 0 ||
                    mynah_seanet_decode(batch_cpu[request], &cpu_weights,
                                        batch_inputs[request][step], 1u,
                                        batch_cpu_output[request]) != 0) {
                    batch_ok = 0;
                    break;
                }
                for (size_t d = 0; d < 2u; ++d) {
                    if (fabsf(batch_cpu_output[request][d] -
                              batch_gpu_output[request][d]) > 3.0e-3f) {
                        std::snprintf(
                            local, sizeof(local),
                            "CUDA batched resident decoder mismatch at step %zu request %zu",
                            step, request);
                        batch_ok = 0;
                        break;
                    }
                }
                if (!batch_ok) break;
            }
        }
        if (!batch_ok) {
            result = -1;
            if (e != nullptr && ec > 0u)
                std::snprintf(e, ec, "%s", local[0] != '\0'
                                  ? local
                                  : "CUDA batched resident decoder self-test failed");
        }
        for (size_t i = 0; i < 2u; ++i) {
            if (batch_inputs_dev[i] != nullptr) cudaFree(batch_inputs_dev[i]);
            if (batch_outputs_dev[i] != nullptr) cudaFree(batch_outputs_dev[i]);
            if (batch_decoders[i] != nullptr)
                mynah_cuda_decoder_close(st, batch_decoders[i]);
            if (batch_cpu[i] != nullptr)
                mynah_seanet_state_destroy(batch_cpu[i]);
        }
    }
    return result;
}

/* One block owns one attention head.  The score reduction is shared by all
 * value lanes; the output accumulator stays in the destination buffer, so a
 * head_width larger than the block size is still supported without a dynamic
 * per-thread array.  This is the same online-softmax recurrence used by the
 * Metal backend and avoids materialising a [heads, valid] score matrix. */
__global__ static void k_self_attention(const float *qkv, float *kcache,
                                        float *vcache, size_t position,
                                        size_t cache_stride, size_t valid,
                                        int heads, int head_width, float scale,
                                        float *out) {
    const int head = (int)blockIdx.x;
    if (head >= heads) return;
    const int tid = (int)threadIdx.x;
    const size_t width = (size_t)heads * (size_t)head_width;
    const size_t hbase = (size_t)head * (size_t)head_width;
    const size_t cache_base = position * cache_stride + hbase;
    const float *q = qkv + hbase;
    const float *k = qkv + width + hbase;
    const float *v = qkv + width * 2u + hbase;
    for (int d = tid; d < head_width; d += (int)blockDim.x) {
        kcache[cache_base + (size_t)d] = k[d];
        vcache[cache_base + (size_t)d] = v[d];
        out[hbase + (size_t)d] = 0.0f;
    }
    __syncthreads();
    __shared__ float partial[256];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float correction;
    __shared__ float probability;
    if (tid == 0) {
        maximum = -1.0e30f;
        denominator = 0.0f;
    }
    __syncthreads();
    for (size_t s = 0; s < valid; ++s) {
        const float *ks = kcache + s * cache_stride + hbase;
        const float *vs = vcache + s * cache_stride + hbase;
        float local = 0.0f;
        for (int d = tid; d < head_width; d += (int)blockDim.x)
            local += q[d] * ks[d];
        partial[tid] = local;
        __syncthreads();
        for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) partial[tid] += partial[tid + offset];
            __syncthreads();
        }
        if (tid == 0) {
            const float score = partial[0] * scale;
            const float next = fmaxf(maximum, score);
            correction = expf(maximum - next);
            probability = expf(score - next);
            denominator = denominator * correction + probability;
            maximum = next;
        }
        __syncthreads();
        for (int d = tid; d < head_width; d += (int)blockDim.x) {
            const size_t index = hbase + (size_t)d;
            out[index] = out[index] * correction + probability * vs[d];
        }
        __syncthreads();
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = tid; d < head_width; d += (int)blockDim.x)
        out[hbase + (size_t)d] *= inv;
}

__global__ static void k_cross_attention(const float *q, const float *kcache,
                                         const float *vcache, size_t valid,
                                         size_t cache_stride, int heads,
                                         int head_width, float scale,
                                         float *out) {
    const int head = (int)blockIdx.x;
    if (head >= heads) return;
    const int tid = (int)threadIdx.x;
    const size_t hbase = (size_t)head * (size_t)head_width;
    __shared__ float partial[256];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float correction;
    __shared__ float probability;
    for (int d = tid; d < head_width; d += (int)blockDim.x)
        out[hbase + (size_t)d] = 0.0f;
    if (tid == 0) {
        maximum = -1.0e30f;
        denominator = 0.0f;
    }
    __syncthreads();
    for (size_t s = 0; s < valid; ++s) {
        const float *ks = kcache + s * cache_stride + hbase;
        const float *vs = vcache + s * cache_stride + hbase;
        float local = 0.0f;
        for (int d = tid; d < head_width; d += (int)blockDim.x)
            local += q[hbase + (size_t)d] * ks[d];
        partial[tid] = local;
        __syncthreads();
        for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) partial[tid] += partial[tid + offset];
            __syncthreads();
        }
        if (tid == 0) {
            const float score = partial[0] * scale;
            const float next = fmaxf(maximum, score);
            correction = expf(maximum - next);
            probability = expf(score - next);
            denominator = denominator * correction + probability;
            maximum = next;
        }
        __syncthreads();
        for (int d = tid; d < head_width; d += (int)blockDim.x) {
            const size_t index = hbase + (size_t)d;
            out[index] = out[index] * correction + probability * vs[d];
        }
        __syncthreads();
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = tid; d < head_width; d += (int)blockDim.x)
        out[hbase + (size_t)d] *= inv;
}

/* SHARED: positions [0, prefix_len) are read from the shared voice prefix
 * planes (stride `width`) instead of the cache, which is then never touched
 * below prefix_len (MYNAH_CUDA_SHARED_VOICE). SHARED=false is the plain
 * kernel; both walk the positions in the same order. */
template <bool SHARED>
__global__ static void k_self_attention_bf16(
    const float *qkv, uint16_t *kcache, uint16_t *vcache, size_t position,
    size_t cache_stride, size_t valid, int heads, int head_width, float scale,
    float *out, const uint16_t *kprefix, const uint16_t *vprefix,
    size_t prefix_len) {
    const int head = (int)blockIdx.x;
    if (head >= heads) return;
    const int tid = (int)threadIdx.x;
    const size_t width = (size_t)heads * (size_t)head_width;
    const size_t hbase = (size_t)head * (size_t)head_width;
    const size_t cache_base = position * cache_stride + hbase;
    const float *q = qkv + hbase;
    const float *k = qkv + width + hbase;
    const float *v = qkv + width * 2u + hbase;
    for (int d = tid; d < head_width; d += (int)blockDim.x) {
        kcache[cache_base + (size_t)d] = cuda_bf16_from_float(k[d]);
        vcache[cache_base + (size_t)d] = cuda_bf16_from_float(v[d]);
        out[hbase + (size_t)d] = 0.0f;
    }
    __syncthreads();
    __shared__ float partial[256];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float correction;
    __shared__ float probability;
    if (tid == 0) {
        maximum = -1.0e30f;
        denominator = 0.0f;
    }
    __syncthreads();
    for (size_t s = 0; s <= position; ++s) {
        const bool from_prefix = SHARED && s < prefix_len;
        const uint16_t *ks = from_prefix ? kprefix + s * width + hbase
                                         : kcache + s * cache_stride + hbase;
        const uint16_t *vs = from_prefix ? vprefix + s * width + hbase
                                         : vcache + s * cache_stride + hbase;
        float local = 0.0f;
        for (int d = tid; d < head_width; d += (int)blockDim.x)
            local += q[d] * cuda_bf16_to_float(ks[d]);
        partial[tid] = local;
        __syncthreads();
        for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) partial[tid] += partial[tid + offset];
            __syncthreads();
        }
        if (tid == 0) {
            const float score = partial[0] * scale;
            const float next = fmaxf(maximum, score);
            correction = expf(maximum - next);
            probability = expf(score - next);
            denominator = denominator * correction + probability;
            maximum = next;
        }
        __syncthreads();
        for (int d = tid; d < head_width; d += (int)blockDim.x) {
            const size_t index = hbase + (size_t)d;
            out[index] = out[index] * correction + probability *
                         cuda_bf16_to_float(vs[d]);
        }
        __syncthreads();
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = tid; d < head_width; d += (int)blockDim.x)
        out[hbase + (size_t)d] *= inv;
}

/* Interleaved RoPE for the fused QKV row.  PocketTTS uses the same absolute
 * position and frequency construction as transformer_ar.c; keeping the
 * rotation on the device avoids a host round-trip between QKV projection and
 * resident attention. */
__global__ static void k_rope_qk(float *qkv, size_t position, int heads,
                                 int head_width, float max_period) {
    const int pair = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int half = head_width / 2;
    const int total = heads * half;
    if (pair >= total) return;
    const int head = pair / half;
    const int i = pair % half;
    const size_t width = (size_t)heads * (size_t)head_width;
    const size_t qbase = (size_t)head * (size_t)head_width + (size_t)(2 * i);
    const size_t kbase = width + qbase;
    /* Match transformer_ar.c: the slope is formed from a double-precision
     * logarithm and rounded once before the float32 exp/angle operations. */
    const float slope = (float)(-log((double)max_period) * 2.0 /
                                (double)head_width);
    const float frequency = expf((float)i * slope);
    const float angle = (float)position * frequency;
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(angle, &sine, &cosine);
    const float q0 = qkv[qbase];
    const float q1 = qkv[qbase + 1u];
    qkv[qbase] = q0 * cosine - q1 * sine;
    qkv[qbase + 1u] = q0 * sine + q1 * cosine;
    const float k0 = qkv[kbase];
    const float k1 = qkv[kbase + 1u];
    qkv[kbase] = k0 * cosine - k1 * sine;
    qkv[kbase + 1u] = k0 * sine + k1 * cosine;
}

__global__ static void k_rope_qk_batch(float *qkv, const size_t *positions,
                                       int batch, int heads, int head_width,
                                       float max_period) {
    const int half = head_width / 2;
    const size_t per_request = (size_t)heads * (size_t)half;
    const size_t pair = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    const size_t total = (size_t)batch * per_request;
    if (pair >= total) return;
    const size_t request = pair / per_request;
    const size_t local = pair % per_request;
    const int head = (int)(local / (size_t)half);
    const int i = (int)(local % (size_t)half);
    const size_t width = (size_t)heads * (size_t)head_width;
    const size_t row = request * width * 3u;
    const size_t qbase = row + (size_t)head * (size_t)head_width +
                         (size_t)(2 * i);
    const size_t kbase = qbase + width;
    const float slope = (float)(-log((double)max_period) * 2.0 /
                                (double)head_width);
    const float frequency = expf((float)i * slope);
    const float angle = (float)positions[request] * frequency;
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(angle, &sine, &cosine);
    const float q0 = qkv[qbase];
    const float q1 = qkv[qbase + 1u];
    qkv[qbase] = q0 * cosine - q1 * sine;
    qkv[qbase + 1u] = q0 * sine + q1 * cosine;
    const float k0 = qkv[kbase];
    const float k1 = qkv[kbase + 1u];
    qkv[kbase] = k0 * cosine - k1 * sine;
    qkv[kbase + 1u] = k0 * sine + k1 * cosine;
}

/* MYNAH_CUDA_BF16_FUSE: k_bias_add on the fused QKV row followed by
 * k_rope_qk_batch.  The bias is added in FP32 before the rotation exactly as
 * the separate epilogue did, and v (which RoPE leaves alone) takes its bias
 * here too: thread `pair` also owns v elements 2i and 2i+1 of its head. */
__global__ static void k_rope_qk_bias_batch(float *qkv, const float *bias,
                                            const size_t *positions,
                                            int batch, int heads,
                                            int head_width,
                                            float max_period) {
    const int half = head_width / 2;
    const size_t per_request = (size_t)heads * (size_t)half;
    const size_t pair = (size_t)blockIdx.x * (size_t)blockDim.x +
                        (size_t)threadIdx.x;
    const size_t total = (size_t)batch * per_request;
    if (pair >= total) return;
    const size_t request = pair / per_request;
    const size_t local = pair % per_request;
    const int head = (int)(local / (size_t)half);
    const int i = (int)(local % (size_t)half);
    const size_t width = (size_t)heads * (size_t)head_width;
    const size_t row = request * width * 3u;
    const size_t column = (size_t)head * (size_t)head_width + (size_t)(2 * i);
    const size_t qbase = row + column;
    const size_t kbase = qbase + width;
    const size_t vbase = kbase + width;
    const float slope = (float)(-log((double)max_period) * 2.0 /
                                (double)head_width);
    const float frequency = expf((float)i * slope);
    const float angle = (float)positions[request] * frequency;
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(angle, &sine, &cosine);
    const float q0 = qkv[qbase] + bias[column];
    const float q1 = qkv[qbase + 1u] + bias[column + 1u];
    qkv[qbase] = q0 * cosine - q1 * sine;
    qkv[qbase + 1u] = q0 * sine + q1 * cosine;
    const float k0 = qkv[kbase] + bias[width + column];
    const float k1 = qkv[kbase + 1u] + bias[width + column + 1u];
    qkv[kbase] = k0 * cosine - k1 * sine;
    qkv[kbase + 1u] = k0 * sine + k1 * cosine;
    qkv[vbase] = qkv[vbase] + bias[2u * width + column];
    qkv[vbase + 1u] = qkv[vbase + 1u] + bias[2u * width + column + 1u];
}

static int attention_threads(size_t head_width) {
    int threads = 32;
    while ((size_t)threads < head_width && threads < 256) threads <<= 1;
    return threads;
}

extern "C" int mynah_cuda_self_attention_dev(
    void *opaque, const float *qkv, float *kcache, float *vcache,
    size_t position, size_t cache_stride, size_t valid, size_t heads,
    size_t head_width, float scale, float *out, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (qkv == nullptr || kcache == nullptr || vcache == nullptr || out == nullptr ||
        heads == 0 || head_width == 0 || valid == 0 || heads > (size_t)INT_MAX ||
        head_width > (size_t)INT_MAX || cache_stride == 0 || position >= valid ||
        heads > SIZE_MAX / head_width) {
        set_error(e, ec, "invalid CUDA self-attention dimensions");
        return -1;
    }
    const size_t width = heads * head_width;
    if (cache_stride < width ||
        (valid - 1u) > (SIZE_MAX - (width - 1u)) / cache_stride) {
        set_error(e, ec, "CUDA self-attention cache stride overflow");
        return -1;
    }
    k_self_attention<<<(int)heads, attention_threads(head_width), 0, st->stream>>>(
        qkv, kcache, vcache, position, cache_stride, valid,
        (int)heads, (int)head_width, scale, out);
    return ce(cudaGetLastError(), e, ec);
}

/* ------------------------------------------------------------------ */
/*  Cross-request causal transformer tile                              */
/* ------------------------------------------------------------------ */

/* C[m][n] = bias[n] + sum_k A[m][k] * W[n][k], A row-major [M][K], W row-major
 * [N][K].  Every output element is one fp32 accumulator walked over k in
 * ascending order, whatever M is and wherever the row sits in the tile, so a
 * request computes the same bits alone or inside a wide gang.  That is the
 * property cuBLAS does not promise (it picks algorithms and split-K by M). */
#define TILE_GEMM_BM 64
#define TILE_GEMM_BN 64
#define TILE_GEMM_BK 16
/* BF16 == true: W is resident bf16 bits and A is rounded to bf16 on load, so
 * the tile computes the same bf16 x bf16 -> fp32 representation as the
 * per-request cuBLAS BF16 path, still with one ordered fp32 accumulator. */
template <bool BF16>
__global__ static void k_tile_gemm(const float *A, const void *Wv,
                                   const float *bias, float *C, int M, int N,
                                   int K) {
    const float *W = static_cast<const float *>(Wv);
    const uint16_t *W16 = static_cast<const uint16_t *>(Wv);
    __shared__ float As[TILE_GEMM_BK][TILE_GEMM_BM + 4];
    __shared__ float Ws[TILE_GEMM_BK][TILE_GEMM_BN + 4];
    const int tid = (int)threadIdx.x;
    const int tx = tid % 16;
    const int ty = tid / 16;
    const int m0 = (int)blockIdx.y * TILE_GEMM_BM;
    const int n0 = (int)blockIdx.x * TILE_GEMM_BN;
    float acc[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE_GEMM_BK) {
        for (int i = 0; i < 4; ++i) {
            const int idx = tid + i * 256;
            const int r = idx / TILE_GEMM_BK;
            const int c = idx % TILE_GEMM_BK;
            const int k = k0 + c;
            if (BF16) {
                As[c][r] = (m0 + r < M && k < K)
                               ? cuda_bf16_to_float(cuda_bf16_from_float(
                                     A[(size_t)(m0 + r) * K + k]))
                               : 0.0f;
                Ws[c][r] = (n0 + r < N && k < K)
                               ? cuda_bf16_to_float(W16[(size_t)(n0 + r) * K + k])
                               : 0.0f;
            } else {
                As[c][r] = (m0 + r < M && k < K) ? A[(size_t)(m0 + r) * K + k] : 0.0f;
                Ws[c][r] = (n0 + r < N && k < K) ? W[(size_t)(n0 + r) * K + k] : 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TILE_GEMM_BK; ++kk) {
            float a[4], w[4];
            for (int i = 0; i < 4; ++i) a[i] = As[kk][ty * 4 + i];
            for (int j = 0; j < 4; ++j) w[j] = Ws[kk][tx * 4 + j];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], w[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx * 4 + j;
            if (n >= N) continue;
            C[(size_t)m * N + n] = acc[i][j] + (bias != nullptr ? bias[n] : 0.0f);
        }
    }
}

/* Split-K variant of k_tile_gemm. The number of splits S depends on the
 * weight's shape (N, K) and the SM count only, never on M; split s is one
 * ascending fma chain over its own k range, and the partials are added in the
 * fixed order ((P0 + P1) + P2) + ... . A row's bits therefore still do not
 * depend on how many other rows share the call, while small-M tiles get
 * enough independent blocks to fill the card. (Same scheme as the mynah-asr
 * cohort GEMM.) */
__global__ static void k_tile_gemm_part(const float *A, const float *W, float *P,
                                        int M, int N, int K, int KC) {
    __shared__ float As[TILE_GEMM_BK][TILE_GEMM_BM + 4];
    __shared__ float Ws[TILE_GEMM_BK][TILE_GEMM_BN + 4];
    const int tid = (int)threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = (int)blockIdx.y * TILE_GEMM_BM;
    const int n0 = (int)blockIdx.x * TILE_GEMM_BN;
    const int kb = (int)blockIdx.z * KC;
    const int ke = min(K, kb + KC);
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = 0.0f;
    const int lrow = tid >> 2, lk = (tid & 3) * 4;
    for (int k0 = kb; k0 < ke; k0 += TILE_GEMM_BK) {
        {
            const int m = m0 + lrow;
            const float *src = A + (size_t)m * K + k0 + lk;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int k = k0 + lk + u;
                As[lk + u][lrow] = (m < M && k < ke) ? src[u] : 0.0f;
            }
        }
        {
            const int n = n0 + lrow;
            const float *src = W + (size_t)n * K + k0 + lk;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const int k = k0 + lk + u;
                Ws[lk + u][lrow] = (n < N && k < ke) ? src[u] : 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE_GEMM_BK; ++k) {
            float a[4], w[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) a[i] = As[k][ty * 4 + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) w[j] = Ws[k][tx * 4 + j];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(a[i], w[j], acc[i][j]);
        }
        __syncthreads();
    }
    float *Ps = P + (size_t)blockIdx.z * (size_t)M * (size_t)N;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = m0 + ty * 4 + i;
        if (m >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx * 4 + j;
            if (n < N) Ps[(size_t)m * N + n] = acc[i][j];
        }
    }
}

__global__ static void k_tile_gemm_reduce(const float *P, int S, int M, int N,
                                          const float *bias, float *C) {
    const int n = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int m = (int)blockIdx.y;
    if (n >= N || m >= M) return;
    const size_t mn = (size_t)M * (size_t)N, o = (size_t)m * N + n;
    float v = P[o];
    for (int sp = 1; sp < S; ++sp) v += P[(size_t)sp * mn + o];
    C[o] = v + (bias != nullptr ? bias[n] : 0.0f);
}

static int tile_gemm_splits(int sms, int N, int K) {
    const int ntiles = (N + TILE_GEMM_BN - 1) / TILE_GEMM_BN;
    int S = (2 * sms + ntiles - 1) / ntiles;
    const int smax = K / 256;
    if (S > smax) S = smax;
    return S < 1 ? 1 : S;
}

__global__ static void k_bias_rows(float *C, const float *bias, int M, int N) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < M * N) C[i] += bias[i % N];
}

/* The tile's rows are packed: row m is token `rowmap[m].y` of request
 * `rowmap[m].x`, so requests may contribute different token counts. */
__global__ static void k_tile_gather(const float *const *inputs,
                                     const int2 *rowmap, float *x, int dim,
                                     int total) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= total) return;
    const int2 rt = rowmap[i / dim];
    x[i] = inputs[rt.x][(size_t)rt.y * dim + (i % dim)];
}

/* Row-major [m][dim] to each request's channel-major [dim][positions]. */
__global__ static void k_tile_scatter(const float *x, float *const *outputs,
                                      const int2 *rowmap, int positions,
                                      int dim, int total) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= total) return;
    const int2 rt = rowmap[i / dim];
    outputs[rt.x][(size_t)(i % dim) * positions + rt.y] = x[i];
}

__device__ static inline float tile_kv_load(const float *p) { return *p; }
__device__ static inline float tile_kv_load(const uint16_t *p) {
    return cuda_bf16_to_float(*p);
}
__device__ static inline void tile_kv_store(float *p, float v) { *p = v; }
__device__ static inline void tile_kv_store(uint16_t *p, float v) {
    *p = cuda_bf16_from_float(v);
}

/* int8 backbone KV (default, MYNAH_CUDA_KV_DTYPE): a stored position of a K or V plane is one
 * record of `dim` int8 values followed by `heads` float scales; value d of
 * head h is rec[h * hw + d] * scale[h], scale = max|x| / 127 over the head.
 * The shared prefix planes stay BF16, so the prefix element type differs
 * from the row's only for int8. */
template <typename KV> struct tile_prefix_type { typedef KV type; };
template <> struct tile_prefix_type<int8_t> { typedef uint16_t type; };

template <typename KV>
__device__ static inline float tile_row_load(const KV *rec, int dim, int h,
                                             int hw, int d) {
    (void)dim;
    return tile_kv_load(rec + (size_t)h * hw + d);
}
template <>
__device__ inline float tile_row_load<int8_t>(const int8_t *rec, int dim,
                                              int h, int hw, int d) {
    const float scale = *reinterpret_cast<const float *>(rec + dim + 4 * h);
    return (float)rec[(size_t)h * hw + d] * scale;
}

__device__ static inline int8_t cuda_kv_q8(float x, float inv) {
    float q = rintf(x * inv);
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    return (int8_t)q;
}

/* RoPE on q and k at each row's absolute position (the formula of k_rope_qk),
 * then k and v into the request's cache at slot (absolute % ring), layout
 * [K ring][V ring] per layer. One block per row.
 *
 * PREFIX (mynah_backend_tile_desc.skip): the row does not store its first
 * skip[r] positions, the slot is (absolute - skip) % ring and `ring` counts
 * the stored slots. The host refuses a start below the skip, so nothing is
 * written there. PREFIX=false is the plain layout, the same code as before. */
/* `strides` (MYNAH_CUDA_KV_VMM, desc.kv_strides): per row, elements between
 * consecutive slots and from the K slot to the V slot. NULL = the plain
 * layout (dim, ring * dim), the arithmetic this kernel always did. */
template <typename KV, bool PREFIX>
__global__ static void k_tile_rope_store(float *qkv, void *const *kv,
                                         const long long *start,
                                         const int2 *rowmap, int layer,
                                         int layers, int heads, int head_width,
                                         const long long *rings, float max_period,
                                         const long long *skips,
                                         const long long *strides) {
    const int m = (int)blockIdx.x;
    const int2 rt = rowmap[m];
    const long long ring = rings[rt.x];
    const long long absolute = start[rt.x] + rt.y;
    const long long skip = PREFIX ? skips[rt.x] : 0;
    const long long slot = (absolute - skip) % ring;
    const int half = head_width / 2;
    const int dim = heads * head_width;
    float *row = qkv + (size_t)m * 3u * (size_t)dim;
    KV *kbase = static_cast<KV *>(kv[(size_t)rt.x * layers + layer]);
    const size_t pitch = strides != nullptr ? (size_t)strides[2 * rt.x] : (size_t)dim;
    const size_t voff = strides != nullptr ? (size_t)strides[2 * rt.x + 1]
                                           : (size_t)ring * dim;
    KV *kslot = kbase + (size_t)slot * pitch;
    KV *vslot = kbase + voff + (size_t)slot * pitch;
    const float slope = (float)(-log((double)max_period) * 2.0 /
                                (double)head_width);
    for (int pair = (int)threadIdx.x; pair < heads * half;
         pair += (int)blockDim.x) {
        const int head = pair / half;
        const int i = pair % half;
        const size_t qi = (size_t)head * head_width + (size_t)(2 * i);
        const float frequency = expf((float)i * slope);
        const float angle = (float)absolute * frequency;
        float sine = 0.0f, cosine = 0.0f;
        sincosf(angle, &sine, &cosine);
        const float q0 = row[qi], q1 = row[qi + 1u];
        row[qi] = q0 * cosine - q1 * sine;
        row[qi + 1u] = q0 * sine + q1 * cosine;
        const float k0 = row[dim + qi], k1 = row[dim + qi + 1u];
        tile_kv_store(kslot + qi, k0 * cosine - k1 * sine);
        tile_kv_store(kslot + qi + 1u, k0 * sine + k1 * cosine);
        tile_kv_store(vslot + qi, row[2 * dim + qi]);
        tile_kv_store(vslot + qi + 1u, row[2 * dim + qi + 1u]);
    }
}

/* The int8 form of k_tile_rope_store (int8 backbone KV): one warp
 * per head (blockDim = heads * 32), the same RoPE formula; the rotated k is
 * written back into qkv (nothing reads qkv's k after this) so the second pass
 * quantizes what the first one rotated. `strides` is required: (pitch, voff)
 * in bytes. */
template <bool PREFIX>
__global__ static void k_tile_rope_store_i8(float *qkv, void *const *kv,
                                            const long long *start,
                                            const int2 *rowmap, int layer,
                                            int layers, int heads,
                                            int head_width,
                                            const long long *rings,
                                            float max_period,
                                            const long long *skips,
                                            const long long *strides) {
    const int m = (int)blockIdx.x;
    const int head = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    if (head >= heads) return;
    const int2 rt = rowmap[m];
    const long long ring = rings[rt.x];
    const long long absolute = start[rt.x] + rt.y;
    const long long skip = PREFIX ? skips[rt.x] : 0;
    const long long slot = (absolute - skip) % ring;
    const int half = head_width / 2;
    const int dim = heads * head_width;
    float *row = qkv + (size_t)m * 3u * (size_t)dim;
    int8_t *kbase = static_cast<int8_t *>(kv[(size_t)rt.x * layers + layer]);
    const size_t pitch = (size_t)strides[2 * rt.x];
    const size_t voff = (size_t)strides[2 * rt.x + 1];
    int8_t *kslot = kbase + (size_t)slot * pitch;
    int8_t *vslot = kbase + voff + (size_t)slot * pitch;
    const float slope = (float)(-log((double)max_period) * 2.0 /
                                (double)head_width);
    float kmax = 0.0f, vmax = 0.0f;
    for (int i = lane; i < half; i += 32) {
        const size_t qi = (size_t)head * head_width + (size_t)(2 * i);
        const float frequency = expf((float)i * slope);
        const float angle = (float)absolute * frequency;
        float sine = 0.0f, cosine = 0.0f;
        sincosf(angle, &sine, &cosine);
        const float q0 = row[qi], q1 = row[qi + 1u];
        row[qi] = q0 * cosine - q1 * sine;
        row[qi + 1u] = q0 * sine + q1 * cosine;
        const float k0 = row[dim + qi], k1 = row[dim + qi + 1u];
        const float r0 = k0 * cosine - k1 * sine;
        const float r1 = k0 * sine + k1 * cosine;
        row[dim + qi] = r0;
        row[dim + qi + 1u] = r1;
        kmax = fmaxf(kmax, fmaxf(fabsf(r0), fabsf(r1)));
        vmax = fmaxf(vmax, fmaxf(fabsf(row[2 * dim + qi]),
                                 fabsf(row[2 * dim + qi + 1u])));
    }
    for (int off = 16; off > 0; off >>= 1) {
        kmax = fmaxf(kmax, __shfl_xor_sync(0xffffffffu, kmax, off));
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, off));
    }
    const float kinv = kmax > 0.0f ? 127.0f / kmax : 0.0f;
    const float vinv = vmax > 0.0f ? 127.0f / vmax : 0.0f;
    for (int i = lane; i < half; i += 32) {
        const size_t qi = (size_t)head * head_width + (size_t)(2 * i);
        kslot[qi] = cuda_kv_q8(row[dim + qi], kinv);
        kslot[qi + 1u] = cuda_kv_q8(row[dim + qi + 1u], kinv);
        vslot[qi] = cuda_kv_q8(row[2 * dim + qi], vinv);
        vslot[qi + 1u] = cuda_kv_q8(row[2 * dim + qi + 1u], vinv);
    }
    if (lane == 0) {
        reinterpret_cast<float *>(kslot + dim)[head] = kmax / 127.0f;
        reinterpret_cast<float *>(vslot + dim)[head] = vmax / 127.0f;
    }
}

/* One warp per (row, head): scores over the causal window in shared memory,
 * a two-pass softmax, then each lane owns output dims. Every reduction order
 * depends only on the window length, never on the batch. `context` 0 means
 * the whole prefix; `window_cap` bounds the per-warp score buffer.
 *
 * PREFIX: positions below skip[r] are read from the shared planes
 * prefix[r * layers + layer] ([K skip][V skip] x dim), the rest from the row
 * at slot (p - skip) % ring. Same values in the same order as a row that
 * stores its prefix, so the result is bit-identical. */
template <typename KV, bool PREFIX>
__global__ static void k_tile_attention(const float *qkv, void *const *kv,
                                        const long long *start,
                                        const int2 *rowmap, int layer,
                                        int layers, int heads, int head_width,
                                        long long context, const long long *rings,
                                        int window_cap, float scale, float *out,
                                        int rows_total, const long long *skips,
                                        void *const *prefix,
                                        const long long *strides) {
    extern __shared__ float tile_scores[];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int work = (int)blockIdx.x * ((int)blockDim.x >> 5) + warp;
    if (work >= rows_total * heads) return;
    const int m = work / heads;
    const int h = work % heads;
    const int2 rt = rowmap[m];
    const long long ring = rings[rt.x];
    const long long absolute = start[rt.x] + rt.y;
    const long long first =
        (context > 0 && absolute + 1 > context) ? absolute + 1 - context : 0;
    const int n = (int)(absolute - first + 1);
    if (n > window_cap) return; /* refused on the host; never reached */
    const int dim = heads * head_width;
    float *scores = tile_scores + (size_t)warp * window_cap;
    const float *q = qkv + (size_t)m * 3u * dim + (size_t)h * head_width;
    typedef typename tile_prefix_type<KV>::type PKV;
    const KV *kbase = static_cast<const KV *>(kv[(size_t)rt.x * layers + layer]);
    const size_t pitch = strides != nullptr ? (size_t)strides[2 * rt.x] : (size_t)dim;
    const KV *vbase = kbase + (strides != nullptr ? (size_t)strides[2 * rt.x + 1]
                                                  : (size_t)ring * dim);
    const long long skip = PREFIX ? skips[rt.x] : 0;
    const PKV *pkbase = PREFIX && skip > 0
        ? static_cast<const PKV *>(prefix[(size_t)rt.x * layers + layer]) : nullptr;
    const PKV *pvbase = PREFIX && skip > 0 ? pkbase + (size_t)skip * dim : nullptr;
    float local_max = -INFINITY;
    for (int s = lane; s < n; s += 32) {
        const long long p = first + s;
        float dot = 0.0f;
        if (PREFIX && p < skip) {
            const PKV *k = pkbase + (size_t)p * dim + (size_t)h * head_width;
            for (int d = 0; d < head_width; ++d)
                dot = fmaf(q[d], tile_kv_load(k + d), dot);
        } else {
            const KV *rec = kbase + (size_t)((p - skip) % ring) * pitch;
            for (int d = 0; d < head_width; ++d)
                dot = fmaf(q[d], tile_row_load(rec, dim, h, head_width, d), dot);
        }
        const float score = dot * scale;
        scores[s] = score;
        local_max = fmaxf(local_max, score);
    }
    for (int off = 16; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, off));
    float local_sum = 0.0f;
    for (int s = lane; s < n; s += 32) {
        const float p = expf(scores[s] - local_max);
        scores[s] = p;
        local_sum += p;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(0xffffffffu, local_sum, off);
    __syncwarp();
    const float inv = local_sum > 0.0f ? 1.0f / local_sum : 0.0f;
    for (int d = lane; d < head_width; d += 32) {
        float acc = 0.0f;
        for (int s = 0; s < n; ++s) {
            const long long p = first + s;
            const float value = (PREFIX && p < skip)
                ? tile_kv_load(pvbase + (size_t)p * dim + (size_t)h * head_width + d)
                : tile_row_load(vbase + (size_t)((p - skip) % ring) * pitch, dim,
                                h, head_width, d);
            acc = fmaf(scores[s], value, acc);
        }
        out[(size_t)m * dim + (size_t)h * head_width + d] = acc * inv;
    }
}

/* Grouped variant: one block per (group of up to TILE_ATTN_Q consecutive
 * tokens of one request, head). The K/V window is streamed through shared
 * memory in TILE_ATTN_CH-position chunks that every query of the group
 * reuses, instead of each (token, head) warp reading the whole window from
 * global memory. One warp per query, online softmax over the chunks. The
 * result depends only on the request's own tokens and cache. */
#define TILE_ATTN_Q 16
#define TILE_ATTN_CH 32
#define TILE_ATTN_HW 128
/* PREFIX: as in k_tile_attention, positions below skip[r] come from the
 * shared prefix planes and the row stores slot (p - skip) % ring. */
template <typename KV, bool PREFIX>
__global__ static void k_tile_attention_grouped(
    const float *qkv, void *const *kv, const long long *start,
    const int2 *rowmap, const int2 *groups, int layer, int layers, int heads,
    int head_width, long long context, const long long *rings, float scale,
    float *out, const long long *skips, void *const *prefix,
    const long long *strides) {
    __shared__ float ks[TILE_ATTN_CH][TILE_ATTN_HW + 1];
    __shared__ float vs[TILE_ATTN_CH][TILE_ATTN_HW];
    __shared__ float qs[TILE_ATTN_Q][TILE_ATTN_HW];
    const int2 group = groups[blockIdx.x];   /* x = first packed row, y = rows */
    const int h = (int)blockIdx.y;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int dim = heads * head_width;
    const int2 rt0 = rowmap[group.x];
    const int r = rt0.x;
    const long long ring = rings[r];
    const long long abs0 = start[r] + rt0.y;
    const long long abs_last = abs0 + group.y - 1;
    const long long lo = (context > 0 && abs0 + 1 > context) ? abs0 + 1 - context : 0;
    typedef typename tile_prefix_type<KV>::type PKV;
    const KV *kbase = static_cast<const KV *>(kv[(size_t)r * layers + layer]);
    const size_t pitch = strides != nullptr ? (size_t)strides[2 * r] : (size_t)dim;
    const KV *vbase = kbase + (strides != nullptr ? (size_t)strides[2 * r + 1]
                                                  : (size_t)ring * dim);
    const long long skip = PREFIX ? skips[r] : 0;
    const PKV *pkbase = PREFIX && skip > 0
        ? static_cast<const PKV *>(prefix[(size_t)r * layers + layer]) : nullptr;
    const PKV *pvbase = PREFIX && skip > 0 ? pkbase + (size_t)skip * dim : nullptr;
    for (int i = (int)threadIdx.x; i < group.y * head_width; i += (int)blockDim.x) {
        const int q = i / head_width, d = i % head_width;
        qs[q][d] = qkv[(size_t)(group.x + q) * 3u * dim + (size_t)h * head_width + d];
    }
    const bool active = warp < group.y;
    const long long my_abs = abs0 + warp;
    const long long my_first =
        (context > 0 && my_abs + 1 > context) ? my_abs + 1 - context : 0;
    float running_max = -INFINITY, running_sum = 0.0f;
    float acc[TILE_ATTN_HW / 32];
    for (int k = 0; k < TILE_ATTN_HW / 32; ++k) acc[k] = 0.0f;
    for (long long c0 = lo; c0 <= abs_last; c0 += TILE_ATTN_CH) {
        __syncthreads();
        for (int i = (int)threadIdx.x; i < TILE_ATTN_CH * head_width;
             i += (int)blockDim.x) {
            const int j = i / head_width, d = i % head_width;
            const long long p = c0 + j;
            float kval = 0.0f, vval = 0.0f;
            if (PREFIX && p < skip) {
                const size_t at = (size_t)p * dim + (size_t)h * head_width + d;
                kval = tile_kv_load(pkbase + at);
                vval = tile_kv_load(pvbase + at);
            } else if (p <= abs_last) {
                const size_t slot = (size_t)((p - skip) % ring) * pitch;
                kval = tile_row_load(kbase + slot, dim, h, head_width, d);
                vval = tile_row_load(vbase + slot, dim, h, head_width, d);
            }
            ks[j][d] = kval;
            vs[j][d] = vval;
        }
        __syncthreads();
        if (!active) continue;
        const long long p = c0 + lane;
        float score = -INFINITY;
        if (p >= my_first && p <= my_abs) {
            float dot = 0.0f;
            for (int d = 0; d < head_width; ++d) dot = fmaf(qs[warp][d], ks[lane][d], dot);
            score = dot * scale;
        }
        float chunk_max = score;
        for (int off = 16; off > 0; off >>= 1)
            chunk_max = fmaxf(chunk_max, __shfl_xor_sync(0xffffffffu, chunk_max, off));
        if (chunk_max == -INFINITY) continue;
        const float next_max = fmaxf(running_max, chunk_max);
        const float correction = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
        const float prob = score == -INFINITY ? 0.0f : expf(score - next_max);
        float chunk_sum = prob;
        for (int off = 16; off > 0; off >>= 1)
            chunk_sum += __shfl_xor_sync(0xffffffffu, chunk_sum, off);
        running_sum = running_sum * correction + chunk_sum;
        running_max = next_max;
        for (int k = 0; k < TILE_ATTN_HW / 32; ++k) acc[k] *= correction;
        for (int j = 0; j < TILE_ATTN_CH; ++j) {
            const float pj = __shfl_sync(0xffffffffu, prob, j);
            if (pj == 0.0f) continue;
            for (int k = 0; k < TILE_ATTN_HW / 32; ++k) {
                const int d = lane + 32 * k;
                if (d < head_width) acc[k] = fmaf(pj, vs[j][d], acc[k]);
            }
        }
    }
    if (!active) return;
    const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    float *o = out + (size_t)(group.x + warp) * dim + (size_t)h * head_width;
    for (int k = 0; k < TILE_ATTN_HW / 32; ++k) {
        const int d = lane + 32 * k;
        if (d < head_width) o[d] = acc[k] * inv;
    }
}

/* The RoPE + K/V store launch: the element-wise kernel for f32/bf16, the
 * warp-per-head quantizing kernel for int8 records. */
template <typename KV, bool PREFIX>
struct tile_rope_store_launcher {
    static void run(cuda_backend_state *st, float *qkv, void *const *kv,
                    const long long *start, const int2 *rowmap, int layer,
                    int layers, int heads, int head_width,
                    const long long *rings, float max_period, size_t M,
                    const long long *skips, const long long *strides) {
        k_tile_rope_store<KV, PREFIX><<<(int)M, 256, 0, st->stream>>>(
            qkv, kv, start, rowmap, layer, layers, heads, head_width, rings,
            max_period, skips, strides);
    }
};
template <bool PREFIX>
struct tile_rope_store_launcher<int8_t, PREFIX> {
    static void run(cuda_backend_state *st, float *qkv, void *const *kv,
                    const long long *start, const int2 *rowmap, int layer,
                    int layers, int heads, int head_width,
                    const long long *rings, float max_period, size_t M,
                    const long long *skips, const long long *strides) {
        k_tile_rope_store_i8<PREFIX><<<(int)M, heads * 32, 0, st->stream>>>(
            qkv, kv, start, rowmap, layer, layers, heads, head_width, rings,
            max_period, skips, strides);
    }
};
template <typename KV, bool PREFIX>
static void tile_rope_store_launch(cuda_backend_state *st, float *qkv,
                                   void *const *kv, const long long *start,
                                   const int2 *rowmap, int layer, int layers,
                                   int heads, int head_width,
                                   const long long *rings, float max_period,
                                   size_t M, const long long *skips,
                                   const long long *strides) {
    tile_rope_store_launcher<KV, PREFIX>::run(st, qkv, kv, start, rowmap, layer,
                                              layers, heads, head_width, rings,
                                              max_period, M, skips, strides);
}

/* One layer's RoPE + K/V store + attention of the tile, the launches the
 * driver below always made, with the PREFIX variants selected by template. */
template <typename KV, bool PREFIX>
static void tile_attention_layer(cuda_backend_state *st, float *qkv,
                                 void *const *kv, const long long *start,
                                 const int2 *rowmap, const int2 *groups,
                                 int layer, int layers, int heads,
                                 int head_width, long long context,
                                 const long long *rings, float max_period,
                                 size_t M, bool use_grouped, dim3 group_grid,
                                 unsigned blocks, int warps_per_block,
                                 size_t smem, int window, float scale,
                                 float *att, const long long *skips,
                                 void *const *prefix,
                                 const long long *strides) {
    tile_rope_store_launch<KV, PREFIX>(st, qkv, kv, start, rowmap, layer,
                                       layers, heads, head_width, rings,
                                       max_period, M, skips, strides);
    if (use_grouped)
        k_tile_attention_grouped<KV, PREFIX><<<group_grid, TILE_ATTN_Q * 32, 0,
                                               st->stream>>>(
            qkv, kv, start, rowmap, groups, layer, layers, heads, head_width,
            context, rings, scale, att, skips, prefix, strides);
    else
        k_tile_attention<KV, PREFIX><<<blocks, warps_per_block * 32, smem,
                                       st->stream>>>(
            qkv, kv, start, rowmap, layer, layers, heads, head_width, context,
            rings, window, scale, att, (int)M, skips, prefix, strides);
}

static int tile_reserve(cuda_backend_state *st, size_t rows, size_t dim,
                        size_t ffn, size_t meta_bytes, char *e, size_t ec) {
    cuda_tile_workspace &w = st->tile;
    size_t md = 0u, mf = 0u;
    if (!cuda_size_mul(rows, dim, &md) || !cuda_size_mul(rows, ffn, &mf) ||
        md > (size_t)INT_MAX / 3u || mf > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA tile workspace size overflow");
        return -1;
    }
    if (md > w.md_cap || mf > w.mf_cap) {
        cudaFree(w.x); cudaFree(w.xn); cudaFree(w.qkv);
        cudaFree(w.att); cudaFree(w.proj); cudaFree(w.ffn_buf);
        w.x = w.xn = w.qkv = w.att = w.proj = w.ffn_buf = nullptr;
        if (md < w.md_cap) md = w.md_cap;
        if (mf < w.mf_cap) mf = w.mf_cap;
        w.md_cap = w.mf_cap = 0u;
        if (ce(cudaMalloc(&w.x, md * sizeof(float)), e, ec) ||
            ce(cudaMalloc(&w.xn, md * sizeof(float)), e, ec) ||
            ce(cudaMalloc(&w.qkv, 3u * md * sizeof(float)), e, ec) ||
            ce(cudaMalloc(&w.att, md * sizeof(float)), e, ec) ||
            ce(cudaMalloc(&w.proj, md * sizeof(float)), e, ec) ||
            ce(cudaMalloc(&w.ffn_buf, mf * sizeof(float)), e, ec))
            return -1;
        w.md_cap = md;
        w.mf_cap = mf;
    }
    if (meta_bytes > w.meta_cap) {
        if (w.meta_event != nullptr) (void)cudaEventSynchronize(w.meta_event);
        cudaFree(w.meta_dev);
        cudaFreeHost(w.meta_host);
        w.meta_dev = nullptr;
        w.meta_host = nullptr;
        w.meta_cap = 0u;
        if (ce(cudaMalloc(&w.meta_dev, meta_bytes), e, ec) ||
            ce(cudaHostAlloc(&w.meta_host, meta_bytes, cudaHostAllocDefault), e, ec))
            return -1;
        w.meta_cap = meta_bytes;
    }
    if (w.meta_event == nullptr &&
        ce(cudaEventCreateWithFlags(&w.meta_event, cudaEventDisableTiming), e, ec))
        return -1;
    if (w.sms == 0) {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&w.sms, cudaDevAttrMultiProcessorCount, dev) !=
                cudaSuccess || w.sms <= 0)
            w.sms = 40;
    }
    /* The widest split-K partial buffer any projection of this tile needs. */
    const size_t shapes[4][2] = {{3u * dim, dim}, {dim, dim}, {ffn, dim}, {dim, ffn}};
    size_t need = 0u;
    for (const auto &shape : shapes) {
        const int S = tile_gemm_splits(w.sms, (int)shape[0], (int)shape[1]);
        if (S <= 1) continue;
        const size_t floats = (size_t)S * rows * shape[0];
        if (floats > need) need = floats;
    }
    if (need > w.splitk_cap) {
        cudaFree(w.splitk);
        w.splitk = nullptr;
        w.splitk_cap = 0u;
        if (ce(cudaMalloc(&w.splitk, need * sizeof(float)), e, ec)) return -1;
        w.splitk_cap = need;
    }
    const size_t a16_need = rows * (ffn > dim ? ffn : dim);
    if (st->tile_cublas && a16_need > w.a16_cap) {
        cudaFree(w.a16);
        w.a16 = nullptr;
        w.a16_cap = 0u;
        if (ce(cudaMalloc((void **)&w.a16, a16_need * sizeof(uint16_t)), e, ec))
            return -1;
        w.a16_cap = a16_need;
    }
    return 0;
}

static void tile_release(cuda_backend_state *st) {
    cuda_tile_workspace &w = st->tile;
    cudaFree(w.x); cudaFree(w.xn); cudaFree(w.qkv);
    cudaFree(w.att); cudaFree(w.proj); cudaFree(w.ffn_buf);
    cudaFree(w.meta_dev);
    cudaFree(w.splitk);
    cudaFree(w.a16);
    if (w.meta_host != nullptr) cudaFreeHost(w.meta_host);
    if (w.meta_event != nullptr) cudaEventDestroy(w.meta_event);
    w = cuda_tile_workspace();
}

/* Fixed-order tile GEMM on TF32 tensor cores: C[M,N] = A[M,K] W[N,K]^T + b.
 * Each output accumulates over K in the same order (32-wide slabs, then
 * 8-wide MMA steps) whatever M is and whichever block computes it, so a row's
 * bits do not depend on the other rows of the call; it keeps the prefill
 * invariance of the SIMT kernel at tensor-core speed.  Needs sm_80+. */
static constexpr int TILE_TC_BM = 64, TILE_TC_BN = 64, TILE_TC_BK = 32;
static constexpr int TILE_TC_LD = TILE_TC_BK + 4;
static_assert(2 * TILE_TC_BM * TILE_TC_LD >= TILE_TC_BM * (TILE_TC_BN + 4),
              "the epilogue reuses the operand staging buffer");

__global__ static void __launch_bounds__(128)
k_tile_gemm_tc(const float *A, const float *W, const float *bias, float *C,
               int M, int N, int K, int KC, float *P) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    __shared__ __align__(32) float sm[2 * TILE_TC_BM * TILE_TC_LD];
    float *As = sm;
    float *Ws = sm + TILE_TC_BM * TILE_TC_LD;
    const int m0 = (int)blockIdx.y * TILE_TC_BM;
    const int n0 = (int)blockIdx.x * TILE_TC_BN;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    /* Split-K: block z owns K range [z*KC, min(K, (z+1)*KC)); the split
     * count depends on N and K only, and k_tile_gemm_reduce adds the
     * partials in split order, so the result stays independent of M. */
    const int kbeg = (int)blockIdx.z * KC;
    const int kend = kbeg + KC < K ? kbeg + KC : K;
    for (int k0 = kbeg; k0 < kend; k0 += TILE_TC_BK) {
        for (int i = 0; i < (TILE_TC_BM * TILE_TC_BK) / 128; ++i) {
            const int idx = tid + 128 * i;
            const int r = idx / TILE_TC_BK, c = idx % TILE_TC_BK;
            const int gk = k0 + c;
            const int gm = m0 + r, gn = n0 + r;
            As[r * TILE_TC_LD + c] = gm < M && gk < kend
                ? wmma::__float_to_tf32(A[(size_t)gm * (size_t)K + (size_t)gk]) : 0.0f;
            Ws[r * TILE_TC_LD + c] = gn < N && gk < kend
                ? wmma::__float_to_tf32(W[(size_t)gn * (size_t)K + (size_t)gk]) : 0.0f;
        }
        __syncthreads();
        for (int kk = 0; kk < TILE_TC_BK; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                           wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                           wmma::col_major> b[2];
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(a[i], As + (wm * 32 + i * 16) * TILE_TC_LD + kk,
                                       TILE_TC_LD);
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(b[j], Ws + (wn * 32 + j * 16) * TILE_TC_LD + kk,
                                       TILE_TC_LD);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    constexpr int CLD = TILE_TC_BN + 4;
    float *Cs = sm;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(Cs + (wm * 32 + i * 16) * CLD + wn * 32 + j * 16,
                                    acc[i][j], CLD, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < TILE_TC_BM * TILE_TC_BN; idx += 128) {
        const int r = idx / TILE_TC_BN, c = idx % TILE_TC_BN;
        const int gm = m0 + r, gn = n0 + c;
        if (gm >= M || gn >= N) continue;
        const size_t o = (size_t)gm * (size_t)N + (size_t)gn;
        if (P != nullptr)
            P[(size_t)blockIdx.z * (size_t)M * (size_t)N + o] = Cs[r * CLD + c];
        else
            C[o] = Cs[r * CLD + c] + (bias != nullptr ? bias[gn] : 0.0f);
    }
#endif
}

/* MYNAH_CUDA_TILE_TC=2: the same fixed-order TF32 tile with 4x the work per
 * block. 128x128 outputs per block, 8 warps of 32x64, and the next K slab
 * prefetched into registers while the current one is multiplied. Every output
 * still accumulates in exactly k_tile_gemm_tc's order: the same split-K ranges,
 * 32-wide slabs ascending, 8-wide MMA steps ascending, operands rounded with
 * __float_to_tf32, and 16x16 fragments that start on multiples of 16 in both M
 * and N. The results are therefore bit-identical to k_tile_gemm_tc (the box
 * check compares them), and the prefill invariance carries over. The
 * epilogue goes through a small per-warp staging tile, so the static shared
 * memory stays under 48 KiB without an opt-in. */
static constexpr int TC2_BM = 128, TC2_BN = 128, TC2_BK = 32, TC2_LD = TC2_BK + 4;
static constexpr int TC2_THREADS = 256;
static constexpr int TC2_SLD = 20; /* per-warp 16x16 staging, padded */

__global__ static void __launch_bounds__(TC2_THREADS)
k_tile_gemm_tc2(const float *A, const float *W, const float *bias, float *C,
                int M, int N, int K, int KC, float *P) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    __shared__ __align__(32) float As[TC2_BM * TC2_LD];
    __shared__ __align__(32) float Ws[TC2_BN * TC2_LD];
    __shared__ __align__(32) float St[8 * 16 * TC2_SLD];
    const int m0 = (int)blockIdx.y * TC2_BM;
    const int n0 = (int)blockIdx.x * TC2_BN;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1;   /* 4 x 2 warps: 32 rows x 64 cols each */
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][4];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    const int kbeg = (int)blockIdx.z * KC;
    const int kend = kbeg + KC < K ? kbeg + KC : K;
    constexpr int PER = (TC2_BM * TC2_BK) / TC2_THREADS; /* 16 per operand */
    float ra[PER], rw[PER];
    auto fetch = [&](int k0) {
        for (int i = 0; i < PER; ++i) {
            const int idx = tid + TC2_THREADS * i;
            const int r = idx / TC2_BK, c = idx % TC2_BK;
            const int gk = k0 + c, gm = m0 + r, gn = n0 + r;
            ra[i] = gm < M && gk < kend
                ? wmma::__float_to_tf32(A[(size_t)gm * (size_t)K + (size_t)gk]) : 0.0f;
            rw[i] = gn < N && gk < kend
                ? wmma::__float_to_tf32(W[(size_t)gn * (size_t)K + (size_t)gk]) : 0.0f;
        }
    };
    auto stash = [&]() {
        for (int i = 0; i < PER; ++i) {
            const int idx = tid + TC2_THREADS * i;
            const int r = idx / TC2_BK, c = idx % TC2_BK;
            As[r * TC2_LD + c] = ra[i];
            Ws[r * TC2_LD + c] = rw[i];
        }
    };
    if (kbeg < kend) fetch(kbeg);
    for (int k0 = kbeg; k0 < kend; k0 += TC2_BK) {
        stash();
        __syncthreads();
        if (k0 + TC2_BK < kend) fetch(k0 + TC2_BK);   /* overlaps the MMAs below */
        for (int kk = 0; kk < TC2_BK; kk += 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                           wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                           wmma::col_major> b[4];
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(a[i], As + (wm * 32 + i * 16) * TC2_LD + kk, TC2_LD);
            for (int j = 0; j < 4; ++j)
                wmma::load_matrix_sync(b[j], Ws + (wn * 64 + j * 16) * TC2_LD + kk, TC2_LD);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 4; ++j)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    float *st = St + warp * 16 * TC2_SLD;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            wmma::store_matrix_sync(st, acc[i][j], TC2_SLD, wmma::mem_row_major);
            __syncwarp();
            const int rb = m0 + wm * 32 + i * 16, cb = n0 + wn * 64 + j * 16;
            for (int e = lane; e < 256; e += 32) {
                const int r = e >> 4, c = e & 15;
                const int gm = rb + r, gn = cb + c;
                if (gm >= M || gn >= N) continue;
                const size_t o = (size_t)gm * (size_t)N + (size_t)gn;
                const float v = st[r * TC2_SLD + c];
                if (P != nullptr)
                    P[(size_t)blockIdx.z * (size_t)M * (size_t)N + o] = v;
                else
                    C[o] = v + (bias != nullptr ? bias[gn] : 0.0f);
            }
            __syncwarp();
        }
    }
#endif
}

/* MYNAH_CUDA_PREFILL_BF16TC: the fixed-order prefill tile on BF16 tensor
 * cores. The resident BF16 weight copy halves the weight traffic that bounds
 * the prefill (all 24 layers' weights are read once per tile call), the
 * activations are rounded to BF16 as they are staged, and the MMAs accumulate
 * in fp32. The order is fixed exactly as in k_tile_gemm_tc: split-K ranges
 * that depend on N and K only, 32-wide K slabs ascending, 16-wide MMA steps
 * ascending, partials added in split order. A row's result therefore does not
 * depend on how many rows share the call, so a text sent in pieces still gives
 * the same audio as the same text sent whole. 64x128 outputs per block, 8
 * warps of 32x32, the next slab prefetched into registers. */
static constexpr int TB_BM = 64, TB_BN = 128, TB_BK = 32, TB_LD = TB_BK + 8;
static constexpr int TB_THREADS = 256;
static constexpr int TB_SLD = 20; /* per-warp 16x16 fp32 staging, padded */

__global__ static void __launch_bounds__(TB_THREADS)
k_tile_gemm_bf16tc(const float *A, const uint16_t *W, const float *bias,
                   float *C, int M, int N, int K, int KC, float *P) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    __shared__ __align__(32) uint16_t As[TB_BM * TB_LD];
    __shared__ __align__(32) uint16_t Ws[TB_BN * TB_LD];
    __shared__ __align__(32) float St[8 * 16 * TB_SLD];
    const int m0 = (int)blockIdx.y * TB_BM;
    const int n0 = (int)blockIdx.x * TB_BN;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3; /* 2 x 4 warps of 32 x 32 */
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
    const int kbeg = (int)blockIdx.z * KC;
    const int kend = kbeg + KC < K ? kbeg + KC : K;
    /* A slab: 64 x 32 floats = 512 float4, 2 per thread.
     * W slab: 128 x 32 bf16 = 512 uint4 (8 bf16 each), 2 per thread. */
    float4 ra[2];
    uint4 rw[2];
    auto fetch = [&](int k0) {
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + TB_THREADS * i;
            const int ar = idx >> 3, ac = (idx & 7) * 4;
            const int gm = m0 + ar, ak = k0 + ac;
            ra[i] = gm < M && ak < kend
                ? *reinterpret_cast<const float4 *>(A + (size_t)gm * (size_t)K + (size_t)ak)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const int wr = idx >> 2, wc = (idx & 3) * 8;
            const int gn = n0 + wr, wk = k0 + wc;
            rw[i] = gn < N && wk < kend
                ? *reinterpret_cast<const uint4 *>(W + (size_t)gn * (size_t)K + (size_t)wk)
                : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    auto stash = [&]() {
        for (int i = 0; i < 2; ++i) {
            const int idx = tid + TB_THREADS * i;
            const int ar = idx >> 3, ac = (idx & 7) * 4;
            uint2 packed;
            packed.x = (uint32_t)cuda_bf16_from_float(ra[i].x) |
                       ((uint32_t)cuda_bf16_from_float(ra[i].y) << 16);
            packed.y = (uint32_t)cuda_bf16_from_float(ra[i].z) |
                       ((uint32_t)cuda_bf16_from_float(ra[i].w) << 16);
            *reinterpret_cast<uint2 *>(As + ar * TB_LD + ac) = packed;
            const int wr = idx >> 2, wc = (idx & 3) * 8;
            *reinterpret_cast<uint4 *>(Ws + wr * TB_LD + wc) = rw[i];
        }
    };
    if (kbeg < kend) fetch(kbeg);
    for (int k0 = kbeg; k0 < kend; k0 += TB_BK) {
        stash();
        __syncthreads();
        if (k0 + TB_BK < kend) fetch(k0 + TB_BK); /* overlaps the MMAs below */
        for (int kk = 0; kk < TB_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                           wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                           wmma::col_major> b[2];
            for (int i = 0; i < 2; ++i)
                wmma::load_matrix_sync(
                    a[i], reinterpret_cast<const __nv_bfloat16 *>(
                              As + (wm * 32 + i * 16) * TB_LD + kk), TB_LD);
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(
                    b[j], reinterpret_cast<const __nv_bfloat16 *>(
                              Ws + (wn * 32 + j * 16) * TB_LD + kk), TB_LD);
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    float *st = St + warp * 16 * TB_SLD;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            wmma::store_matrix_sync(st, acc[i][j], TB_SLD, wmma::mem_row_major);
            __syncwarp();
            const int rb = m0 + wm * 32 + i * 16, cb = n0 + wn * 32 + j * 16;
            for (int e = lane; e < 256; e += 32) {
                const int r = e >> 4, c = e & 15;
                const int gm = rb + r, gn = cb + c;
                if (gm >= M || gn >= N) continue;
                const size_t o = (size_t)gm * (size_t)N + (size_t)gn;
                const float v = st[r * TB_SLD + c];
                if (P != nullptr)
                    P[(size_t)blockIdx.z * (size_t)M * (size_t)N + o] = v;
                else
                    C[o] = v + (bias != nullptr ? bias[gn] : 0.0f);
            }
            __syncwarp();
        }
    }
#endif
}

/* MYNAH_CUDA_PREFILL_BF16TC: 0 = never; unset or empty = default on for
 * tiles whose weights are already bf16 (MYNAH_CUDA_QUANT=bf16, the Pocket
 * default), so MYNAH_CUDA_QUANT=f32 keeps the f32 weights in the prefill too;
 * any other value = also the fixed-order f32-weight prefill (the explicit
 * opt-in measured before bf16 weights became the default).  Needs sm_80+. */
static int cuda_prefill_bf16tc_level(void) {
    static const int level = [] {
        const char *v = std::getenv("MYNAH_CUDA_PREFILL_BF16TC");
        if (v != nullptr && std::strcmp(v, "0") == 0) return 0;
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   dev) != cudaSuccess || major < 8)
            return 0;
        return v == nullptr || *v == '\0' ? 1 : 2;
    }();
    return level;
}

static int cuda_tile_tc_level(void) {
    static const int level = [] {
        const char *v = std::getenv("MYNAH_CUDA_TILE_TC");
        if (v == nullptr || std::strcmp(v, "0") == 0) return 0;
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
            major < 8)
            return 0;
        return std::strcmp(v, "2") == 0 ? 2 : 1;
    }();
    return level;
}

static bool cuda_tile_tc_usable(void) {
    static const bool on = [] {
        if (!cuda_env_enabled("MYNAH_CUDA_TILE_TC", false)) return false;
        int dev = 0, major = 0;
        return cudaGetDevice(&dev) == cudaSuccess &&
               cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                      dev) == cudaSuccess && major >= 8;
    }();
    return on;
}

/* The BF16 tensor-core tile (k_tile_gemm_bf16tc) for one tile GEMM, when
 * MYNAH_CUDA_PREFILL_BF16TC allows it for these weights (`f32_weights`: the
 * engine resolved f32 for this stage) and the shape fits; returns 1 when it
 * does not (the caller takes another path), 0 on success, -1 on error. */
static int tile_gemm_bf16tc(cuda_backend_state *st, const float *A,
                            const float *hw, const float *db, float *C,
                            size_t M, size_t N, size_t K, bool f32_weights,
                            char *e, size_t ec) {
    const int level = cuda_prefill_bf16tc_level();
    if (K % (size_t)TB_BK != 0u || ((uintptr_t)A & 15u) != 0u || level == 0 ||
        (f32_weights && level < 2))
        return 1;
    static std::atomic<bool> announced{false};
    if (!announced.exchange(true))
        std::fprintf(stderr,
                     "mynah-tts: CUDA fixed-order prefill tile on bf16 "
                     "tensor cores (MYNAH_CUDA_PREFILL_BF16TC=1; fp32 "
                     "accumulate, batch-invariant)\n");
    cuda_tile_workspace &w = st->tile;
    uint16_t *dw16 = nullptr;
    if (cached_weight_bf16(st, hw, N * K, &dw16, e, ec)) return -1;
    int S = tile_gemm_splits(w.sms > 0 ? w.sms : 40, (int)N, (int)K);
    if (w.splitk == nullptr || (size_t)S * M * N > w.splitk_cap) S = 1;
    int KC = (int)((K + (size_t)S - 1u) / (size_t)S);
    KC = (KC + TB_BK - 1) / TB_BK * TB_BK;
    const dim3 tgrid((unsigned)((N + TB_BN - 1) / TB_BN),
                     (unsigned)((M + TB_BM - 1) / TB_BM), (unsigned)S);
    k_tile_gemm_bf16tc<<<tgrid, TB_THREADS, 0, st->stream>>>(
        A, dw16, db, C, (int)M, (int)N, (int)K, KC, S > 1 ? w.splitk : nullptr);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (S == 1) return 0;
    const dim3 g2((unsigned)((N + 255) / 256), (unsigned)M);
    k_tile_gemm_reduce<<<g2, 256, 0, st->stream>>>(w.splitk, S, (int)M,
                                                   (int)N, db, C);
    return ce(cudaGetLastError(), e, ec) ? -1 : 0;
}

static int tile_gemm(cuda_backend_state *st, const float *A, const float *hw,
                     const float *hb, float *C, size_t M, size_t N, size_t K,
                     int bf16, char *e, size_t ec) {
    float *db = nullptr;
    if (hb != nullptr && cached_weight(st, hb, N * sizeof(float), &db, e, ec))
        return -1;
    cuda_tile_workspace &w = st->tile;
    const dim3 grid((unsigned)((N + TILE_GEMM_BN - 1) / TILE_GEMM_BN),
                    (unsigned)((M + TILE_GEMM_BM - 1) / TILE_GEMM_BM));
    if (bf16) {
        /* MYNAH_CUDA_QUANT=bf16: the resident BF16 weight copy through the
         * ordered fp32-accumulating tile kernel, whichever tile mode the f32
         * path runs.  A cuBLAS BF16 tile was measured on the L4: its
         * M-dependent rounding put the Mimi solo-vs-gang gap at 1.1e-3,
         * outside the tensor-core parity band the self-check allows; this
         * kernel keeps the tile batch-invariant at bf16 weight traffic.
         * MYNAH_CUDA_PREFILL_BF16TC: the same invariance on tensor cores,
         * taken whenever the call asks for a fixed order, or when there is no
         * cuBLAS bf16 path for it; a call that does not (the Mimi tile,
         * MYNAH_CUDA_PREFILL_FIXED=0) keeps cuBLAS below. */
        const bool cublas_bf16 = st->tile_cublas && !w.fixed_order &&
                                 w.a16 != nullptr && M * K <= w.a16_cap;
        if (!cublas_bf16) {
            const int tc = tile_gemm_bf16tc(st, A, hw, db, C, M, N, K, false,
                                            e, ec);
            if (tc <= 0) return tc;
        }
        uint16_t *dw16 = nullptr;
        if (cached_weight_bf16(st, hw, N * K, &dw16, e, ec)) return -1;
        if (cublas_bf16) {
            /* Order not fixed: the same bf16 tensor-core GEMM as the decode
             * step's bf16 Linears (activation rounded to bf16, fp32
             * accumulate), instead of the SIMT kernel. */
            k_f32_to_bf16<<<((int)(M * K) + 255) / 256, 256, 0, st->stream>>>(
                A, w.a16, (int)(M * K));
            if (ce(cudaGetLastError(), e, ec) ||
                cbe(cublasSetStream(st->cublas, st->stream), e, ec))
                return -1;
            const float alpha = 1.0f, beta = 0.0f;
            if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N,
                                 (int)M, (int)K, &alpha, dw16, CUDA_R_16BF,
                                 (int)K, w.a16, CUDA_R_16BF, (int)K, &beta, C,
                                 CUDA_R_32F, (int)N, CUBLAS_COMPUTE_32F,
                                 CUBLAS_GEMM_DEFAULT), e, ec))
                return -1;
            if (db == nullptr) return 0;
            const int total = (int)(M * N);
            k_bias_rows<<<(total + 255) / 256, 256, 0, st->stream>>>(C, db, (int)M,
                                                                     (int)N);
            return ce(cudaGetLastError(), e, ec);
        }
        k_tile_gemm<true><<<grid, 256, 0, st->stream>>>(A, dw16, db, C, (int)M,
                                                          (int)N, (int)K);
        return ce(cudaGetLastError(), e, ec);
    }
    if (w.fixed_order) {
        /* f32 weights: only an explicit MYNAH_CUDA_PREFILL_BF16TC=1. */
        const int tc = tile_gemm_bf16tc(st, A, hw, db, C, M, N, K, true, e,
                                        ec);
        if (tc <= 0) return tc;
    }
    float *dw = nullptr;
    if (cached_weight(st, hw, N * K * sizeof(float), &dw, e, ec)) return -1;
    if (w.fixed_order && cuda_tile_tc_usable()) {
        int S = tile_gemm_splits(w.sms > 0 ? w.sms : 40, (int)N, (int)K);
        if (w.splitk == nullptr || (size_t)S * M * N > w.splitk_cap) S = 1;
        int KC = (int)((K + (size_t)S - 1u) / (size_t)S);
        KC = (KC + TILE_TC_BK - 1) / TILE_TC_BK * TILE_TC_BK;
        if (cuda_tile_tc_level() == 2) {
            const dim3 tgrid((unsigned)((N + TC2_BN - 1) / TC2_BN),
                             (unsigned)((M + TC2_BM - 1) / TC2_BM), (unsigned)S);
            k_tile_gemm_tc2<<<tgrid, TC2_THREADS, 0, st->stream>>>(
                A, dw, db, C, (int)M, (int)N, (int)K, KC, S > 1 ? w.splitk : nullptr);
        } else {
            const dim3 tgrid((unsigned)((N + TILE_TC_BN - 1) / TILE_TC_BN),
                             (unsigned)((M + TILE_TC_BM - 1) / TILE_TC_BM),
                             (unsigned)S);
            k_tile_gemm_tc<<<tgrid, 128, 0, st->stream>>>(
                A, dw, db, C, (int)M, (int)N, (int)K, KC, S > 1 ? w.splitk : nullptr);
        }
        if (ce(cudaGetLastError(), e, ec)) return -1;
        if (S == 1) return 0;
        const dim3 g2((unsigned)((N + 255) / 256), (unsigned)M);
        k_tile_gemm_reduce<<<g2, 256, 0, st->stream>>>(w.splitk, S, (int)M,
                                                       (int)N, db, C);
        return ce(cudaGetLastError(), e, ec);
    }
    if (st->tile_cublas && !w.fixed_order) {
        /* Tensor-core GEMM through cuBLAS: faster, but cuBLAS picks its
         * algorithm by M, so a row's bits can depend on its batch. */
        if (cbe(cublasSetStream(st->cublas, st->stream), e, ec)) return -1;
        const float alpha = 1.0f, beta = 0.0f;
        if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)N,
                             (int)M, (int)K, &alpha, dw, CUDA_R_32F, (int)K, A,
                             CUDA_R_32F, (int)K, &beta, C, CUDA_R_32F, (int)N,
                             cuda_compute_type(st), cuda_gemm_algo(st)),
                e, ec))
            return -1;
        if (db == nullptr) return 0;
        const int total = (int)(M * N);
        k_bias_rows<<<(total + 255) / 256, 256, 0, st->stream>>>(C, db, (int)M,
                                                                 (int)N);
        return ce(cudaGetLastError(), e, ec);
    }
    static const bool splitk_on = cuda_env_enabled("MYNAH_CUDA_TILE_SPLITK", true);
    const int S = tile_gemm_splits(w.sms > 0 ? w.sms : 40, (int)N, (int)K);
    if (splitk_on && S > 1 && w.splitk != nullptr && (size_t)S * M * N <= w.splitk_cap) {
        int KC = (int)((K + (size_t)S - 1u) / (size_t)S);
        KC = (KC + TILE_GEMM_BK - 1) / TILE_GEMM_BK * TILE_GEMM_BK;
        const dim3 grid((unsigned)((N + TILE_GEMM_BN - 1) / TILE_GEMM_BN),
                        (unsigned)((M + TILE_GEMM_BM - 1) / TILE_GEMM_BM),
                        (unsigned)S);
        k_tile_gemm_part<<<grid, 256, 0, st->stream>>>(A, dw, w.splitk, (int)M,
                                                       (int)N, (int)K, KC);
        if (ce(cudaGetLastError(), e, ec)) return -1;
        const dim3 g2((unsigned)((N + 255) / 256), (unsigned)M);
        k_tile_gemm_reduce<<<g2, 256, 0, st->stream>>>(w.splitk, S, (int)M,
                                                       (int)N, db, C);
        return ce(cudaGetLastError(), e, ec);
    }
    k_tile_gemm<false><<<grid, 256, 0, st->stream>>>(A, dw, db, C, (int)M,
                                                       (int)N, (int)K);
    return ce(cudaGetLastError(), e, ec);
}

static int tile_norm(cuda_backend_state *st, const float *in, float *out,
                     const float *gain, const float *bias, size_t M, size_t dim,
                     float eps, char *e, size_t ec) {
    float *dg = nullptr, *db = nullptr;
    if (cached_weight(st, gain, dim * sizeof(float), &dg, e, ec)) return -1;
    if (bias != nullptr && cached_weight(st, bias, dim * sizeof(float), &db, e, ec))
        return -1;
    k_layer_norm<<<(int)M, 256, 0, st->stream>>>(out, in, dg, db, (int)dim, eps,
                                                (int)M);
    return ce(cudaGetLastError(), e, ec);
}

static int tile_residual(cuda_backend_state *st, float *x, const float *y,
                         const float *scale, size_t M, size_t dim, char *e,
                         size_t ec) {
    const int total = (int)(M * dim);
    if (scale == nullptr) {
        k_residual_add<<<(total + 255) / 256, 256, 0, st->stream>>>(x, y, total);
    } else {
        float *ds = nullptr;
        if (cached_weight(st, scale, dim * sizeof(float), &ds, e, ec)) return -1;
        k_scaled_residual_rows_add<<<(total + 255) / 256, 256, 0, st->stream>>>(
            x, y, ds, (int)M, (int)dim);
    }
    return ce(cudaGetLastError(), e, ec);
}

/* cuBLAS chooses its algorithm (and split-K) by M and the tensor-core
 * modes round inputs, so a row's bits then depend on its batch mates.  The
 * same holds for the reduced-precision weight paths (MYNAH_CUDA_QUANT=bf16
 * runs cuBLAS BF16 GEMMs; int8 dequantizes through its own tensor-core
 * kernels), whichever tile mode is set, and for the BF16 SEANet decoder
 * (MYNAH_CUDA_SEANET_BF16: single vs batched BF16 GEMMs). */
extern "C" int mynah_cuda_batch_invariant(void *opaque) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr) return 1;
    return !(st->fast_math || st->tf32 || st->tile_cublas || st->quant_weights ||
             cuda_seanet_bf16_enabled());
}

extern "C" int mynah_cuda_tile_transformer_dev(void *opaque,
                                               const mynah_backend_tile_desc *d,
                                               char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || d == nullptr || d->layer == nullptr ||
        d->input == nullptr || d->kv == nullptr || d->start == nullptr ||
        d->rows == 0u || d->positions == 0u || d->dim == 0u || d->heads == 0u ||
        d->ffn == 0u || d->layers == 0u || d->dim % d->heads != 0u ||
        (d->dim / d->heads) % 2u != 0u ||
        d->rows > 4096u || d->positions > 4096u || d->layers > 64u ||
        d->dim > 65536u || d->ffn > 65536u) {
        set_error(e, ec, "invalid CUDA tile transformer description");
        return -1;
    }
    /* Pack the rows and bound every attention window before touching the GPU. */
    size_t M = 0u;
    size_t window = 0u;
    /* Rows that do not store a leading prefix (desc.skip). The PREFIX kernel
     * variants run only when at least one row has one, so a call without
     * skips launches exactly the kernels it always did. */
    bool prefixed = false;
    for (size_t r = 0; r < d->rows; ++r) {
        const size_t n = d->count != nullptr ? d->count[r] : d->positions;
        if (n > d->positions) {
            set_error(e, ec, "CUDA tile row count exceeds its positions");
            return -1;
        }
        const size_t end = d->start[r] + n;
        const size_t ring = d->rings != nullptr ? d->rings[r] : d->ring;
        const size_t skip = d->skip != nullptr ? d->skip[r] : 0u;
        if (ring == 0u ||
            (d->context != 0u && ring < d->context + d->positions - 1u)) {
            set_error(e, ec, "CUDA tile ring cannot hold the attention window");
            return -1;
        }
        if (skip != 0u) {
            /* The prefix is read-only and the row holds [skip, end): a start
             * below the skip would write into the shared planes' place, and a
             * windowed (ring) cache would wrap into it. */
            if (d->prefix == nullptr || d->context != 0u || d->start[r] < skip ||
                skip > (size_t)LLONG_MAX) {
                set_error(e, ec, "invalid CUDA tile shared prefix");
                return -1;
            }
            for (size_t l = 0; l < d->layers; ++l) {
                if (d->prefix[r * d->layers + l] == nullptr) {
                    set_error(e, ec, "invalid CUDA tile shared prefix");
                    return -1;
                }
            }
            prefixed = true;
        }
        if (d->context == 0u && end - skip > ring) {
            set_error(e, ec, "CUDA tile position exceeds the cache");
            return -1;
        }
        if (d->kv_int8 &&
            (d->kv_strides == nullptr || d->heads > 32u ||
             d->kv_strides[2u * r] < d->dim + 4u * d->heads ||
             d->kv_strides[2u * r] % 4u != 0u ||
             d->kv_strides[2u * r + 1u] % 4u != 0u)) {
            set_error(e, ec, "invalid CUDA tile int8 KV layout");
            return -1;
        }
        if (d->kv_strides != nullptr) {
            /* MYNAH_CUDA_KV_VMM: slot s of K at s * pitch, of V at voff +
             * s * pitch; the two must not overlap within one slot. */
            const size_t pitch = d->kv_strides[2u * r];
            const size_t voff = d->kv_strides[2u * r + 1u];
            if (pitch < d->dim || pitch > (size_t)LLONG_MAX ||
                voff > (size_t)LLONG_MAX ||
                !((voff >= d->dim && voff <= pitch - d->dim) ||
                  (ring <= (size_t)LLONG_MAX / pitch && voff >= ring * pitch))) {
                set_error(e, ec, "invalid CUDA tile cache stride");
                return -1;
            }
        }
        const size_t w = d->context != 0u && end > d->context ? d->context : end;
        if (w > window) window = w;
        M += n;
    }
    if (M == 0u) return 0;
    if (M > (size_t)INT_MAX / d->ffn || window > 12288u) {
        set_error(e, ec, "CUDA tile is too large (rows or attention window)");
        return -1;
    }
    const size_t kv_count = d->rows * d->layers;
    /* Attention groups: up to TILE_ATTN_Q consecutive tokens of one request. */
    size_t group_count = 0u;
    for (size_t r = 0; r < d->rows; ++r) {
        const size_t n = d->count != nullptr ? d->count[r] : d->positions;
        group_count += (n + TILE_ATTN_Q - 1u) / TILE_ATTN_Q;
    }
    /* With prefixed rows the staging also carries the prefix pointer table
     * (after the kv table) and the per-row skip (after the rings); without,
     * the layout is the one it always was. */
    const size_t prefix_count = prefixed ? kv_count : 0u;
    const size_t skip_count = prefixed ? d->rows : 0u;
    /* MYNAH_CUDA_KV_VMM: per-row (pitch, V offset) pairs after the skips;
     * absent (and the layout unchanged) without kv_strides. */
    const size_t stride_count = d->kv_strides != nullptr ? 2u * d->rows : 0u;
    const size_t meta_bytes = (2u * d->rows + kv_count + prefix_count) * sizeof(void *) +
                              (2u * d->rows + skip_count + stride_count) *
                                  sizeof(long long) +
                              M * sizeof(int2) + group_count * sizeof(int2);
    if (tile_reserve(st, M, d->dim, d->ffn, meta_bytes, e, ec)) return -1;
    cuda_tile_workspace &w = st->tile;

    /* The pinned staging is reused per call: wait until the previous call's
     * copy has consumed it before overwriting. */
    if (ce(cudaEventSynchronize(w.meta_event), e, ec)) return -1;
    char *host = static_cast<char *>(w.meta_host);
    const size_t off_out = d->rows * sizeof(void *);
    const size_t off_kv = 2u * d->rows * sizeof(void *);
    const size_t off_prefix = off_kv + kv_count * sizeof(void *);
    const size_t off_start = off_prefix + prefix_count * sizeof(void *);
    const size_t off_ring = off_start + d->rows * sizeof(long long);
    const size_t off_skip = off_ring + d->rows * sizeof(long long);
    const size_t off_stride = off_skip + skip_count * sizeof(long long);
    const size_t off_map = off_stride + stride_count * sizeof(long long);
    memcpy(host, d->input, d->rows * sizeof(void *));
    if (d->output != nullptr)
        memcpy(host + off_out, d->output, d->rows * sizeof(void *));
    else
        memset(host + off_out, 0, d->rows * sizeof(void *));
    memcpy(host + off_kv, d->kv, kv_count * sizeof(void *));
    if (prefixed) {
        /* Rows without a skip get a null entry; the kernels never read it. */
        void **host_prefix = reinterpret_cast<void **>(host + off_prefix);
        long long *host_skip = reinterpret_cast<long long *>(host + off_skip);
        for (size_t r = 0; r < d->rows; ++r) {
            host_skip[r] = (long long)d->skip[r];
            for (size_t l = 0; l < d->layers; ++l)
                host_prefix[r * d->layers + l] =
                    d->skip[r] != 0u ? d->prefix[r * d->layers + l] : nullptr;
        }
    }
    if (stride_count != 0u) {
        long long *host_stride = reinterpret_cast<long long *>(host + off_stride);
        for (size_t i = 0; i < stride_count; ++i)
            host_stride[i] = (long long)d->kv_strides[i];
    }
    long long *host_start = reinterpret_cast<long long *>(host + off_start);
    long long *host_ring = reinterpret_cast<long long *>(host + off_ring);
    int2 *host_map = reinterpret_cast<int2 *>(host + off_map);
    const size_t off_groups = off_map + M * sizeof(int2);
    int2 *host_groups = reinterpret_cast<int2 *>(host + off_groups);
    size_t m = 0u, g = 0u;
    for (size_t r = 0; r < d->rows; ++r) {
        host_start[r] = (long long)d->start[r];
        host_ring[r] = (long long)(d->rings != nullptr ? d->rings[r] : d->ring);
        const size_t n = d->count != nullptr ? d->count[r] : d->positions;
        for (size_t t = 0; t < n; t += TILE_ATTN_Q)
            host_groups[g++] = make_int2((int)(m + t),
                                         (int)(n - t < TILE_ATTN_Q ? n - t : TILE_ATTN_Q));
        for (size_t t = 0; t < n; ++t) host_map[m++] = make_int2((int)r, (int)t);
    }
    if (ce(cudaMemcpyAsync(w.meta_dev, w.meta_host, meta_bytes,
                           cudaMemcpyHostToDevice, st->stream), e, ec) ||
        ce(cudaEventRecord(w.meta_event, st->stream), e, ec))
        return -1;
    char *dev = static_cast<char *>(w.meta_dev);
    const float *const *d_in = reinterpret_cast<const float *const *>(dev);
    float *const *d_out = reinterpret_cast<float *const *>(dev + off_out);
    void *const *d_kv = reinterpret_cast<void *const *>(dev + off_kv);
    const long long *d_start = reinterpret_cast<const long long *>(dev + off_start);
    const long long *d_ring = reinterpret_cast<const long long *>(dev + off_ring);
    void *const *d_prefix =
        prefixed ? reinterpret_cast<void *const *>(dev + off_prefix) : nullptr;
    const long long *d_skip =
        prefixed ? reinterpret_cast<const long long *>(dev + off_skip) : nullptr;
    const long long *d_stride = stride_count != 0u
        ? reinterpret_cast<const long long *>(dev + off_stride) : nullptr;
    const int2 *d_map = reinterpret_cast<const int2 *>(dev + off_map);
    const int2 *d_groups = reinterpret_cast<const int2 *>(dev + off_groups);
    static const bool grouped = cuda_env_enabled("MYNAH_CUDA_TILE_ATTN_GROUPED", true);
    const bool use_grouped = grouped && (d->dim / d->heads) <= TILE_ATTN_HW;
    const dim3 group_grid((unsigned)group_count, (unsigned)d->heads);

    const int dim = (int)d->dim;
    const int heads = (int)d->heads;
    const int head_width = dim / heads;
    const int total = (int)(M * d->dim);
    w.fixed_order = d->fixed_order != 0;
    k_tile_gather<<<(total + 255) / 256, 256, 0, st->stream>>>(d_in, d_map, w.x,
                                                               dim, total);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    const float scale = 1.0f / sqrtf((float)head_width);
    int warps_per_block = (int)((48u * 1024u) / (window * sizeof(float)));
    if (warps_per_block > 4) warps_per_block = 4;
    if (warps_per_block < 1) warps_per_block = 1;
    const int work = (int)M * heads;
    const size_t smem = (size_t)warps_per_block * window * sizeof(float);
    for (size_t l = 0; l < d->layers; ++l) {
        const mynah_transformer_tile_layer *L = &d->layer[l];
        if (tile_norm(st, w.x, w.xn, L->norm1_weight, L->norm1_bias, M, d->dim,
                      d->layernorm_eps, e, ec) ||
            tile_gemm(st, w.xn, L->in_proj_weight, L->in_proj_bias, w.qkv, M,
                      3u * d->dim, d->dim, d->weight_bf16, e, ec))
            return -1;
        const unsigned blocks = (unsigned)((work + warps_per_block - 1) /
                                           warps_per_block);
        if (d->kv_int8) {
            if (prefixed)
                tile_attention_layer<int8_t, true>(
                    st, w.qkv, d_kv, d_start, d_map, d_groups, (int)l,
                    (int)d->layers, heads, head_width, (long long)d->context,
                    d_ring, d->max_period, M, use_grouped, group_grid, blocks,
                    warps_per_block, smem, (int)window, scale, w.att, d_skip,
                    d_prefix, d_stride);
            else
                tile_attention_layer<int8_t, false>(
                    st, w.qkv, d_kv, d_start, d_map, d_groups, (int)l,
                    (int)d->layers, heads, head_width, (long long)d->context,
                    d_ring, d->max_period, M, use_grouped, group_grid, blocks,
                    warps_per_block, smem, (int)window, scale, w.att, nullptr,
                    nullptr, d_stride);
        } else if (d->kv_bf16) {
            if (prefixed)
                tile_attention_layer<uint16_t, true>(
                    st, w.qkv, d_kv, d_start, d_map, d_groups, (int)l,
                    (int)d->layers, heads, head_width, (long long)d->context,
                    d_ring, d->max_period, M, use_grouped, group_grid, blocks,
                    warps_per_block, smem, (int)window, scale, w.att, d_skip,
                    d_prefix, d_stride);
            else
                tile_attention_layer<uint16_t, false>(
                    st, w.qkv, d_kv, d_start, d_map, d_groups, (int)l,
                    (int)d->layers, heads, head_width, (long long)d->context,
                    d_ring, d->max_period, M, use_grouped, group_grid, blocks,
                    warps_per_block, smem, (int)window, scale, w.att, nullptr,
                    nullptr, d_stride);
        } else {
            if (prefixed)
                tile_attention_layer<float, true>(
                    st, w.qkv, d_kv, d_start, d_map, d_groups, (int)l,
                    (int)d->layers, heads, head_width, (long long)d->context,
                    d_ring, d->max_period, M, use_grouped, group_grid, blocks,
                    warps_per_block, smem, (int)window, scale, w.att, d_skip,
                    d_prefix, d_stride);
            else
                tile_attention_layer<float, false>(
                    st, w.qkv, d_kv, d_start, d_map, d_groups, (int)l,
                    (int)d->layers, heads, head_width, (long long)d->context,
                    d_ring, d->max_period, M, use_grouped, group_grid, blocks,
                    warps_per_block, smem, (int)window, scale, w.att, nullptr,
                    nullptr, d_stride);
        }
        if (ce(cudaGetLastError(), e, ec)) return -1;
        if (tile_gemm(st, w.att, L->out_proj_weight, L->out_proj_bias, w.proj, M,
                      d->dim, d->dim, d->weight_bf16, e, ec) ||
            tile_residual(st, w.x, w.proj, L->layer_scale_1, M, d->dim, e, ec) ||
            tile_norm(st, w.x, w.xn, L->norm2_weight, L->norm2_bias, M, d->dim,
                      d->layernorm_eps, e, ec) ||
            tile_gemm(st, w.xn, L->linear1_weight, L->linear1_bias, w.ffn_buf, M,
                      d->ffn, d->dim, d->weight_bf16, e, ec))
            return -1;
        const int ffn_total = (int)(M * d->ffn);
        k_gelu<<<(ffn_total + 255) / 256, 256, 0, st->stream>>>(w.ffn_buf,
                                                                 ffn_total);
        if (ce(cudaGetLastError(), e, ec) ||
            tile_gemm(st, w.ffn_buf, L->linear2_weight, L->linear2_bias, w.proj,
                      M, d->dim, d->ffn, d->weight_bf16, e, ec) ||
            tile_residual(st, w.x, w.proj, L->layer_scale_2, M, d->dim, e, ec))
            return -1;
    }
    if (d->output != nullptr) {
        k_tile_scatter<<<(total + 255) / 256, 256, 0, st->stream>>>(
            w.x, d_out, d_map, (int)d->positions, dim, total);
        if (ce(cudaGetLastError(), e, ec)) return -1;
    }
    st->tile_calls.fetch_add(1ull, std::memory_order_relaxed);
    st->tile_rows.fetch_add((unsigned long long)M, std::memory_order_relaxed);
    return 0;
}

extern "C" int mynah_cuda_self_attention_bf16_dev(
    void *opaque, const float *qkv, void *kcache, void *vcache,
    size_t position, size_t cache_stride, size_t valid, size_t heads,
    size_t head_width, float scale, float *out, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st->kv_int8) {
        set_error(e, ec, "CUDA int8 KV: single-row attention goes through the "
                         "batched entry point");
        return -1;
    }
    if (qkv == nullptr || kcache == nullptr || vcache == nullptr || out == nullptr ||
        heads == 0u || head_width == 0u || valid == 0u || heads > (size_t)INT_MAX ||
        head_width > (size_t)INT_MAX || cache_stride == 0u || position >= valid ||
        heads > SIZE_MAX / head_width) {
        set_error(e, ec, "invalid CUDA BF16 self-attention dimensions");
        return -1;
    }
    const size_t width = heads * head_width;
    if (cache_stride < width ||
        (valid - 1u) > (SIZE_MAX - (width - 1u)) / cache_stride) {
        set_error(e, ec, "CUDA BF16 self-attention cache stride overflow");
        return -1;
    }
    k_self_attention_bf16<false><<<(int)heads, attention_threads(head_width), 0,
                                   st->stream>>>(
        qkv, static_cast<uint16_t *>(kcache), static_cast<uint16_t *>(vcache),
        position, cache_stride, valid, (int)heads, (int)head_width, scale, out,
        nullptr, nullptr, 0u);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_self_attention_bf16_prefix_dev(
    void *opaque, const float *qkv, void *kcache, void *vcache,
    const void *kprefix, const void *vprefix, size_t prefix_len,
    size_t position, size_t cache_stride, size_t valid, size_t heads,
    size_t head_width, float scale, float *out, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st->kv_int8) {
        set_error(e, ec, "CUDA int8 KV: single-row attention goes through the "
                         "batched entry point");
        return -1;
    }
    if (qkv == nullptr || kcache == nullptr || vcache == nullptr || out == nullptr ||
        heads == 0u || head_width == 0u || valid == 0u || heads > (size_t)INT_MAX ||
        head_width > (size_t)INT_MAX || cache_stride == 0u || position >= valid ||
        heads > SIZE_MAX / head_width) {
        set_error(e, ec, "invalid CUDA BF16 self-attention dimensions");
        return -1;
    }
    if (prefix_len > position ||
        (prefix_len != 0u && (kprefix == nullptr || vprefix == nullptr))) {
        set_error(e, ec, "invalid CUDA BF16 shared voice prefix");
        return -1;
    }
    const size_t width = heads * head_width;
    if (cache_stride < width ||
        (valid - 1u) > (SIZE_MAX - (width - 1u)) / cache_stride) {
        set_error(e, ec, "CUDA BF16 self-attention cache stride overflow");
        return -1;
    }
    k_self_attention_bf16<true><<<(int)heads, attention_threads(head_width), 0,
                                  st->stream>>>(
        qkv, static_cast<uint16_t *>(kcache), static_cast<uint16_t *>(vcache),
        position, cache_stride, valid, (int)heads, (int)head_width, scale, out,
        static_cast<const uint16_t *>(kprefix),
        static_cast<const uint16_t *>(vprefix), prefix_len);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_cross_attention_dev(
    void *opaque, const float *q, const float *kcache, const float *vcache,
    size_t valid, size_t cache_stride, size_t heads, size_t head_width,
    float scale, float *out, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (q == nullptr || kcache == nullptr || vcache == nullptr || out == nullptr ||
        heads == 0 || head_width == 0 || valid == 0 || heads > (size_t)INT_MAX ||
        head_width > (size_t)INT_MAX || cache_stride == 0) {
        set_error(e, ec, "invalid CUDA cross-attention dimensions");
        return -1;
    }
    if (heads > SIZE_MAX / head_width) {
        set_error(e, ec, "CUDA cross-attention width overflow");
        return -1;
    }
    const size_t width = heads * head_width;
    if (cache_stride < width ||
        (valid - 1u) > (SIZE_MAX - (width - 1u)) / cache_stride) {
        set_error(e, ec, "CUDA cross-attention cache stride overflow");
        return -1;
    }
    k_cross_attention<<<(int)heads, attention_threads(head_width), 0, st->stream>>>(
        q, kcache, vcache, valid, cache_stride, (int)heads,
        (int)head_width, scale, out);
    return ce(cudaGetLastError(), e, ec);
}

__global__ static void k_self_attention_batch(
    const float *qkv, float *const *kcache, float *const *vcache,
    const size_t *positions, const size_t *cache_strides, int batch, int heads,
    int head_width, float scale, float *out) {
    const int head = (int)blockIdx.x;
    const int request = (int)blockIdx.y;
    if (head >= heads || request >= batch) return;
    const int tid = (int)threadIdx.x;
    const size_t width = (size_t)heads * (size_t)head_width;
    const size_t hbase = (size_t)head * (size_t)head_width;
    const size_t position = positions[request];
    const size_t cache_stride = cache_strides[request];
    const size_t cache_base = position * cache_stride + hbase;
    float *request_k = kcache[request];
    float *request_v = vcache[request];
    const float *request_qkv = qkv + (size_t)request * width * 3u;
    const float *q = request_qkv + hbase;
    const float *k = request_qkv + width + hbase;
    const float *v = request_qkv + width * 2u + hbase;
    for (int d = tid; d < head_width; d += (int)blockDim.x) {
        request_k[cache_base + (size_t)d] = k[d];
        request_v[cache_base + (size_t)d] = v[d];
        out[(size_t)request * width + hbase + (size_t)d] = 0.0f;
    }
    __syncthreads();
    __shared__ float partial[256];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float correction;
    __shared__ float probability;
    if (tid == 0) {
        maximum = -1.0e30f;
        denominator = 0.0f;
    }
    __syncthreads();
    for (size_t s = 0; s <= position; ++s) {
        const float *ks = request_k + s * cache_stride + hbase;
        const float *vs = request_v + s * cache_stride + hbase;
        float local = 0.0f;
        for (int d = tid; d < head_width; d += (int)blockDim.x)
            local += q[d] * ks[d];
        partial[tid] = local;
        __syncthreads();
        for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) partial[tid] += partial[tid + offset];
            __syncthreads();
        }
        if (tid == 0) {
            const float score = partial[0] * scale;
            const float next = fmaxf(maximum, score);
            correction = expf(maximum - next);
            probability = expf(score - next);
            denominator = denominator * correction + probability;
            maximum = next;
        }
        __syncthreads();
        for (int d = tid; d < head_width; d += (int)blockDim.x) {
            const size_t index = (size_t)request * width + hbase + (size_t)d;
            out[index] = out[index] * correction + probability * vs[d];
        }
        __syncthreads();
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = tid; d < head_width; d += (int)blockDim.x)
        out[(size_t)request * width + hbase + (size_t)d] *= inv;
}

/* SHARED: positions [0, prefix_len[request]) come from the shared voice
 * prefix planes (stride `width`), as in k_self_attention_bf16_batch_fast; the
 * row's own cache is then never touched below the prefix, so a row that does
 * not store its prefix (MYNAH_CUDA_SHARED_VOICE) is correct here too. */
template <bool SHARED>
__global__ static void k_self_attention_bf16_batch(
    const float *qkv, const uint16_t *const *kcache,
    const uint16_t *const *vcache, const size_t *positions,
    const size_t *cache_strides, int batch, int heads, int head_width,
    float scale, float *out, const uint16_t *const *kprefix,
    const uint16_t *const *vprefix, const size_t *prefix_len) {
    const int head = (int)blockIdx.x;
    const int request = (int)blockIdx.y;
    if (head >= heads || request >= batch) return;
    const int tid = (int)threadIdx.x;
    const size_t width = (size_t)heads * (size_t)head_width;
    const size_t hbase = (size_t)head * (size_t)head_width;
    const size_t position = positions[request];
    const size_t cache_stride = cache_strides[request];
    const size_t cache_base = position * cache_stride + hbase;
    uint16_t *request_k = const_cast<uint16_t *>(kcache[request]);
    uint16_t *request_v = const_cast<uint16_t *>(vcache[request]);
    const float *request_qkv = qkv + (size_t)request * width * 3u;
    const float *q = request_qkv + hbase;
    const float *k = request_qkv + width + hbase;
    const float *v = request_qkv + width * 2u + hbase;
    for (int d = tid; d < head_width; d += (int)blockDim.x) {
        request_k[cache_base + (size_t)d] = cuda_bf16_from_float(k[d]);
        request_v[cache_base + (size_t)d] = cuda_bf16_from_float(v[d]);
        out[(size_t)request * width + hbase + (size_t)d] = 0.0f;
    }
    __syncthreads();
    __shared__ float partial[256];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float correction;
    __shared__ float probability;
    if (tid == 0) {
        maximum = -1.0e30f;
        denominator = 0.0f;
    }
    __syncthreads();
    const size_t shared_len = SHARED ? prefix_len[request] : 0u;
    for (size_t s = 0; s <= position; ++s) {
        const bool from_prefix = SHARED && s < shared_len;
        const uint16_t *ks = from_prefix
            ? kprefix[request] + s * width + hbase
            : kcache[request] + s * cache_stride + hbase;
        const uint16_t *vs = from_prefix
            ? vprefix[request] + s * width + hbase
            : vcache[request] + s * cache_stride + hbase;
        float local = 0.0f;
        for (int d = tid; d < head_width; d += (int)blockDim.x)
            local += q[d] * cuda_bf16_to_float(ks[d]);
        partial[tid] = local;
        __syncthreads();
        for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) partial[tid] += partial[tid + offset];
            __syncthreads();
        }
        if (tid == 0) {
            const float score = partial[0] * scale;
            const float next = fmaxf(maximum, score);
            correction = expf(maximum - next);
            probability = expf(score - next);
            denominator = denominator * correction + probability;
            maximum = next;
        }
        __syncthreads();
        for (int d = tid; d < head_width; d += (int)blockDim.x) {
            const size_t index = (size_t)request * width + hbase + (size_t)d;
            out[index] = out[index] * correction + probability *
                         cuda_bf16_to_float(vs[d]);
        }
        __syncthreads();
    }
    const float inv = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    for (int d = tid; d < head_width; d += (int)blockDim.x)
        out[(size_t)request * width + hbase + (size_t)d] *= inv;
}

/* Decode attention over a BF16 KV cache, one block per (head, row).  The
 * legacy kernel above walks the context one position at a time with a block
 * reduction per position; this one scores a chunk of 128 positions at once
 * (one thread per position, 128-bit K loads), keeps an online softmax per
 * chunk and accumulates V in registers.  The reduction order depends only on
 * the row's own length, so the result is independent of the batch.  Needs
 * head_width dividing 128 and a multiple of 8, 16-byte aligned rows. */
static constexpr int CUDA_ATTN_FAST_THREADS = 128;

__device__ static float cuda_warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ static float cuda_warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

/* SHARED: positions [0, prefix_len[request]) are read from the request's
 * voice-prefix planes (kprefix/vprefix, stride `width`) instead of its own
 * cache. The values are the same bf16 numbers the prefill copies into the
 * row, so the result is bit-identical; rows of one voice read one copy, which
 * stays in L2 instead of streaming once per row from DRAM. */
/* BF16OUT (MYNAH_CUDA_BF16_FUSE): the output row is stored as the RNE BF16
 * rounding of the same FP32 value into `out_bf16` (the staged activation of
 * the out_proj GEMM) instead of FP32 into `out`. */
template <bool SHARED, bool BF16OUT>
__global__ static void __launch_bounds__(CUDA_ATTN_FAST_THREADS)
k_self_attention_bf16_batch_fast(
    const float *qkv, const uint16_t *const *kcache,
    const uint16_t *const *vcache, const size_t *positions,
    const size_t *cache_strides, int batch, int heads, int head_width,
    float scale, float *out, const uint16_t *const *kprefix,
    const uint16_t *const *vprefix, const size_t *prefix_len,
    uint16_t *out_bf16) {
    const int head = (int)blockIdx.x;
    const int request = (int)blockIdx.y;
    if (head >= heads || request >= batch) return;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const size_t width = (size_t)heads * (size_t)head_width;
    const size_t hbase = (size_t)head * (size_t)head_width;
    const size_t position = positions[request];
    const size_t stride = cache_strides[request];
    uint16_t *request_k = const_cast<uint16_t *>(kcache[request]);
    uint16_t *request_v = const_cast<uint16_t *>(vcache[request]);
    const size_t shared_len = SHARED ? prefix_len[request] : 0u;
    const uint16_t *shared_k = SHARED ? kprefix[request] : nullptr;
    const uint16_t *shared_v = SHARED ? vprefix[request] : nullptr;
    const float *request_qkv = qkv + (size_t)request * width * 3u;
    __shared__ float qs[CUDA_ATTN_FAST_THREADS];
    __shared__ float probs[CUDA_ATTN_FAST_THREADS];
    __shared__ float red_max[CUDA_ATTN_FAST_THREADS / 32];
    __shared__ float red_sum[CUDA_ATTN_FAST_THREADS / 32];
    if (tid < head_width) {
        const size_t at = position * stride + hbase + (size_t)tid;
        request_k[at] = cuda_bf16_from_float(request_qkv[width + hbase + (size_t)tid]);
        request_v[at] = cuda_bf16_from_float(request_qkv[width * 2u + hbase + (size_t)tid]);
        qs[tid] = request_qkv[hbase + (size_t)tid];
    }
    __syncthreads();
    const int groups = CUDA_ATTN_FAST_THREADS / head_width;
    const int d = tid % head_width;
    const int g = tid / head_width;
    const size_t n = position + 1u;
    float running_max = -1.0e30f;
    float denominator = 0.0f;
    float acc = 0.0f;
    for (size_t c0 = 0; c0 < n; c0 += CUDA_ATTN_FAST_THREADS) {
        const size_t s = c0 + (size_t)tid;
        float score = -1.0e30f;
        if (s < n) {
            const uint4 *kr = reinterpret_cast<const uint4 *>(
                SHARED && s < shared_len ? shared_k + s * width + hbase
                                         : request_k + s * stride + hbase);
            float dot = 0.0f;
            for (int i = 0; i < head_width / 8; ++i) {
                const uint4 w = kr[i];
                const float *qq = qs + i * 8;
                dot += qq[0] * __uint_as_float(w.x << 16) +
                       qq[1] * __uint_as_float(w.x & 0xffff0000u) +
                       qq[2] * __uint_as_float(w.y << 16) +
                       qq[3] * __uint_as_float(w.y & 0xffff0000u) +
                       qq[4] * __uint_as_float(w.z << 16) +
                       qq[5] * __uint_as_float(w.z & 0xffff0000u) +
                       qq[6] * __uint_as_float(w.w << 16) +
                       qq[7] * __uint_as_float(w.w & 0xffff0000u);
            }
            score = dot * scale;
        }
        float m = cuda_warp_max(score);
        if (lane == 0) red_max[warp] = m;
        __syncthreads();
        m = red_max[0];
        for (int w = 1; w < CUDA_ATTN_FAST_THREADS / 32; ++w) m = fmaxf(m, red_max[w]);
        const float next = fmaxf(running_max, m);
        const float p = s < n ? expf(score - next) : 0.0f;
        probs[tid] = p;
        float sum = cuda_warp_sum(p);
        if (lane == 0) red_sum[warp] = sum;
        __syncthreads();
        sum = red_sum[0];
        for (int w = 1; w < CUDA_ATTN_FAST_THREADS / 32; ++w) sum += red_sum[w];
        const float correction = expf(running_max - next);
        denominator = denominator * correction + sum;
        running_max = next;
        acc *= correction;
        const size_t left = n - c0;
        const int count = left < (size_t)CUDA_ATTN_FAST_THREADS
                              ? (int)left : CUDA_ATTN_FAST_THREADS;
        if (SHARED && c0 < shared_len) {
            for (int j = g; j < count; j += groups) {
                const size_t s = c0 + (size_t)j;
                const uint16_t *v = s < shared_len
                    ? shared_v + s * width + hbase + (size_t)d
                    : request_v + s * stride + hbase + (size_t)d;
                acc += probs[j] * cuda_bf16_to_float(*v);
            }
        } else {
            const uint16_t *vbase = request_v + c0 * stride + hbase + (size_t)d;
            for (int j = g; j < count; j += groups)
                acc += probs[j] * cuda_bf16_to_float(vbase[(size_t)j * stride]);
        }
        __syncthreads();
    }
    probs[tid] = acc;
    __syncthreads();
    if (g == 0) {
        float total = probs[d];
        for (int k = 1; k < groups; ++k) total += probs[k * head_width + d];
        const float value = denominator > 0.0f ? total / denominator : 0.0f;
        if (BF16OUT)
            out_bf16[(size_t)request * width + hbase + (size_t)d] =
                cuda_bf16_from_float(value);
        else
            out[(size_t)request * width + hbase + (size_t)d] = value;
    }
}

/* MYNAH_CUDA_ATTN_SPLIT=1: flash-decoding layout of the same decode attention
 * (same inputs, tables, SHARED/BF16OUT behaviour and K/V write as
 * k_self_attention_bf16_batch_fast), for head_width 64 only.  One 128-thread
 * block per (head, row), 4 warps.  Inside a warp, lane group g = lane / 8
 * (4 groups) owns one position and lane sub = lane % 8 owns dims
 * [8 sub, 8 sub + 8): one uint4 load per lane reads a whole 128-byte K (or V)
 * head row per group, so a warp load instruction fetches 4 positions' rows,
 * and each lane keeps UNROLL K plus UNROLL V loads in flight per step.  The
 * dot product is reduced inside the 8-lane group (3 shuffles); each lane keeps
 * its own running max / denominator / 8 accumulators (online softmax, no
 * block barrier in the loop).  Warp w walks positions
 * [w * 16 + k * 64, w * 16 + k * 64 + 16), k = 0, 1, ...; at the end the 4
 * groups of a warp merge by xor-shuffle (8, then 16) and the 4 warps merge in
 * shared memory in warp order.  The partition and the merge order depend only
 * on the row's own length, so the result is deterministic and independent of
 * the batch; it differs from the fast kernel in the last bits (different
 * summation order), all math FP32. */
static constexpr int CUDA_ATTN_SPLIT_THREADS = 128;
static constexpr int CUDA_ATTN_SPLIT_WARPS = CUDA_ATTN_SPLIT_THREADS / 32;
static constexpr int CUDA_ATTN_SPLIT_UNROLL = 4;
static constexpr int CUDA_ATTN_SPLIT_WARP_TILE = 4 * CUDA_ATTN_SPLIT_UNROLL;
static constexpr int CUDA_ATTN_SPLIT_TILE =
    CUDA_ATTN_SPLIT_WARPS * CUDA_ATTN_SPLIT_WARP_TILE;
static constexpr int CUDA_ATTN_SPLIT_HEAD_WIDTH = 64;

__device__ static inline void cuda_bf16x8_unpack(const uint4 w, float f[8]) {
    f[0] = __uint_as_float(w.x << 16);
    f[1] = __uint_as_float(w.x & 0xffff0000u);
    f[2] = __uint_as_float(w.y << 16);
    f[3] = __uint_as_float(w.y & 0xffff0000u);
    f[4] = __uint_as_float(w.z << 16);
    f[5] = __uint_as_float(w.z & 0xffff0000u);
    f[6] = __uint_as_float(w.w << 16);
    f[7] = __uint_as_float(w.w & 0xffff0000u);
}

template <bool SHARED, bool BF16OUT>
__global__ static void __launch_bounds__(CUDA_ATTN_SPLIT_THREADS)
k_self_attention_bf16_batch_split(
    const float *qkv, const uint16_t *const *kcache,
    const uint16_t *const *vcache, const size_t *positions,
    const size_t *cache_strides, int batch, int heads, int head_width,
    float scale, float *out, const uint16_t *const *kprefix,
    const uint16_t *const *vprefix, const size_t *prefix_len,
    uint16_t *out_bf16) {
    constexpr int HW = CUDA_ATTN_SPLIT_HEAD_WIDTH;
    constexpr int U = CUDA_ATTN_SPLIT_UNROLL;
    const int head = (int)blockIdx.x;
    const int request = (int)blockIdx.y;
    if (head >= heads || request >= batch || head_width != HW) return;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 3;
    const int sub = lane & 7;
    const size_t width = (size_t)heads * (size_t)HW;
    const size_t hbase = (size_t)head * (size_t)HW;
    const size_t position = positions[request];
    const size_t stride = cache_strides[request];
    uint16_t *request_k = const_cast<uint16_t *>(kcache[request]);
    uint16_t *request_v = const_cast<uint16_t *>(vcache[request]);
    const size_t shared_len = SHARED ? prefix_len[request] : 0u;
    const uint16_t *shared_k = SHARED ? kprefix[request] : nullptr;
    const uint16_t *shared_v = SHARED ? vprefix[request] : nullptr;
    const float *request_qkv = qkv + (size_t)request * width * 3u;
    __shared__ float part_m[CUDA_ATTN_SPLIT_WARPS];
    __shared__ float part_l[CUDA_ATTN_SPLIT_WARPS];
    __shared__ float part_acc[CUDA_ATTN_SPLIT_WARPS][HW];
    /* The new K/V first, exactly as the fast kernel (position >= the shared
     * prefix length, so a stripped row is written only where it stores). */
    if (tid < HW) {
        const size_t at = position * stride + hbase + (size_t)tid;
        request_k[at] = cuda_bf16_from_float(request_qkv[width + hbase + (size_t)tid]);
        request_v[at] = cuda_bf16_from_float(request_qkv[width * 2u + hbase + (size_t)tid]);
    }
    float q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = request_qkv[hbase + (size_t)(sub * 8 + i)];
    /* Makes the K/V store above visible to the whole block before any load. */
    __syncthreads();
    const size_t n = position + 1u;
    float m = -1.0e30f;
    float l = 0.0f;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    for (size_t base = (size_t)warp * CUDA_ATTN_SPLIT_WARP_TILE; base < n;
         base += CUDA_ATTN_SPLIT_TILE) {
        uint4 kw[U];
        uint4 vw[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const size_t s = base + (size_t)(u * 4 + group);
            kw[u] = make_uint4(0u, 0u, 0u, 0u);
            vw[u] = make_uint4(0u, 0u, 0u, 0u);
            if (s < n) {
                const bool pre = SHARED && s < shared_len;
                const uint16_t *kr = pre ? shared_k + s * width + hbase
                                         : request_k + s * stride + hbase;
                const uint16_t *vr = pre ? shared_v + s * width + hbase
                                         : request_v + s * stride + hbase;
                kw[u] = reinterpret_cast<const uint4 *>(kr)[sub];
                vw[u] = reinterpret_cast<const uint4 *>(vr)[sub];
            }
        }
        float sc[U];
        float next = m;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            float kf[8];
            cuda_bf16x8_unpack(kw[u], kf);
            float dot = q[0] * kf[0] + q[1] * kf[1] + q[2] * kf[2] +
                        q[3] * kf[3] + q[4] * kf[4] + q[5] * kf[5] +
                        q[6] * kf[6] + q[7] * kf[7];
            dot += __shfl_xor_sync(0xffffffffu, dot, 1);
            dot += __shfl_xor_sync(0xffffffffu, dot, 2);
            dot += __shfl_xor_sync(0xffffffffu, dot, 4);
            const bool valid = base + (size_t)(u * 4 + group) < n;
            sc[u] = valid ? dot * scale : -1.0e30f;
            next = fmaxf(next, sc[u]);
        }
        const float correction = expf(m - next);
        l *= correction;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] *= correction;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const bool valid = base + (size_t)(u * 4 + group) < n;
            const float p = valid ? expf(sc[u] - next) : 0.0f;
            l += p;
            float vf[8];
            cuda_bf16x8_unpack(vw[u], vf);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += p * vf[i];
        }
        m = next;
    }
    /* Merge the 4 position groups of the warp (fixed xor order). */
#pragma unroll
    for (int off = 8; off < 32; off <<= 1) {
        const float mo = __shfl_xor_sync(0xffffffffu, m, off);
        const float lo = __shfl_xor_sync(0xffffffffu, l, off);
        const float mm = fmaxf(m, mo);
        const float c1 = expf(m - mm);
        const float c2 = expf(mo - mm);
        l = l * c1 + lo * c2;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float ao = __shfl_xor_sync(0xffffffffu, acc[i], off);
            acc[i] = acc[i] * c1 + ao * c2;
        }
        m = mm;
    }
    if (lane < 8) {
        if (lane == 0) {
            part_m[warp] = m;
            part_l[warp] = l;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) part_acc[warp][sub * 8 + i] = acc[i];
    }
    __syncthreads();
    /* Merge the warps in warp order. A warp with no position has m = -1e30
     * and l = 0, so its weight is 0 (warp 0 always holds position 0). */
    if (tid < HW) {
        float mm = part_m[0];
#pragma unroll
        for (int w = 1; w < CUDA_ATTN_SPLIT_WARPS; ++w) mm = fmaxf(mm, part_m[w]);
        float den = 0.0f;
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < CUDA_ATTN_SPLIT_WARPS; ++w) {
            const float c = expf(part_m[w] - mm);
            den += part_l[w] * c;
            total += part_acc[w][tid] * c;
        }
        const float value = den > 0.0f ? total / den : 0.0f;
        const size_t at = (size_t)request * width + hbase + (size_t)tid;
        if (BF16OUT)
            out_bf16[at] = cuda_bf16_from_float(value);
        else
            out[at] = value;
    }
}

static bool cuda_backbone_attn_split_enabled(void) {
    static const bool on = [] {
        /* Default on; MYNAH_CUDA_ATTN_SPLIT=0 is the rollback to the fast
         * kernel. */
        const char *s = getenv("MYNAH_CUDA_ATTN_SPLIT");
        return s == nullptr || strcmp(s, "0") != 0;
    }();
    return on;
}

static bool cuda_backbone_attn_fast_enabled(void) {
    static const bool on = [] {
        const char *s = getenv("MYNAH_CUDA_BACKBONE_ATTN");
        return s == nullptr || strcmp(s, "legacy") != 0;
    }();
    return on;
}

extern "C" int mynah_cuda_self_attention_batch_dev(
    void *opaque, const float *qkv, float *const *kcache,
    float *const *vcache, const size_t *positions,
    const size_t *cache_strides, size_t batch, size_t heads,
    size_t head_width, float scale,
    float *out, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (qkv == nullptr || kcache == nullptr || vcache == nullptr ||
        positions == nullptr || cache_strides == nullptr || out == nullptr ||
        batch == 0u ||
        batch > st->batch_meta_cap || heads == 0u || head_width == 0u ||
        heads > (size_t)INT_MAX || head_width > (size_t)INT_MAX ||
        heads > SIZE_MAX / head_width) {
        set_error(e, ec, "invalid CUDA batched self-attention dimensions");
        return -1;
    }
    const size_t width = heads * head_width;
    size_t qkv_count = 0;
    if (!cuda_size_mul(width, 3u, &qkv_count) ||
        !cuda_size_mul(batch, qkv_count, &qkv_count)) {
        set_error(e, ec, "CUDA batched self-attention size overflow");
        return -1;
    }
    for (size_t i = 0; i < batch; ++i) {
        if (kcache[i] == nullptr || vcache[i] == nullptr ||
            positions[i] == SIZE_MAX ||
            cache_strides[i] < width ||
            positions[i] > (SIZE_MAX - (width - 1u)) / cache_strides[i]) {
            set_error(e, ec, "invalid CUDA batched self-attention cache");
            return -1;
        }
    }
    if (ce(cudaMemcpyAsync(st->dev_batch_k_cache, kcache,
                           batch * sizeof(*kcache), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_v_cache, vcache,
                           batch * sizeof(*vcache), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_positions, positions,
                           batch * sizeof(*positions), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_cache_strides, cache_strides,
                           batch * sizeof(*cache_strides), cudaMemcpyHostToDevice,
                           st->stream), e, ec)) return -1;
    dim3 grid((unsigned)heads, (unsigned)batch, 1u);
    k_self_attention_batch<<<grid, attention_threads(head_width), 0, st->stream>>>(
        qkv, st->dev_batch_k_cache, st->dev_batch_v_cache,
        st->dev_batch_positions, st->dev_batch_cache_strides,
        (int)batch, (int)heads,
        (int)head_width, scale, out);
    return ce(cudaGetLastError(), e, ec);
}

/* Launches the split decode attention on the tables already uploaded to
 * st->dev_batch_* (prefix tables only for SHARED); `staged` != NULL selects
 * the BF16OUT store. */
template <bool SHARED>
static void cuda_launch_attn_split(cuda_backend_state *st, dim3 grid,
                                   const float *qkv, size_t batch,
                                   size_t heads, size_t head_width,
                                   float scale, float *out, uint16_t *staged) {
    const uint16_t *const *kp = SHARED
        ? reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_prefix)
        : nullptr;
    const uint16_t *const *vp = SHARED
        ? reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_prefix)
        : nullptr;
    const size_t *pl = SHARED ? st->dev_batch_prefix_len : nullptr;
    const auto *kc = reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_cache);
    const auto *vc = reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_cache);
    if (staged != nullptr)
        k_self_attention_bf16_batch_split<SHARED, true>
            <<<grid, CUDA_ATTN_SPLIT_THREADS, 0, st->stream>>>(
            qkv, kc, vc, st->dev_batch_positions, st->dev_batch_cache_strides,
            (int)batch, (int)heads, (int)head_width, scale, out, kp, vp, pl,
            staged);
    else
        k_self_attention_bf16_batch_split<SHARED, false>
            <<<grid, CUDA_ATTN_SPLIT_THREADS, 0, st->stream>>>(
            qkv, kc, vc, st->dev_batch_positions, st->dev_batch_cache_strides,
            (int)batch, (int)heads, (int)head_width, scale, out, kp, vp, pl,
            nullptr);
}

/* int8 backbone KV (default, MYNAH_CUDA_KV_DTYPE): k_self_attention_bf16_batch_split over int8 row
 * records (see tile_row_load: `width` int8 values then `heads` float scales
 * per position, `cache_strides` in bytes). Same partition, online softmax and
 * merge order; the new K/V is quantized per head (scale = max|x| / 127) before
 * the block reads it, so the current position is read back exactly as later
 * steps will read it. The shared voice prefix stays BF16. */
__device__ static inline void cuda_i8x8_unpack(const uint4 w, float f[8]) {
    const float s = __uint_as_float(w.z);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[i] = (float)(signed char)(w.x >> (8 * i)) * s;
        f[4 + i] = (float)(signed char)(w.y >> (8 * i)) * s;
    }
}

template <bool SHARED, bool BF16OUT>
__global__ static void __launch_bounds__(CUDA_ATTN_SPLIT_THREADS)
k_self_attention_i8_batch_split(
    const float *qkv, const int8_t *const *kcache,
    const int8_t *const *vcache, const size_t *positions,
    const size_t *cache_strides, int batch, int heads, int head_width,
    float scale, float *out, const uint16_t *const *kprefix,
    const uint16_t *const *vprefix, const size_t *prefix_len,
    uint16_t *out_bf16) {
    constexpr int HW = CUDA_ATTN_SPLIT_HEAD_WIDTH;
    constexpr int U = CUDA_ATTN_SPLIT_UNROLL;
    const int head = (int)blockIdx.x;
    const int request = (int)blockIdx.y;
    if (head >= heads || request >= batch || head_width != HW) return;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int group = lane >> 3;
    const int sub = lane & 7;
    const size_t width = (size_t)heads * (size_t)HW;
    const size_t hbase = (size_t)head * (size_t)HW;
    const size_t position = positions[request];
    const size_t stride = cache_strides[request];
    int8_t *request_k = const_cast<int8_t *>(kcache[request]);
    int8_t *request_v = const_cast<int8_t *>(vcache[request]);
    const size_t shared_len = SHARED ? prefix_len[request] : 0u;
    const uint16_t *shared_k = SHARED ? kprefix[request] : nullptr;
    const uint16_t *shared_v = SHARED ? vprefix[request] : nullptr;
    const float *request_qkv = qkv + (size_t)request * width * 3u;
    __shared__ float part_m[CUDA_ATTN_SPLIT_WARPS];
    __shared__ float part_l[CUDA_ATTN_SPLIT_WARPS];
    __shared__ float part_acc[CUDA_ATTN_SPLIT_WARPS][HW];
    __shared__ float amax[2][2];
    float kx = 0.0f, vx = 0.0f;
    if (tid < HW) {
        kx = request_qkv[width + hbase + (size_t)tid];
        vx = request_qkv[width * 2u + hbase + (size_t)tid];
        const float km = cuda_warp_max(fabsf(kx));
        const float vm = cuda_warp_max(fabsf(vx));
        if (lane == 0) {
            amax[0][warp] = km;
            amax[1][warp] = vm;
        }
    }
    __syncthreads();
    if (tid < HW) {
        const float km = fmaxf(amax[0][0], amax[0][1]);
        const float vm = fmaxf(amax[1][0], amax[1][1]);
        int8_t *krec = request_k + position * stride;
        int8_t *vrec = request_v + position * stride;
        krec[hbase + (size_t)tid] = cuda_kv_q8(kx, km > 0.0f ? 127.0f / km : 0.0f);
        vrec[hbase + (size_t)tid] = cuda_kv_q8(vx, vm > 0.0f ? 127.0f / vm : 0.0f);
        if (tid == 0) {
            reinterpret_cast<float *>(krec + width)[head] = km / 127.0f;
            reinterpret_cast<float *>(vrec + width)[head] = vm / 127.0f;
        }
    }
    float q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) q[i] = request_qkv[hbase + (size_t)(sub * 8 + i)];
    __syncthreads();
    const size_t n = position + 1u;
    float m = -1.0e30f;
    float l = 0.0f;
    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = 0.0f;
    for (size_t base = (size_t)warp * CUDA_ATTN_SPLIT_WARP_TILE; base < n;
         base += CUDA_ATTN_SPLIT_TILE) {
        /* Raw words in flight: a BF16 prefix row (uint4) or an int8 row as
         * (8 bytes, scale bits, 0); `pre` says which. */
        uint4 kw[U];
        uint4 vw[U];
        bool pre[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const size_t s = base + (size_t)(u * 4 + group);
            kw[u] = make_uint4(0u, 0u, 0u, 0u);
            vw[u] = make_uint4(0u, 0u, 0u, 0u);
            pre[u] = SHARED && s < shared_len;
            if (s < n) {
                if (pre[u]) {
                    kw[u] = reinterpret_cast<const uint4 *>(
                        shared_k + s * width + hbase)[sub];
                    vw[u] = reinterpret_cast<const uint4 *>(
                        shared_v + s * width + hbase)[sub];
                } else {
                    const int8_t *kr = request_k + s * stride;
                    const int8_t *vr = request_v + s * stride;
                    const uint2 kq = reinterpret_cast<const uint2 *>(kr + hbase)[sub];
                    const uint2 vq = reinterpret_cast<const uint2 *>(vr + hbase)[sub];
                    kw[u] = make_uint4(kq.x, kq.y,
                        __float_as_uint(reinterpret_cast<const float *>(kr + width)[head]), 0u);
                    vw[u] = make_uint4(vq.x, vq.y,
                        __float_as_uint(reinterpret_cast<const float *>(vr + width)[head]), 0u);
                }
            }
        }
        float sc[U];
        float next = m;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            float kf[8];
            if (pre[u]) cuda_bf16x8_unpack(kw[u], kf);
            else cuda_i8x8_unpack(kw[u], kf);
            float dot = q[0] * kf[0] + q[1] * kf[1] + q[2] * kf[2] +
                        q[3] * kf[3] + q[4] * kf[4] + q[5] * kf[5] +
                        q[6] * kf[6] + q[7] * kf[7];
            dot += __shfl_xor_sync(0xffffffffu, dot, 1);
            dot += __shfl_xor_sync(0xffffffffu, dot, 2);
            dot += __shfl_xor_sync(0xffffffffu, dot, 4);
            const bool valid = base + (size_t)(u * 4 + group) < n;
            sc[u] = valid ? dot * scale : -1.0e30f;
            next = fmaxf(next, sc[u]);
        }
        const float correction = expf(m - next);
        l *= correction;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] *= correction;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const bool valid = base + (size_t)(u * 4 + group) < n;
            const float p = valid ? expf(sc[u] - next) : 0.0f;
            l += p;
            float vf[8];
            if (pre[u]) cuda_bf16x8_unpack(vw[u], vf);
            else cuda_i8x8_unpack(vw[u], vf);
#pragma unroll
            for (int i = 0; i < 8; ++i) acc[i] += p * vf[i];
        }
        m = next;
    }
#pragma unroll
    for (int off = 8; off < 32; off <<= 1) {
        const float mo = __shfl_xor_sync(0xffffffffu, m, off);
        const float lo = __shfl_xor_sync(0xffffffffu, l, off);
        const float mm = fmaxf(m, mo);
        const float c1 = expf(m - mm);
        const float c2 = expf(mo - mm);
        l = l * c1 + lo * c2;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float ao = __shfl_xor_sync(0xffffffffu, acc[i], off);
            acc[i] = acc[i] * c1 + ao * c2;
        }
        m = mm;
    }
    if (lane < 8) {
        if (lane == 0) {
            part_m[warp] = m;
            part_l[warp] = l;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) part_acc[warp][sub * 8 + i] = acc[i];
    }
    __syncthreads();
    if (tid < HW) {
        float mm = part_m[0];
#pragma unroll
        for (int w = 1; w < CUDA_ATTN_SPLIT_WARPS; ++w) mm = fmaxf(mm, part_m[w]);
        float den = 0.0f;
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < CUDA_ATTN_SPLIT_WARPS; ++w) {
            const float c = expf(part_m[w] - mm);
            den += part_l[w] * c;
            total += part_acc[w][tid] * c;
        }
        const float value = den > 0.0f ? total / den : 0.0f;
        const size_t at = (size_t)request * width + hbase + (size_t)tid;
        if (BF16OUT)
            out_bf16[at] = cuda_bf16_from_float(value);
        else
            out[at] = value;
    }
}

template <bool SHARED>
static void cuda_launch_attn_i8(cuda_backend_state *st, dim3 grid,
                                const float *qkv, size_t batch, size_t heads,
                                size_t head_width, float scale, float *out,
                                uint16_t *staged) {
    const uint16_t *const *kp = SHARED
        ? reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_prefix)
        : nullptr;
    const uint16_t *const *vp = SHARED
        ? reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_prefix)
        : nullptr;
    const size_t *pl = SHARED ? st->dev_batch_prefix_len : nullptr;
    const auto *kc = reinterpret_cast<const int8_t *const *>(st->dev_batch_k_cache);
    const auto *vc = reinterpret_cast<const int8_t *const *>(st->dev_batch_v_cache);
    if (staged != nullptr)
        k_self_attention_i8_batch_split<SHARED, true>
            <<<grid, CUDA_ATTN_SPLIT_THREADS, 0, st->stream>>>(
            qkv, kc, vc, st->dev_batch_positions, st->dev_batch_cache_strides,
            (int)batch, (int)heads, (int)head_width, scale, out, kp, vp, pl,
            staged);
    else
        k_self_attention_i8_batch_split<SHARED, false>
            <<<grid, CUDA_ATTN_SPLIT_THREADS, 0, st->stream>>>(
            qkv, kc, vc, st->dev_batch_positions, st->dev_batch_cache_strides,
            (int)batch, (int)heads, (int)head_width, scale, out, kp, vp, pl,
            nullptr);
}

extern "C" int mynah_cuda_set_kv_int8(void *opaque, int on) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr) return -1;
    st->kv_int8 = on != 0;
    return 0;
}

/* `stage` (MYNAH_CUDA_BF16_FUSE): the result goes to the staged BF16
 * activation instead of `out`; the legacy kernel still writes `out` and a
 * cast stages it, so the staged values do not depend on the kernel. */
static int cuda_self_attention_bf16_batch(
    cuda_backend_state *st, const float *qkv, void *const *kcache,
    void *const *vcache, void *const *kprefix, void *const *vprefix,
    const size_t *prefix_len, const size_t *positions,
    const size_t *cache_strides, size_t batch, size_t heads,
    size_t head_width, float scale, float *out, char *e, size_t ec,
    bool stage = false) {
    if (qkv == nullptr || kcache == nullptr || vcache == nullptr ||
        positions == nullptr || cache_strides == nullptr || out == nullptr ||
        batch == 0u || batch > st->batch_meta_cap || heads == 0u ||
        head_width == 0u || heads > (size_t)INT_MAX ||
        head_width > (size_t)INT_MAX || heads > SIZE_MAX / head_width) {
        set_error(e, ec, "invalid CUDA BF16 batched self-attention dimensions");
        return -1;
    }
    const size_t width = heads * head_width;
    size_t qkv_count = 0u;
    if (!cuda_size_mul(width, 3u, &qkv_count) ||
        !cuda_size_mul(batch, qkv_count, &qkv_count)) {
        set_error(e, ec, "CUDA BF16 batched self-attention size overflow");
        return -1;
    }
    size_t out_count = 0u;
    if (stage && (!cuda_size_mul(batch, width, &out_count) ||
                  out_count > (size_t)INT_MAX ||
                  ensure_bf16_activation(st, out_count, e, ec) != 0)) {
        if (out_count > (size_t)INT_MAX)
            set_error(e, ec, "CUDA BF16 staged attention is too large");
        return -1;
    }
    uint16_t *staged = stage ? st->dev_bf16_activation : nullptr;
    for (size_t i = 0; i < batch; ++i) {
        if (kcache[i] == nullptr || vcache[i] == nullptr ||
            positions[i] == SIZE_MAX || cache_strides[i] < width ||
            positions[i] > (SIZE_MAX - (width - 1u)) / cache_strides[i]) {
            set_error(e, ec, "invalid CUDA BF16 batched self-attention cache");
            return -1;
        }
        if (prefix_len != nullptr && prefix_len[i] != 0u &&
            (prefix_len[i] > positions[i] || kprefix[i] == nullptr ||
             vprefix[i] == nullptr)) {
            set_error(e, ec, "invalid CUDA BF16 shared voice prefix");
            return -1;
        }
    }
    if (ce(cudaMemcpyAsync(st->dev_batch_k_cache, kcache,
                           batch * sizeof(*kcache), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_v_cache, vcache,
                           batch * sizeof(*vcache), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_positions, positions,
                           batch * sizeof(*positions), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_cache_strides, cache_strides,
                           batch * sizeof(*cache_strides), cudaMemcpyHostToDevice,
                           st->stream), e, ec)) return -1;
    dim3 grid((unsigned)heads, (unsigned)batch, 1u);
    if (st->kv_int8) {
        /* int8 records: the split layout only (head width 64, 16-byte
         * aligned records and BF16 prefix rows). */
        const bool with_prefix = prefix_len != nullptr;
        bool ok = head_width == (size_t)CUDA_ATTN_SPLIT_HEAD_WIDTH &&
                  width % 8u == 0u;
        for (size_t i = 0; ok && i < batch; ++i)
            ok = cache_strides[i] % 16u == 0u &&
                 cache_strides[i] >= width + 4u * heads &&
                 ((uintptr_t)kcache[i] & 15u) == 0u &&
                 ((uintptr_t)vcache[i] & 15u) == 0u &&
                 (!with_prefix || prefix_len[i] == 0u ||
                  (((uintptr_t)kprefix[i] & 15u) == 0u &&
                   ((uintptr_t)vprefix[i] & 15u) == 0u));
        if (!ok) {
            set_error(e, ec, "CUDA int8 KV attention needs head width 64 and "
                             "16-byte aligned records");
            return -1;
        }
        if (with_prefix &&
            (ce(cudaMemcpyAsync(st->dev_batch_k_prefix, kprefix,
                                batch * sizeof(*kprefix), cudaMemcpyHostToDevice,
                                st->stream), e, ec) ||
             ce(cudaMemcpyAsync(st->dev_batch_v_prefix, vprefix,
                                batch * sizeof(*vprefix), cudaMemcpyHostToDevice,
                                st->stream), e, ec) ||
             ce(cudaMemcpyAsync(st->dev_batch_prefix_len, prefix_len,
                                batch * sizeof(*prefix_len),
                                cudaMemcpyHostToDevice, st->stream), e, ec)))
            return -1;
        static std::atomic<bool> i8_announced{false};
        if (!i8_announced.exchange(true))
            std::fprintf(stderr,
                         "mynah-tts: CUDA decode attention reads int8 KV rows "
                         "(int8 KV, the default; MYNAH_CUDA_KV_DTYPE=bf16 rolls back)\n");
        if (with_prefix)
            cuda_launch_attn_i8<true>(st, grid, qkv, batch, heads, head_width,
                                      scale, out, stage ? staged : nullptr);
        else
            cuda_launch_attn_i8<false>(st, grid, qkv, batch, heads, head_width,
                                       scale, out, stage ? staged : nullptr);
        return ce(cudaGetLastError(), e, ec);
    }
    bool fast = cuda_backbone_attn_fast_enabled() && head_width % 8u == 0u &&
                head_width <= (size_t)CUDA_ATTN_FAST_THREADS &&
                (size_t)CUDA_ATTN_FAST_THREADS % head_width == 0u;
    for (size_t i = 0; fast && i < batch; ++i)
        fast = cache_strides[i] % 8u == 0u &&
               ((uintptr_t)kcache[i] & 15u) == 0u &&
               ((uintptr_t)vcache[i] & 15u) == 0u;
    /* With prefix tables every kernel reads positions [0, prefix_len) from
     * the shared planes: a row may not store its prefix at all
     * (MYNAH_CUDA_SHARED_VOICE), so no fallback may read the row there. The
     * fast kernel additionally needs 16-byte aligned prefix rows; otherwise
     * the legacy kernel takes the prefix variant. Without prefix tables the
     * kernels are exactly the ones this function always launched. */
    const bool prefixed = prefix_len != nullptr;
    bool shared = fast && prefixed && width % 8u == 0u;
    for (size_t i = 0; shared && i < batch; ++i)
        shared = ((uintptr_t)kprefix[i] & 15u) == 0u &&
                 ((uintptr_t)vprefix[i] & 15u) == 0u;
    /* MYNAH_CUDA_ATTN_SPLIT=1: the flash-decoding kernel wherever the fast
     * kernel would run and the head is 64 wide; same tables, same fallbacks. */
    const bool split = fast && head_width == (size_t)CUDA_ATTN_SPLIT_HEAD_WIDTH &&
                       cuda_backbone_attn_split_enabled();
    if (split) {
        static std::atomic<bool> split_announced{false};
        if (!split_announced.exchange(true))
            std::fprintf(stderr,
                         "mynah-tts: CUDA decode attention uses the split "
                         "(flash-decoding) kernel (MYNAH_CUDA_ATTN_SPLIT=1)\n");
    }
    if (prefixed &&
        (ce(cudaMemcpyAsync(st->dev_batch_k_prefix, kprefix,
                            batch * sizeof(*kprefix), cudaMemcpyHostToDevice,
                            st->stream), e, ec) ||
         ce(cudaMemcpyAsync(st->dev_batch_v_prefix, vprefix,
                            batch * sizeof(*vprefix), cudaMemcpyHostToDevice,
                            st->stream), e, ec) ||
         ce(cudaMemcpyAsync(st->dev_batch_prefix_len, prefix_len,
                            batch * sizeof(*prefix_len), cudaMemcpyHostToDevice,
                            st->stream), e, ec))) return -1;
    if (prefixed) {
        /* One line per process, the first time the shared path really runs. */
        static std::atomic<bool> announced{false};
        if (!announced.exchange(true))
            std::fprintf(stderr,
                         "mynah-tts: CUDA decode attention reads voice prefixes "
                         "from the shared device voice cache "
                         "(MYNAH_CUDA_SHARED_VOICE=1)\n");
        if (!shared) {
            k_self_attention_bf16_batch<true><<<grid, attention_threads(head_width),
                                                0, st->stream>>>(
                qkv,
                reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_cache),
                reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_cache),
                st->dev_batch_positions, st->dev_batch_cache_strides, (int)batch,
                (int)heads, (int)head_width, scale, out,
                reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_prefix),
                reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_prefix),
                st->dev_batch_prefix_len);
            if (ce(cudaGetLastError(), e, ec)) return -1;
            if (stage) {
                k_f32_to_bf16<<<((int)out_count + 255) / 256, 256, 0, st->stream>>>(
                    out, staged, (int)out_count);
                return ce(cudaGetLastError(), e, ec);
            }
            return 0;
        }
        if (split) {
            cuda_launch_attn_split<true>(st, grid, qkv, batch, heads, head_width,
                                         scale, out, stage ? staged : nullptr);
            return ce(cudaGetLastError(), e, ec);
        }
        if (stage) {
            k_self_attention_bf16_batch_fast<true, true>
                <<<grid, CUDA_ATTN_FAST_THREADS, 0, st->stream>>>(
                qkv,
                reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_cache),
                reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_cache),
                st->dev_batch_positions, st->dev_batch_cache_strides,
                (int)batch, (int)heads, (int)head_width, scale, out,
                reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_prefix),
                reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_prefix),
                st->dev_batch_prefix_len, staged);
        } else {
        k_self_attention_bf16_batch_fast<true, false><<<grid, CUDA_ATTN_FAST_THREADS, 0,
                                                 st->stream>>>(
            qkv,
            reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_cache),
            reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_cache),
            st->dev_batch_positions, st->dev_batch_cache_strides, (int)batch,
            (int)heads, (int)head_width, scale, out,
            reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_prefix),
            reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_prefix),
            st->dev_batch_prefix_len, nullptr);
        }
        return ce(cudaGetLastError(), e, ec);
    }
    if (fast) {
        if (split) {
            cuda_launch_attn_split<false>(st, grid, qkv, batch, heads, head_width,
                                          scale, out, stage ? staged : nullptr);
            return ce(cudaGetLastError(), e, ec);
        }
        if (stage) {
            k_self_attention_bf16_batch_fast<false, true>
                <<<grid, CUDA_ATTN_FAST_THREADS, 0, st->stream>>>(
                qkv,
                reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_cache),
                reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_cache),
                st->dev_batch_positions, st->dev_batch_cache_strides,
                (int)batch, (int)heads, (int)head_width, scale, out, nullptr,
                nullptr, nullptr, staged);
        } else {
        k_self_attention_bf16_batch_fast<false, false><<<grid, CUDA_ATTN_FAST_THREADS, 0,
                                                  st->stream>>>(
            qkv,
            reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_cache),
            reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_cache),
            st->dev_batch_positions, st->dev_batch_cache_strides, (int)batch,
            (int)heads, (int)head_width, scale, out, nullptr, nullptr, nullptr,
            nullptr);
        }
        return ce(cudaGetLastError(), e, ec);
    }
    k_self_attention_bf16_batch<false><<<grid, attention_threads(head_width), 0,
                                         st->stream>>>(
        qkv,
        reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_cache),
        reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_cache),
        st->dev_batch_positions, st->dev_batch_cache_strides, (int)batch,
        (int)heads, (int)head_width, scale, out, nullptr, nullptr, nullptr);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (stage) {
        k_f32_to_bf16<<<((int)out_count + 255) / 256, 256, 0, st->stream>>>(
            out, staged, (int)out_count);
        return ce(cudaGetLastError(), e, ec);
    }
    return 0;
}

extern "C" int mynah_cuda_self_attention_bf16_batch_dev(
    void *opaque, const float *qkv, void *const *kcache,
    void *const *vcache, const size_t *positions,
    const size_t *cache_strides, size_t batch, size_t heads,
    size_t head_width, float scale, float *out, char *e, size_t ec) {
    return cuda_self_attention_bf16_batch(
        static_cast<cuda_backend_state *>(opaque), qkv, kcache, vcache,
        nullptr, nullptr, nullptr, positions, cache_strides, batch, heads,
        head_width, scale, out, e, ec);
}

extern "C" int mynah_cuda_self_attention_bf16_prefix_batch_dev(
    void *opaque, const float *qkv, void *const *kcache,
    void *const *vcache, void *const *kprefix, void *const *vprefix,
    const size_t *prefix_len, const size_t *positions,
    const size_t *cache_strides, size_t batch, size_t heads,
    size_t head_width, float scale, float *out, char *e, size_t ec) {
    if (kprefix == nullptr || vprefix == nullptr || prefix_len == nullptr) {
        set_error(e, ec, "invalid CUDA BF16 shared voice prefix tables");
        return -1;
    }
    return cuda_self_attention_bf16_batch(
        static_cast<cuda_backend_state *>(opaque), qkv, kcache, vcache,
        kprefix, vprefix, prefix_len, positions, cache_strides, batch, heads,
        head_width, scale, out, e, ec);
}

/* MYNAH_CUDA_BF16_FUSE: the batched BF16-KV attention whose output is the
 * staged BF16 activation of the out_proj GEMM.  The prefix tables are all
 * NULL (plain rows) or all set (shared voice prefix). */
extern "C" int mynah_cuda_self_attention_bf16_stage_batch_dev(
    void *opaque, const float *qkv, void *const *kcache,
    void *const *vcache, void *const *kprefix, void *const *vprefix,
    const size_t *prefix_len, const size_t *positions,
    const size_t *cache_strides, size_t batch, size_t heads,
    size_t head_width, float scale, float *scratch, char *e, size_t ec) {
    const bool with_prefix = prefix_len != nullptr;
    if (with_prefix && (kprefix == nullptr || vprefix == nullptr)) {
        set_error(e, ec, "invalid CUDA BF16 shared voice prefix tables");
        return -1;
    }
    return cuda_self_attention_bf16_batch(
        static_cast<cuda_backend_state *>(opaque), qkv, kcache, vcache,
        with_prefix ? kprefix : nullptr, with_prefix ? vprefix : nullptr,
        with_prefix ? prefix_len : nullptr, positions, cache_strides, batch,
        heads, head_width, scale, scratch, e, ec, true);
}

/* MYNAH_CUDA_ATTN_SPLIT: the split decode attention against the fast kernel
 * on random data, with and without shared prefix tables, always run by
 * --gpu-self-test (independent of the flag).  Checks: FP32 output within
 * 1e-3 relative (summation order differs), BF16OUT within one BF16 rounding,
 * every row computed alone bit-identical to the same row inside the batch
 * (determinism, batch independence), and with prefix tables the row's own
 * positions below the prefix are NaN, so any read there fails the test. */
template <bool SHARED>
static int cuda_attn_split_case(cuda_backend_state *st, char *e, size_t ec) {
    constexpr int heads = 4, hw = CUDA_ATTN_SPLIT_HEAD_WIDTH;
    constexpr size_t width = (size_t)heads * hw;
    constexpr int rows = 8;
    const size_t positions[rows] = {0u, 1u, 15u, 64u, 127u, 300u, 701u, 2049u};
    const size_t strides[rows] = {width, width + 64u, width, width + 64u,
                                  width, width + 64u, width, width};
    const size_t plen_all[rows] = {0u, 0u, 0u, 40u, 126u, 126u, 126u, 0u};
    const size_t prefix_positions = 126u;
    const float scale = 0.125f;
    uint32_t seed = 0x2545f491u ^ (SHARED ? 0x5bd1e995u : 0u);
    auto uniform = [&](float span) {
        seed = seed * 1664525u + 1013904223u;
        return ((float)(seed >> 8) / 16777216.0f - 0.5f) * 2.0f * span;
    };
    auto bf16 = [](float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return (uint16_t)(bits >> 16);
    };
    /* One device slab: rows K then V, the prefix K/V planes, qkv, outputs. */
    size_t row_elems[rows], offset[rows], total = 0u;
    for (int r = 0; r < rows; ++r) {
        row_elems[r] = (positions[r] + 1u) * strides[r];
        offset[r] = total;
        total += 2u * row_elems[r];
    }
    const size_t prefix_off = total;
    total += 2u * prefix_positions * width;
    std::vector<uint16_t> kv(total);
    for (auto &x : kv) x = bf16(uniform(2.0f));
    if (SHARED)
        for (int r = 0; r < rows; ++r)
            for (size_t i = 0; i < plen_all[r] * strides[r]; ++i) {
                kv[offset[r] + i] = 0x7fc0u;                /* K NaN */
                kv[offset[r] + row_elems[r] + i] = 0x7fc0u; /* V NaN */
            }
    std::vector<float> qkv((size_t)rows * 3u * width);
    for (auto &x : qkv) x = uniform(3.0f);
    const size_t out_n = (size_t)rows * width;
    uint16_t *d_kv = nullptr;
    float *d_f = nullptr; /* qkv | ref | split | single */
    uint16_t *d_bf = nullptr;
    void *d_tab = nullptr;
    int result = -1;
    do {
        if (ce(cudaMalloc((void **)&d_kv, total * sizeof(uint16_t)), e, ec) ||
            ce(cudaMalloc((void **)&d_f, (qkv.size() + 3u * out_n) * sizeof(float)), e, ec) ||
            ce(cudaMalloc((void **)&d_bf, out_n * sizeof(uint16_t)), e, ec) ||
            ce(cudaMalloc(&d_tab, (size_t)rows * 7u * sizeof(void *)), e, ec))
            break;
        const uint16_t *hk[rows], *hv[rows], *hkp[rows], *hvp[rows];
        size_t hplen[rows];
        for (int r = 0; r < rows; ++r) {
            hk[r] = d_kv + offset[r];
            hv[r] = d_kv + offset[r] + row_elems[r];
            hkp[r] = d_kv + prefix_off;
            hvp[r] = d_kv + prefix_off + prefix_positions * width;
            hplen[r] = SHARED ? plen_all[r] : 0u;
        }
        auto **t_k = static_cast<const uint16_t **>(d_tab);
        auto **t_v = t_k + rows;
        auto **t_kp = t_k + 2 * rows;
        auto **t_vp = t_k + 3 * rows;
        auto *t_pos = reinterpret_cast<size_t *>(t_k + 4 * rows);
        auto *t_str = reinterpret_cast<size_t *>(t_k + 5 * rows);
        auto *t_pl = reinterpret_cast<size_t *>(t_k + 6 * rows);
        static_assert(sizeof(size_t) == sizeof(void *), "table layout");
        float *d_qkv = d_f, *d_ref = d_f + qkv.size(), *d_split = d_ref + out_n,
              *d_single = d_split + out_n;
        if (ce(cudaMemcpy(d_kv, kv.data(), total * sizeof(uint16_t), cudaMemcpyHostToDevice), e, ec) ||
            ce(cudaMemcpy(d_qkv, qkv.data(), qkv.size() * sizeof(float), cudaMemcpyHostToDevice), e, ec) ||
            ce(cudaMemcpy(t_k, hk, sizeof(hk), cudaMemcpyHostToDevice), e, ec) ||
            ce(cudaMemcpy(t_v, hv, sizeof(hv), cudaMemcpyHostToDevice), e, ec) ||
            ce(cudaMemcpy(t_kp, hkp, sizeof(hkp), cudaMemcpyHostToDevice), e, ec) ||
            ce(cudaMemcpy(t_vp, hvp, sizeof(hvp), cudaMemcpyHostToDevice), e, ec) ||
            ce(cudaMemcpy(t_pos, positions, sizeof(positions), cudaMemcpyHostToDevice), e, ec) ||
            ce(cudaMemcpy(t_str, strides, sizeof(strides), cudaMemcpyHostToDevice), e, ec) ||
            ce(cudaMemcpy(t_pl, hplen, sizeof(hplen), cudaMemcpyHostToDevice), e, ec))
            break;
        const uint16_t *const *kp = SHARED ? t_kp : nullptr;
        const uint16_t *const *vp = SHARED ? t_vp : nullptr;
        const size_t *pl = SHARED ? t_pl : nullptr;
        dim3 grid((unsigned)heads, (unsigned)rows, 1u);
        k_self_attention_bf16_batch_fast<SHARED, false>
            <<<grid, CUDA_ATTN_FAST_THREADS, 0, st->stream>>>(
            d_qkv, t_k, t_v, t_pos, t_str, rows, heads, hw, scale, d_ref, kp,
            vp, pl, nullptr);
        k_self_attention_bf16_batch_split<SHARED, false>
            <<<grid, CUDA_ATTN_SPLIT_THREADS, 0, st->stream>>>(
            d_qkv, t_k, t_v, t_pos, t_str, rows, heads, hw, scale, d_split, kp,
            vp, pl, nullptr);
        k_self_attention_bf16_batch_split<SHARED, true>
            <<<grid, CUDA_ATTN_SPLIT_THREADS, 0, st->stream>>>(
            d_qkv, t_k, t_v, t_pos, t_str, rows, heads, hw, scale, nullptr, kp,
            vp, pl, d_bf);
        /* Each row alone, tables shifted: must equal the batched row bitwise. */
        dim3 one((unsigned)heads, 1u, 1u);
        for (int r = 0; r < rows; ++r)
            k_self_attention_bf16_batch_split<SHARED, false>
                <<<one, CUDA_ATTN_SPLIT_THREADS, 0, st->stream>>>(
                d_qkv + (size_t)r * 3u * width, t_k + r, t_v + r, t_pos + r,
                t_str + r, 1, heads, hw, scale, d_single + (size_t)r * width,
                SHARED ? t_kp + r : nullptr, SHARED ? t_vp + r : nullptr,
                SHARED ? t_pl + r : nullptr, nullptr);
        if (ce(cudaGetLastError(), e, ec) || ce(cudaStreamSynchronize(st->stream), e, ec))
            break;
        std::vector<float> ref(out_n), split(out_n), single(out_n);
        std::vector<uint16_t> sbf(out_n);
        if (ce(cudaMemcpy(ref.data(), d_ref, out_n * sizeof(float), cudaMemcpyDeviceToHost), e, ec) ||
            ce(cudaMemcpy(split.data(), d_split, out_n * sizeof(float), cudaMemcpyDeviceToHost), e, ec) ||
            ce(cudaMemcpy(single.data(), d_single, out_n * sizeof(float), cudaMemcpyDeviceToHost), e, ec) ||
            ce(cudaMemcpy(sbf.data(), d_bf, out_n * sizeof(uint16_t), cudaMemcpyDeviceToHost), e, ec))
            break;
        result = 0;
        float worst = 0.0f;
        for (size_t i = 0; i < out_n && result == 0; ++i) {
            const float tol = std::fmax(1.0f, std::fabs(ref[i]));
            const float diff = std::fabs(split[i] - ref[i]);
            uint32_t bits = (uint32_t)sbf[i] << 16;
            float as_bf;
            std::memcpy(&as_bf, &bits, sizeof(as_bf));
            if (!std::isfinite(split[i]) || !std::isfinite(ref[i]) ||
                !(diff <= 1e-3f * tol)) {
                std::snprintf(e, ec,
                              "split attention (shared=%d) row %zu dim %zu: "
                              "%.7g vs fast %.7g",
                              SHARED ? 1 : 0, i / width, i % width,
                              (double)split[i], (double)ref[i]);
                result = -1;
            } else if (!(std::fabs(as_bf - ref[i]) <= 1e-2f * tol)) {
                std::snprintf(e, ec,
                              "split attention BF16 out (shared=%d) row %zu "
                              "dim %zu: %.7g vs fast %.7g",
                              SHARED ? 1 : 0, i / width, i % width,
                              (double)as_bf, (double)ref[i]);
                result = -1;
            }
            worst = std::fmax(worst, diff / tol);
        }
        if (result == 0 &&
            std::memcmp(split.data(), single.data(), out_n * sizeof(float)) != 0) {
            std::snprintf(e, ec,
                          "split attention (shared=%d) depends on the batch",
                          SHARED ? 1 : 0);
            result = -1;
        }
        if (result == 0 && getenv("MYNAH_CUDA_ATTN_SPLIT") != nullptr)
            std::fprintf(stderr,
                         "mynah-tts: split attention self-test shared=%d max "
                         "rel diff vs fast %.3g\n",
                         SHARED ? 1 : 0, (double)worst);
    } while (false);
    cudaFree(d_kv);
    cudaFree(d_f);
    cudaFree(d_bf);
    cudaFree(d_tab);
    return result;
}

static int cuda_attn_split_self_test(cuda_backend_state *st, char *e,
                                     size_t ec) {
    if (st == nullptr) return -1;
    if (cuda_attn_split_case<false>(st, e, ec) != 0) return -1;
    return cuda_attn_split_case<true>(st, e, ec);
}

__global__ static void k_gather_kv_batch(
    float *const *kcache, float *const *vcache, const size_t *positions,
    const size_t *cache_strides, int batch, size_t width, float *out) {
    const size_t index = (size_t)blockIdx.x * (size_t)blockDim.x +
                         (size_t)threadIdx.x;
    const size_t total = (size_t)batch * 2u * width;
    if (index >= total) return;
    const size_t request = index / (2u * width);
    const size_t part = (index / width) & 1u;
    const size_t column = index % width;
    const size_t source = positions[request] * cache_strides[request] + column;
    out[index] = (part == 0u ? kcache[request] : vcache[request])[source];
}

__global__ static void k_gather_kv_bf16_batch(
    const uint16_t *const *kcache, const uint16_t *const *vcache,
    const size_t *positions, const size_t *cache_strides, int batch,
    size_t width, float *out) {
    const size_t index = (size_t)blockIdx.x * (size_t)blockDim.x +
                         (size_t)threadIdx.x;
    const size_t total = (size_t)batch * 2u * width;
    if (index >= total) return;
    const size_t request = index / (2u * width);
    const size_t part = (index / width) & 1u;
    const size_t column = index % width;
    const size_t source = positions[request] * cache_strides[request] + column;
    const uint16_t *row = part == 0u ? kcache[request] : vcache[request];
    out[index] = cuda_bf16_to_float(row[source]);
}

extern "C" int mynah_cuda_gather_kv_batch(
    void *opaque, float *const *kcache, float *const *vcache,
    const size_t *positions, const size_t *cache_strides, size_t batch,
    size_t heads, size_t head_width, float *out, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || kcache == nullptr || vcache == nullptr ||
        positions == nullptr || cache_strides == nullptr || out == nullptr ||
        batch == 0u || batch > st->batch_meta_cap || heads == 0u ||
        head_width == 0u || heads > SIZE_MAX / head_width) {
        set_error(e, ec, "invalid CUDA batched K/V gather dimensions");
        return -1;
    }
    const size_t width = heads * head_width;
    size_t total = 0u;
    size_t blocks = 0u;
    if (!cuda_size_mul(batch, 2u, &total) || !cuda_size_mul(total, width, &total) ||
        !cuda_size_add(total, 255u, &blocks) ||
        (blocks /= 256u) > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA batched K/V gather size overflow");
        return -1;
    }
    for (size_t i = 0; i < batch; ++i) {
        if (kcache[i] == nullptr || vcache[i] == nullptr ||
            cache_strides[i] < width ||
            positions[i] > (SIZE_MAX - (width - 1u)) / cache_strides[i]) {
            set_error(e, ec, "invalid CUDA batched K/V gather cache");
            return -1;
        }
    }
    if (ce(cudaMemcpyAsync(st->dev_batch_k_cache, kcache,
                           batch * sizeof(*kcache), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_v_cache, vcache,
                           batch * sizeof(*vcache), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_positions, positions,
                           batch * sizeof(*positions), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_cache_strides, cache_strides,
                           batch * sizeof(*cache_strides),
                           cudaMemcpyHostToDevice, st->stream), e, ec)) {
        return -1;
    }
    k_gather_kv_batch<<<(int)blocks, 256, 0, st->stream>>>(
        st->dev_batch_k_cache, st->dev_batch_v_cache, st->dev_batch_positions,
        st->dev_batch_cache_strides, (int)batch, width, out);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_gather_kv_bf16_batch(
    void *opaque, void *const *kcache, void *const *vcache,
    const size_t *positions, const size_t *cache_strides, size_t batch,
    size_t heads, size_t head_width, float *out, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || kcache == nullptr || vcache == nullptr ||
        positions == nullptr || cache_strides == nullptr || out == nullptr ||
        batch == 0u || batch > st->batch_meta_cap || heads == 0u ||
        head_width == 0u || heads > SIZE_MAX / head_width) {
        set_error(e, ec, "invalid CUDA BF16 batched K/V gather dimensions");
        return -1;
    }
    const size_t width = heads * head_width;
    size_t total = 0u;
    size_t blocks = 0u;
    if (!cuda_size_mul(batch, 2u, &total) || !cuda_size_mul(total, width, &total) ||
        !cuda_size_add(total, 255u, &blocks) || (blocks /= 256u) > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA BF16 batched K/V gather size overflow");
        return -1;
    }
    for (size_t i = 0; i < batch; ++i) {
        if (kcache[i] == nullptr || vcache[i] == nullptr ||
            cache_strides[i] < width ||
            positions[i] > (SIZE_MAX - (width - 1u)) / cache_strides[i]) {
            set_error(e, ec, "invalid CUDA BF16 batched K/V gather cache");
            return -1;
        }
    }
    if (ce(cudaMemcpyAsync(st->dev_batch_k_cache, kcache,
                           batch * sizeof(*kcache), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_v_cache, vcache,
                           batch * sizeof(*vcache), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_positions, positions,
                           batch * sizeof(*positions), cudaMemcpyHostToDevice,
                           st->stream), e, ec) ||
        ce(cudaMemcpyAsync(st->dev_batch_cache_strides, cache_strides,
                           batch * sizeof(*cache_strides), cudaMemcpyHostToDevice,
                           st->stream), e, ec)) return -1;
    k_gather_kv_bf16_batch<<<(int)blocks, 256, 0, st->stream>>>(
        reinterpret_cast<const uint16_t *const *>(st->dev_batch_k_cache),
        reinterpret_cast<const uint16_t *const *>(st->dev_batch_v_cache),
        st->dev_batch_positions, st->dev_batch_cache_strides, (int)batch,
        width, out);
    return ce(cudaGetLastError(), e, ec);
}

/* MYNAH_CUDA_BF16_FUSE: each fused kernel against the unfused kernel
 * sequence it replaces, on the same inputs, compared BIT FOR BIT (memcmp, no
 * tolerance).  This is the claim the fused decode path rests on: a failure
 * here means the compiler contracted or reordered one of the two differently
 * and the fused path is not a drop-in replacement. */
static int cuda_bf16_fuse_self_test(cuda_backend_state *st, char *e,
                                    size_t ec) {
    if (st == nullptr) return -1;
    const int rows = 5, heads = 2, head_width = 32;
    const int width = heads * head_width; /* hidden == attention width */
    const int qkv_cols = 3 * width;
    const int ffn = 2 * width;
    const int big = rows * qkv_cols; /* largest tensor of the test */
    std::vector<float> host((size_t)big * 4u);
    uint32_t seed = 0x9e3779b9u;
    for (auto &v : host) {
        seed = seed * 1664525u + 1013904223u;
        v = ((float)(seed >> 8) / 16777216.0f - 0.5f) * 6.0f;
    }
    const size_t positions[5] = {0u, 3u, 17u, 250u, 1499u};
    /* [0] input, [1] bias/gain, [2] second operand, [3] scratch, [4] a,
     * [5] b; bf16 buffers [6] and [7]; positions at the end. */
    float *d = nullptr;
    const size_t slab = (size_t)big;
    if (ce(cudaMalloc((void **)&d, slab * 8u * sizeof(float) +
                                       sizeof(positions)), e, ec) != 0)
        return -1;
    float *in = d, *bias = d + slab, *other = d + 2u * slab,
          *tmp = d + 3u * slab, *a = d + 4u * slab, *b = d + 5u * slab;
    uint16_t *ha = reinterpret_cast<uint16_t *>(d + 6u * slab);
    uint16_t *hb = reinterpret_cast<uint16_t *>(d + 7u * slab);
    size_t *d_positions = reinterpret_cast<size_t *>(d + 8u * slab);
    std::vector<unsigned char> x((size_t)big * sizeof(float));
    std::vector<unsigned char> y((size_t)big * sizeof(float));
    int result = 0;
    auto same = [&](const void *p, const void *q, size_t bytes,
                    const char *what) -> int {
        if (ce(cudaStreamSynchronize(st->stream), e, ec) != 0 ||
            ce(cudaMemcpy(x.data(), p, bytes, cudaMemcpyDeviceToHost), e,
               ec) != 0 ||
            ce(cudaMemcpy(y.data(), q, bytes, cudaMemcpyDeviceToHost), e,
               ec) != 0)
            return -1;
        if (std::memcmp(x.data(), y.data(), bytes) != 0) {
            std::snprintf(e, ec, "BF16 fused %s differs from the unfused path",
                          what);
            return -1;
        }
        return 0;
    };
    const int blocks_qkv = (big + 255) / 256;
    const int n_ffn = rows * ffn, n_hidden = rows * width;
    do {
        if (ce(cudaMemcpy(in, host.data(), slab * sizeof(float),
                          cudaMemcpyHostToDevice), e, ec) != 0 ||
            ce(cudaMemcpy(bias, host.data() + slab, slab * sizeof(float),
                          cudaMemcpyHostToDevice), e, ec) != 0 ||
            ce(cudaMemcpy(other, host.data() + 2u * slab,
                          slab * sizeof(float), cudaMemcpyHostToDevice), e,
               ec) != 0 ||
            ce(cudaMemcpy(d_positions, positions, sizeof(positions),
                          cudaMemcpyHostToDevice), e, ec) != 0) {
            result = -1;
            break;
        }
        /* LayerNorm (gain = bias slab, bias = other slab). */
        k_layer_norm<<<rows, 256, 0, st->stream>>>(tmp, in, bias, other, width,
                                                   1e-5f, rows);
        k_f32_to_bf16<<<(n_hidden + 255) / 256, 256, 0, st->stream>>>(
            tmp, ha, n_hidden);
        k_layer_norm_bf16<<<rows, 256, 0, st->stream>>>(hb, in, bias, other,
                                                        width, 1e-5f, rows);
        if (ce(cudaGetLastError(), e, ec) != 0 ||
            same(ha, hb, (size_t)n_hidden * sizeof(uint16_t),
                 "layer norm") != 0) {
            result = -1;
            break;
        }
        /* Bias + GELU -> BF16. */
        cudaMemcpyAsync(tmp, in, (size_t)n_ffn * sizeof(float),
                        cudaMemcpyDeviceToDevice, st->stream);
        k_bias_add<<<(n_ffn + 255) / 256, 256, 0, st->stream>>>(tmp, bias,
                                                               rows, ffn);
        k_gelu<<<(n_ffn + 255) / 256, 256, 0, st->stream>>>(tmp, n_ffn);
        k_f32_to_bf16<<<(n_ffn + 255) / 256, 256, 0, st->stream>>>(tmp, ha,
                                                                  n_ffn);
        k_bias_gelu_bf16<<<(n_ffn + 255) / 256, 256, 0, st->stream>>>(
            in, bias, hb, rows, ffn);
        if (ce(cudaGetLastError(), e, ec) != 0 ||
            same(ha, hb, (size_t)n_ffn * sizeof(uint16_t), "bias+GELU") != 0) {
            result = -1;
            break;
        }
        /* QKV bias + RoPE. */
        const int pairs = rows * heads * (head_width / 2);
        cudaMemcpyAsync(a, in, (size_t)big * sizeof(float),
                        cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(b, in, (size_t)big * sizeof(float),
                        cudaMemcpyDeviceToDevice, st->stream);
        k_bias_add<<<blocks_qkv, 256, 0, st->stream>>>(a, bias, rows, qkv_cols);
        k_rope_qk_batch<<<(pairs + 255) / 256, 256, 0, st->stream>>>(
            a, d_positions, rows, heads, head_width, 10000.0f);
        k_rope_qk_bias_batch<<<(pairs + 255) / 256, 256, 0, st->stream>>>(
            b, bias, d_positions, rows, heads, head_width, 10000.0f);
        if (ce(cudaGetLastError(), e, ec) != 0 ||
            same(a, b, (size_t)big * sizeof(float), "QKV bias+RoPE") != 0) {
            result = -1;
            break;
        }
        /* GEMM output bias + residual. */
        cudaMemcpyAsync(a, other, (size_t)n_hidden * sizeof(float),
                        cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(b, other, (size_t)n_hidden * sizeof(float),
                        cudaMemcpyDeviceToDevice, st->stream);
        cudaMemcpyAsync(tmp, in, (size_t)n_hidden * sizeof(float),
                        cudaMemcpyDeviceToDevice, st->stream);
        k_bias_add<<<(n_hidden + 255) / 256, 256, 0, st->stream>>>(
            tmp, bias, rows, width);
        k_residual_add<<<(n_hidden + 255) / 256, 256, 0, st->stream>>>(
            a, tmp, n_hidden);
        k_residual_bias_add<<<(n_hidden + 255) / 256, 256, 0, st->stream>>>(
            b, in, bias, rows, width);
        if (ce(cudaGetLastError(), e, ec) != 0 ||
            same(a, b, (size_t)n_hidden * sizeof(float),
                 "bias+residual") != 0) {
            result = -1;
            break;
        }
    } while (false);
    cudaFree(d);
    return result;
}

extern "C" int mynah_cuda_rope_dev(void *opaque, float *qkv,
                                    size_t position, size_t heads,
                                    size_t head_width, float max_period,
                                    char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (qkv == nullptr || heads == 0u || head_width == 0u ||
        (head_width & 1u) != 0u || heads > (size_t)INT_MAX ||
        head_width > (size_t)INT_MAX || !(max_period > 0.0f) ||
        !isfinite(max_period)) {
        set_error(e, ec, "invalid CUDA RoPE dimensions");
        return -1;
    }
    const size_t half = head_width / 2u;
    if (half == 0u || heads > SIZE_MAX / half) {
        set_error(e, ec, "CUDA RoPE dimensions overflow");
        return -1;
    }
    const size_t pairs = heads * half;
    if (pairs > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA RoPE dimensions overflow launch range");
        return -1;
    }
    k_rope_qk<<<((int)pairs + 255) / 256, 256, 0, st->stream>>>(
        qkv, position, (int)heads, (int)head_width, max_period);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_rope_batch_dev(
    void *opaque, float *qkv, const size_t *positions, size_t batch,
    size_t heads, size_t head_width, float max_period, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || qkv == nullptr || positions == nullptr || batch == 0u ||
        batch > st->batch_meta_cap || heads == 0u || head_width == 0u ||
        (head_width & 1u) != 0u || heads > (size_t)INT_MAX ||
        head_width > (size_t)INT_MAX || !(max_period > 0.0f) ||
        !isfinite(max_period)) {
        set_error(e, ec, "invalid CUDA batched RoPE dimensions");
        return -1;
    }
    const size_t half = head_width / 2u;
    size_t pairs = 0u;
    size_t blocks = 0u;
    if (!cuda_size_mul(batch, heads, &pairs) ||
        !cuda_size_mul(pairs, half, &pairs) ||
        !cuda_size_add(pairs, 255u, &blocks) ||
        (blocks /= 256u) > (size_t)INT_MAX) {
        set_error(e, ec, "CUDA batched RoPE size overflow");
        return -1;
    }
    if (ce(cudaMemcpyAsync(st->dev_batch_positions, positions,
                           batch * sizeof(*positions), cudaMemcpyHostToDevice,
                           st->stream), e, ec)) return -1;
    k_rope_qk_batch<<<(int)blocks, 256, 0, st->stream>>>(
        qkv, st->dev_batch_positions, (int)batch, (int)heads,
        (int)head_width, max_period);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_rope_bias_batch_dev(
    void *opaque, float *qkv, const float *bias, const size_t *positions,
    size_t batch, size_t heads, size_t head_width, float max_period, char *e,
    size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (bias == nullptr)
        return mynah_cuda_rope_batch_dev(opaque, qkv, positions, batch, heads,
                                         head_width, max_period, e, ec);
    if (st == nullptr || qkv == nullptr || positions == nullptr || batch == 0u ||
        batch > st->batch_meta_cap || heads == 0u || head_width == 0u ||
        (head_width & 1u) != 0u || heads > (size_t)INT_MAX ||
        head_width > (size_t)INT_MAX || !(max_period > 0.0f) ||
        !isfinite(max_period)) {
        set_error(e, ec, "invalid CUDA batched RoPE dimensions");
        return -1;
    }
    const size_t half = head_width / 2u;
    size_t pairs = 0u;
    size_t blocks = 0u;
    size_t bias_n = 0u;
    if (!cuda_size_mul(batch, heads, &pairs) ||
        !cuda_size_mul(pairs, half, &pairs) ||
        !cuda_size_add(pairs, 255u, &blocks) ||
        (blocks /= 256u) > (size_t)INT_MAX ||
        !cuda_size_mul(heads, head_width, &bias_n) ||
        !cuda_size_mul(bias_n, 3u * sizeof(float), &bias_n)) {
        set_error(e, ec, "CUDA batched RoPE size overflow");
        return -1;
    }
    float *d_bias = nullptr;
    if (cached_weight(st, bias, bias_n, &d_bias, e, ec)) return -1;
    if (ce(cudaMemcpyAsync(st->dev_batch_positions, positions,
                           batch * sizeof(*positions), cudaMemcpyHostToDevice,
                           st->stream), e, ec)) return -1;
    k_rope_qk_bias_batch<<<(int)blocks, 256, 0, st->stream>>>(
        qkv, d_bias, st->dev_batch_positions, (int)batch, (int)heads,
        (int)head_width, max_period);
    return ce(cudaGetLastError(), e, ec);
}

/* GELU on host data: upload → kernel → download → sync. */
extern "C" int mynah_cuda_gelu_host(void *opaque, float *data, size_t n,
                          char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t bytes = n * sizeof(float);
    if (ensure_scratch(st, bytes, e, ec)) return -1;
    float *d = st->dev_scratch;
    if (ce(cudaMemcpyAsync(d, data, bytes, cudaMemcpyHostToDevice, st->stream), e, ec)) return -1;
    k_gelu<<<((int)n+255)/256, 256, 0, st->stream>>>(d, (int)n);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (ce(cudaMemcpyAsync(data, d, bytes, cudaMemcpyDeviceToHost, st->stream), e, ec)) return -1;
    return ce(cudaStreamSynchronize(st->stream), e, ec);
}

/* GELU in double precision for CPU-matching accuracy.
 * Avoids FP16/FMA precision drift in autoregressive loops. */
__global__ static void k_gelu_f64(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = (double)data[i];
        double cubic = x * x * x;
        double inner = 0.7978845608028654 * (x + 0.044715 * cubic);
        double t = tanh(inner);
        data[i] = (float)(0.5 * x * (1.0 + t));
    }
}

extern "C" int mynah_cuda_gelu_host_f64(void *opaque, float *data, size_t n,
                              char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t bytes = n * sizeof(float);
    if (ensure_scratch(st, bytes, e, ec)) return -1;
    float *d = st->dev_scratch;
    if (ce(cudaMemcpyAsync(d, data, bytes, cudaMemcpyHostToDevice, st->stream), e, ec)) return -1;
    k_gelu_f64<<<((int)n+255)/256, 256, 0, st->stream>>>(d, (int)n);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (ce(cudaMemcpyAsync(data, d, bytes, cudaMemcpyDeviceToHost, st->stream), e, ec)) return -1;
    return ce(cudaStreamSynchronize(st->stream), e, ec);
}

/* ---- CUDA Graph cache for matmul segments ---- */
static void destroy_graphs(cuda_backend_state *st) {
    if (st == nullptr) return;
    for (auto &entry : st->graph_cache) {
        if (entry.exec != nullptr) cudaGraphExecDestroy(entry.exec);
        if (entry.graph != nullptr) cudaGraphDestroy(entry.graph);
        entry.exec = nullptr;
        entry.graph = nullptr;
        entry.valid = false;
    }
    st->graph_cache.clear();
    for (auto &entry : st->pipeline_graphs) {
        if (entry.exec != nullptr) cudaGraphExecDestroy(entry.exec);
        if (entry.graph != nullptr) cudaGraphDestroy(entry.graph);
        entry.exec = nullptr;
        entry.graph = nullptr;
        entry.valid = false;
        entry.capturing = false;
    }
    st->pipeline_graphs.clear();
}

static cuda_pipeline_graph_entry *find_pipeline_graph(
    cuda_backend_state *st, size_t key, const void *identity) {
    for (auto &entry : st->pipeline_graphs) {
        if (entry.key == key && entry.identity == identity) return &entry;
    }
    return nullptr;
}

extern "C" int mynah_cuda_graph_begin(void *opaque, size_t key,
                                      const void *identity, int *replay,
                                      char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (replay != nullptr) *replay = 0;
    if (st == nullptr || identity == nullptr || key == 0u) return 1;
    if (!st->graphs_enabled) {
        st->graph_fallbacks.fetch_add(1ull, std::memory_order_relaxed);
        return 1;
    }
    cuda_pipeline_graph_entry *entry =
        find_pipeline_graph(st, key, identity);
    if (entry != nullptr && entry->valid) {
        if (replay != nullptr) *replay = 1;
        st->graph_replays.fetch_add(1ull, std::memory_order_relaxed);
        return 0;
    }
    if (entry != nullptr && entry->capturing) {
        set_error(e, ec, "CUDA graph capture already active");
        return -1;
    }
    if (st->pipeline_graphs.size() >= CUDA_PIPELINE_GRAPH_CAP) {
        st->graph_fallbacks.fetch_add(1ull, std::memory_order_relaxed);
        return 1;
    }
    if (st->cublas_workspace == nullptr) {
        st->cublas_workspace_cap = 8u * 1024u * 1024u;
        if (ce(cudaMalloc(&st->cublas_workspace, st->cublas_workspace_cap), e,
               ec)) return 1;
    }
    if (cbe(cublasSetWorkspace(st->cublas, st->cublas_workspace,
                               st->cublas_workspace_cap), e, ec) ||
        cbe(cublasSetStream(st->cublas, st->stream), e, ec) ||
        ce(cudaStreamBeginCapture(st->stream, cudaStreamCaptureModeRelaxed), e,
           ec)) {
        return 1;
    }
    cuda_pipeline_graph_entry created{};
    created.key = key;
    created.identity = identity;
    created.valid = false;
    created.capturing = true;
    st->pipeline_graphs.push_back(created);
    return 0;
}

extern "C" int mynah_cuda_graph_end(void *opaque, size_t key,
                                    const void *identity, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || identity == nullptr || key == 0u) return 1;
    cuda_pipeline_graph_entry *entry =
        find_pipeline_graph(st, key, identity);
    if (entry == nullptr || !entry->capturing) return 1;
    cudaGraph_t graph = nullptr;
    const cudaError_t capture_status = cudaStreamEndCapture(st->stream, &graph);
    entry->capturing = false;
    if (capture_status != cudaSuccess || graph == nullptr) {
        if (graph != nullptr) cudaGraphDestroy(graph);
        if (capture_status != cudaSuccess) ce(capture_status, e, ec);
        st->pipeline_graphs.pop_back();
        st->graph_fallbacks.fetch_add(1ull, std::memory_order_relaxed);
        return 1;
    }
    cudaGraphExec_t exec = nullptr;
    const cudaError_t instantiate_status =
        cudaGraphInstantiate(&exec, graph, 0);
    if (instantiate_status != cudaSuccess || exec == nullptr) {
        cudaGraphDestroy(graph);
        if (instantiate_status != cudaSuccess) ce(instantiate_status, e, ec);
        st->pipeline_graphs.pop_back();
        st->graph_fallbacks.fetch_add(1ull, std::memory_order_relaxed);
        return 1;
    }
    entry->graph = graph;
    entry->exec = exec;
    entry->valid = true;
    st->graph_captures.fetch_add(1ull, std::memory_order_relaxed);
    return 0;
}

extern "C" int mynah_cuda_graph_launch(void *opaque, size_t key,
                                       const void *identity, char *e,
                                       size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || identity == nullptr || key == 0u) return -1;
    cuda_pipeline_graph_entry *entry =
        find_pipeline_graph(st, key, identity);
    if (entry == nullptr || !entry->valid || entry->exec == nullptr) {
        set_error(e, ec, "CUDA graph is not instantiated");
        return -1;
    }
    return ce(cudaGraphLaunch(entry->exec, st->stream), e, ec);
}

extern "C" void mynah_cuda_graph_abort(void *opaque, size_t key,
                                        const void *identity) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || identity == nullptr || key == 0u) return;
    cuda_pipeline_graph_entry *entry =
        find_pipeline_graph(st, key, identity);
    if (entry == nullptr || !entry->capturing) return;
    cudaGraph_t graph = nullptr;
    (void)cudaStreamEndCapture(st->stream, &graph);
    if (graph != nullptr) cudaGraphDestroy(graph);
    st->pipeline_graphs.pop_back();
}

extern "C" void mynah_cuda_graph_forget(void *opaque, const void *identity) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || identity == nullptr) return;
    (void)cudaStreamSynchronize(st->stream);
    for (size_t i = 0; i < st->pipeline_graphs.size();) {
        cuda_pipeline_graph_entry &entry = st->pipeline_graphs[i];
        if (entry.identity != identity) {
            ++i;
            continue;
        }
        if (entry.exec != nullptr) cudaGraphExecDestroy(entry.exec);
        if (entry.graph != nullptr) cudaGraphDestroy(entry.graph);
        st->pipeline_graphs.erase(st->pipeline_graphs.begin() + i);
    }
    /* A pooled Pocket decoder (MYNAH_CUDA_SLOT_POOL) is parked instead of
     * closed, so forgetting it must also drop the cross-request batch graphs
     * that name it, as decoder_close does.  Any other identity (a scratch
     * arena) matches no decoder and this is a no-op. */
    destroy_decoder_batch_graphs_for(
        st, static_cast<const mynah_backend_decoder *>(identity));
}

/* MYNAH_CUDA_DEFERRED_RELEASE: graph_forget for a decoder being parked in the
 * engine's slot pool while work queued earlier may still run.  The stream
 * sync above exists for what forget destroys; when nothing is destroyed --
 * no single-request graph under this identity, and every batch graph naming
 * the decoder is kept for reuse with only its row list cleared (host
 * vectors) -- there is nothing to wait for.  Otherwise this is graph_forget,
 * sync included. */
extern "C" void mynah_cuda_graph_forget_parked(void *opaque,
                                                const void *identity) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr || identity == nullptr) return;
    bool destroys = !cuda_decoder_graph_reuse_enabled();
    for (const cuda_pipeline_graph_entry &entry : st->pipeline_graphs)
        destroys = destroys || entry.identity == identity;
    for (const cuda_decoder_batch_graph_entry *entry : st->decoder_batch_graphs) {
        if (destroys) break;
        if (entry->valid && entry->done != nullptr) continue;
        for (const mynah_backend_decoder *row : entry->decoders)
            destroys = destroys || row == identity;
    }
    if (destroys) {
        mynah_cuda_graph_forget(opaque, identity);
        return;
    }
    destroy_decoder_batch_graphs_for(
        st, static_cast<const mynah_backend_decoder *>(identity));
}

/* MYNAH_CUDA_DEFERRED_RELEASE: a point in the stream (an event recorded
 * after everything queued so far), waited on later instead of now.  Null if
 * the event cannot be made; the caller then drains as before. */
extern "C" void *mynah_cuda_fence_record(void *opaque) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (st == nullptr) return nullptr;
    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess)
        return nullptr;
    if (cudaEventRecord(event, st->stream) != cudaSuccess) {
        cudaEventDestroy(event);
        return nullptr;
    }
    return static_cast<void *>(event);
}

extern "C" void mynah_cuda_fence_wait(void *opaque, void *fence) {
    (void)opaque;
    if (fence == nullptr) return;
    cudaEvent_t event = static_cast<cudaEvent_t>(fence);
    (void)cudaEventSynchronize(event);
    cudaEventDestroy(event);
}

/* MYNAH_CUDA_DECODE_OVERLAP: poll a fence without waiting or releasing it. */
extern "C" int mynah_cuda_fence_query(void *opaque, void *fence) {
    (void)opaque;
    if (fence == nullptr) return 1;
    const cudaError_t status = cudaEventQuery(static_cast<cudaEvent_t>(fence));
    if (status == cudaSuccess) return 1;
    if (status == cudaErrorNotReady) {
        (void)cudaGetLastError();   /* not an error; do not leave it pending */
        return 0;
    }
    return -1;
}

/* Wait for a fence and release it, reporting a device error. */
extern "C" int mynah_cuda_fence_sync(void *opaque, void *fence, char *e,
                                     size_t ec) {
    (void)opaque;
    if (fence == nullptr) return 0;
    cudaEvent_t event = static_cast<cudaEvent_t>(fence);
    const cudaError_t status = cudaEventSynchronize(event);
    cudaEventDestroy(event);
    return ce(status, e, ec);
}

static cuda_graph_entry *find_graph(cuda_backend_state *st, size_t rows,
                                    size_t iw, size_t ow,
                                    const void *weight, const void *bias) {
    for (auto &entry : st->graph_cache) {
        if (entry.valid && entry.rows == rows && entry.iw == iw && entry.ow == ow &&
            entry.weight_pointer == weight && entry.bias_pointer == bias)
            return &entry;
    }
    return nullptr;
}

/* Matmul with CUDA Graph: capture on first call, replay on subsequent.
 * Falls back to regular matmul if capture fails. */
extern "C" int mynah_cuda_matmul_graph(void *opaque, const float *input, float *output,
                             size_t rows, size_t iw, size_t ow,
                             const float *weight, const float *bias,
                             char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t in_n = 0, out_n = 0, w_n = 0, total_n = 0;
    if (st == nullptr || validate_cuda_matmul(input, output, weight, rows, iw, ow,
                                               &in_n, &out_n, &w_n, &total_n,
                                               e, ec) != 0) return -1;
    /* The captured graph below intentionally uses the FP16/Tensor-Core
     * pipeline.  Keep the default parity mode on the uncaptured FP32 path;
     * operators opt into this graph together with MYNAH_CUDA_FAST_MATH=1. */
    if (!st->fast_math) return cuda_matmul(opaque, input, output, rows, iw, ow,
                                           weight, bias, e, ec);
    const size_t mapped_need = total_n * sizeof(float);
    if (ensure_host(st, mapped_need, e, ec)) return -1;

    /* Try to find a cached graph. */
    cuda_graph_entry *entry = find_graph(st, rows, iw, ow, weight, bias);
    if (entry && entry->valid) {
        /* Update input in mapped buffer, replay graph. */
        std::memcpy(st->host_buf, input, in_n * sizeof(float));
        if (ce(cudaGraphLaunch(entry->exec, st->stream), e, ec)) return -1;
        if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
        std::memcpy(output, st->host_buf + in_n, out_n * sizeof(float));
        return 0;
    }

    /* First call: capture the graph. */
    half *dw16 = nullptr;
    if (cached_weight_fp16(st, weight, w_n, &dw16, e, ec)) return -1;
    float *db = nullptr;
    if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
    if (ensure_scratch(st, in_n * sizeof(half), e, ec)) return -1;
    half *di16 = (half *)st->dev_scratch;
    float *d_out_mapped = st->dev_buf + in_n;
    std::memcpy(st->host_buf, input, in_n * sizeof(float));

    /* Capture. Alpha/beta must be static (not stack) for graph capture.
     * Pre-allocate cuBLAS workspace to avoid allocation during capture. */
    static const float g_alpha = 1.0f, g_beta = 0.0f;
    if (st->cublas_workspace == nullptr) {
        st->cublas_workspace_cap = 4u * 1024u * 1024u;
        if (ce(cudaMalloc(&st->cublas_workspace, st->cublas_workspace_cap), e, ec)) return -1;
    }
    if (cbe(cublasSetWorkspace(st->cublas, st->cublas_workspace,
                               st->cublas_workspace_cap), e, ec) ||
        cbe(cublasSetStream(st->cublas, st->stream), e, ec) ||
        ce(cudaStreamBeginCapture(st->stream, cudaStreamCaptureModeRelaxed), e, ec)) {
        return cuda_matmul(opaque, input, output, rows, iw, ow, weight, bias, e, ec);
    }
    k_f32_to_f16<<<((int)in_n+255)/256, 256, 0, st->stream>>>(st->dev_buf, di16, (int)in_n);
    const cublasStatus_t gemm_status = cublasGemmEx(
        st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        (int)ow, (int)rows, (int)iw,
        &g_alpha, dw16, CUDA_R_16F, (int)iw,
        di16, CUDA_R_16F, (int)iw,
        &g_beta, d_out_mapped, CUDA_R_32F, (int)ow,
        cuda_compute_type(st), cuda_gemm_algo(st));
    if (db) {
        k_bias_add<<<((int)(rows*ow)+255)/256, 256, 0, st->stream>>>(
            d_out_mapped, db, (int)rows, (int)ow);
    }
    cudaGraph_t graph;
    cudaError_t cap_err = cudaStreamEndCapture(st->stream, &graph);
    if (gemm_status != CUBLAS_STATUS_SUCCESS || cap_err != cudaSuccess || graph == nullptr) {
        /* Capture failed — fall back to regular matmul. */
        return cuda_matmul(opaque, input, output, rows, iw, ow, weight, bias, e, ec);
    }

    cudaGraphExec_t exec;
    if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        cudaGraphDestroy(graph);
        return cuda_matmul(opaque, input, output, rows, iw, ow, weight, bias, e, ec);
    }

    /* Cache the graph. */
    if (st->graph_cache.size() < 16u) {
        st->graph_cache.push_back({rows, iw, ow, weight, bias, graph, exec, true});
    } else {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        return cuda_matmul(opaque, input, output, rows, iw, ow, weight, bias, e, ec);
    }

    /* Replay for this call. */
    if (ce(cudaGraphLaunch(exec, st->stream), e, ec)) return -1;
    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    std::memcpy(output, st->host_buf + in_n, out_n * sizeof(float));
    return 0;
}

/* MYNAH_CUDA_KV_VMM self-test, always run by --gpu-self-test when the device
 * supports virtual memory management (skipped otherwise, flag-independent).
 * Part 1, the range: a mapped prefix survives growth in place (same base,
 * old bytes intact, new chunk writable), a trim keeps what is still covered,
 * a resize past the reservation is refused, and mynah_cuda_dev_free releases
 * a range. Part 2, the layout: the decode attention kernels (fast, split,
 * legacy, with shared prefix tables) and the K/V gather give bit-identical
 * results on a plane-major cache (stride = width) and on the same values laid
 * out position-major in a VMM range (stride = layers * 2 * width, per-layer
 * bases inside the position, a row without its prefix biased below its
 * start), and the new K/V lands on the same values in both. */
static int cuda_kv_vmm_layout_case(cuda_backend_state *st, char *e, size_t ec) {
    constexpr int heads = 4, hw = 64, layers = 3, rows = 3;
    constexpr size_t width = (size_t)heads * hw;
    constexpr size_t position_elems = (size_t)layers * 2u * width;
    const size_t positions[rows] = {5u, 130u, 300u};
    const size_t skip[rows] = {0u, 3u, 0u}; /* row 1 reads [0, 3) from the prefix */
    const float scale = 0.125f;
    uint32_t seed = 0x6d2b79f5u;
    auto uniform = [&](float span) {
        seed = seed * 1664525u + 1013904223u;
        return ((float)(seed >> 8) / 16777216.0f - 0.5f) * 2.0f * span;
    };
    auto bf16 = [](float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return (uint16_t)(bits >> 16);
    };
    /* Plane-major rows [layer][K|V][cap][width] in one cudaMalloc slab;
     * position-major rows [cap - skip][layer][K|V][width] in one VMM range,
     * the row without its prefix first so its biased base points below the
     * range. The prefix planes [layer][K|V][3][width] follow the plane slab. */
    size_t cap[rows], plane_off[rows], vmm_off[rows], plane_total = 0u, vmm_total = 0u;
    const int vmm_order[rows] = {1, 0, 2};
    for (int r = 0; r < rows; ++r) {
        cap[r] = positions[r] + 1u;
        plane_off[r] = plane_total;
        plane_total += (size_t)layers * 2u * cap[r] * width;
    }
    for (int k = 0; k < rows; ++k) {
        const int r = vmm_order[k];
        vmm_off[r] = vmm_total;
        vmm_total += (cap[r] - skip[r]) * position_elems;
    }
    const size_t prefix_n = 3u;
    const size_t prefix_off = plane_total;
    const size_t slab_total = plane_total + (size_t)layers * 2u * prefix_n * width;
    std::vector<uint16_t> plane(slab_total), vmm(vmm_total, 0x7fc0u);
    for (int r = 0; r < rows; ++r)
        for (int l = 0; l < layers; ++l)
            for (int kv = 0; kv < 2; ++kv)
                for (size_t p = 0; p < cap[r]; ++p)
                    for (size_t d = 0; d < width; ++d) {
                        const uint16_t x = bf16(uniform(2.0f));
                        plane[plane_off[r] + (((size_t)l * 2u + kv) * cap[r] + p) * width + d] = x;
                        if (p >= skip[r])
                            vmm[vmm_off[r] + (p - skip[r]) * position_elems +
                                ((size_t)l * 2u + kv) * width + d] = x;
                        else
                            plane[prefix_off + (((size_t)l * 2u + kv) * prefix_n + p) * width + d] = x;
                    }
    std::vector<float> qkv((size_t)rows * 3u * width);
    for (auto &x : qkv) x = uniform(3.0f);
    const size_t out_n = (size_t)rows * width;
    const size_t gather_n = (size_t)rows * 2u * width;
    uint16_t *d_plane = nullptr;
    void *d_vmm = nullptr;
    float *d_f = nullptr; /* qkv | 6 attention outputs | 2 gathers */
    void *d_tab = nullptr;
    size_t mapped = 0u;
    int result = -1;
    do {
        if (mynah_cuda_kv_vmm_alloc(st, vmm_total * sizeof(uint16_t),
                                    vmm_total * sizeof(uint16_t), &d_vmm,
                                    &mapped, e, ec) != 0 ||
            ce(cudaMalloc((void **)&d_plane, slab_total * sizeof(uint16_t)), e, ec) ||
            ce(cudaMalloc((void **)&d_f, (qkv.size() + 6u * out_n + 2u * gather_n) *
                                             sizeof(float)), e, ec) ||
            ce(cudaMalloc(&d_tab, (size_t)rows * 16u * sizeof(void *)), e, ec))
            break;
        if (ce(cudaMemcpy(d_plane, plane.data(), slab_total * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), e, ec) ||
            ce(cudaMemcpy(d_vmm, vmm.data(), vmm_total * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), e, ec) ||
            ce(cudaMemcpy(d_f, qkv.data(), qkv.size() * sizeof(float),
                          cudaMemcpyHostToDevice), e, ec))
            break;
        static_assert(sizeof(size_t) == sizeof(void *), "table layout");
        auto **t = static_cast<const uint16_t **>(d_tab);
        /* [0] plane K, [1] plane V, [2] vmm K, [3] vmm V, [4] prefix K,
         * [5] prefix V, [6] positions, [7] plane strides, [8] vmm strides,
         * [9] prefix lengths; `rows` entries each. */
        auto *t_pos = reinterpret_cast<size_t *>(t + 6 * rows);
        auto *t_ps = reinterpret_cast<size_t *>(t + 7 * rows);
        auto *t_vs = reinterpret_cast<size_t *>(t + 8 * rows);
        auto *t_pl = reinterpret_cast<size_t *>(t + 9 * rows);
        float *d_qkv = d_f;
        float *d_out = d_f + qkv.size();
        float *d_gather = d_out + 6u * out_n;
        const uint16_t *vbase = static_cast<const uint16_t *>(d_vmm);
        result = 0;
        for (int l = 0; l < layers && result == 0; l += layers - 1) {
            const uint16_t *h[6][rows];
            size_t hp[rows], hps[rows], hvs[rows];
            for (int r = 0; r < rows; ++r) {
                h[0][r] = d_plane + plane_off[r] + (size_t)l * 2u * cap[r] * width;
                h[1][r] = h[0][r] + cap[r] * width;
                /* Biased below the row by its skip (integer arithmetic). */
                h[2][r] = reinterpret_cast<const uint16_t *>(
                    (uintptr_t)(vbase + vmm_off[r] + (size_t)l * 2u * width) -
                    (uintptr_t)(skip[r] * position_elems * sizeof(uint16_t)));
                h[3][r] = reinterpret_cast<const uint16_t *>(
                    (uintptr_t)h[2][r] + width * sizeof(uint16_t));
                h[4][r] = d_plane + prefix_off + (size_t)l * 2u * prefix_n * width;
                h[5][r] = h[4][r] + prefix_n * width;
                hp[r] = skip[r];
                hps[r] = width;
                hvs[r] = position_elems;
            }
            for (int k = 0; k < 6 && result == 0; ++k)
                if (ce(cudaMemcpy(t + k * rows, h[k], sizeof(h[k]),
                                  cudaMemcpyHostToDevice), e, ec))
                    result = -1;
            if (result != 0 ||
                ce(cudaMemcpy(t_pos, positions, sizeof(positions), cudaMemcpyHostToDevice), e, ec) ||
                ce(cudaMemcpy(t_ps, hps, sizeof(hps), cudaMemcpyHostToDevice), e, ec) ||
                ce(cudaMemcpy(t_vs, hvs, sizeof(hvs), cudaMemcpyHostToDevice), e, ec) ||
                ce(cudaMemcpy(t_pl, hp, sizeof(hp), cudaMemcpyHostToDevice), e, ec)) {
                result = -1;
                break;
            }
            const dim3 grid((unsigned)heads, (unsigned)rows, 1u);
            for (int lay = 0; lay < 2; ++lay) {
                const uint16_t *const *kc = t + (lay == 0 ? 0 : 2) * rows;
                const uint16_t *const *vc = t + (lay == 0 ? 1 : 3) * rows;
                const size_t *str = lay == 0 ? t_ps : t_vs;
                k_self_attention_bf16_batch_fast<true, false>
                    <<<grid, CUDA_ATTN_FAST_THREADS, 0, st->stream>>>(
                    d_qkv, kc, vc, t_pos, str, rows, heads, hw, scale,
                    d_out + (size_t)lay * out_n, t + 4 * rows, t + 5 * rows,
                    t_pl, nullptr);
                k_self_attention_bf16_batch_split<true, false>
                    <<<grid, CUDA_ATTN_SPLIT_THREADS, 0, st->stream>>>(
                    d_qkv, kc, vc, t_pos, str, rows, heads, hw, scale,
                    d_out + (size_t)(2 + lay) * out_n, t + 4 * rows,
                    t + 5 * rows, t_pl, nullptr);
                k_self_attention_bf16_batch<true>
                    <<<grid, attention_threads((size_t)hw), 0, st->stream>>>(
                    d_qkv, kc, vc, t_pos, str, rows, heads, hw, scale,
                    d_out + (size_t)(4 + lay) * out_n, t + 4 * rows,
                    t + 5 * rows, t_pl);
                k_gather_kv_bf16_batch<<<(int)((gather_n + 255u) / 256u), 256, 0,
                                         st->stream>>>(
                    kc, vc, t_pos, str, rows, width,
                    d_gather + (size_t)lay * gather_n);
            }
            if (ce(cudaGetLastError(), e, ec) ||
                ce(cudaStreamSynchronize(st->stream), e, ec)) {
                result = -1;
                break;
            }
            std::vector<float> got(6u * out_n + 2u * gather_n);
            if (ce(cudaMemcpy(got.data(), d_out, got.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), e, ec)) {
                result = -1;
                break;
            }
            const char *names[3] = {"fast", "split", "legacy"};
            for (int k = 0; k < 3 && result == 0; ++k) {
                const float *a = got.data() + (size_t)(2 * k) * out_n;
                if (std::memcmp(a, a + out_n, out_n * sizeof(float)) != 0) {
                    std::snprintf(e, ec,
                                  "KV VMM layout: %s attention differs between "
                                  "plane-major and position-major (layer %d)",
                                  names[k], l);
                    result = -1;
                }
                for (size_t i = 0; i < out_n && result == 0; ++i)
                    if (!std::isfinite(a[i])) {
                        std::snprintf(e, ec,
                                      "KV VMM layout: %s attention read an "
                                      "unwritten slot (layer %d)", names[k], l);
                        result = -1;
                    }
            }
            const float *g = got.data() + 6u * out_n;
            if (result == 0 &&
                std::memcmp(g, g + gather_n, gather_n * sizeof(float)) != 0) {
                std::snprintf(e, ec,
                              "KV VMM layout: K/V gather differs between the "
                              "layouts (layer %d)", l);
                result = -1;
            }
        }
    } while (false);
    if (d_vmm != nullptr) mynah_cuda_kv_vmm_free(st, d_vmm);
    cudaFree(d_plane);
    cudaFree(d_f);
    cudaFree(d_tab);
    return result;
}

static int cuda_kv_vmm_self_test(cuda_backend_state *st, char *e, size_t ec) {
    if (st == nullptr) return -1;
    size_t gran = 0u;
    char why[192];
    why[0] = '\0';
    if (mynah_cuda_kv_vmm_probe(st, &gran, why, sizeof(why)) != 0) {
        if (getenv("MYNAH_CUDA_KV_VMM") != nullptr)
            std::fprintf(stderr, "mynah-tts: KV VMM self-test skipped: %s\n", why);
        return 0;
    }
    void *p = nullptr;
    size_t mapped = 0u;
    if (mynah_cuda_kv_vmm_alloc(st, 4u * gran, gran, &p, &mapped, e, ec) != 0)
        return -1;
    int result = -1;
    std::vector<unsigned char> host(3u * gran);
    do {
        char refused[192];
        if (mapped != gran) {
            set_error(e, ec, "KV VMM self-test: initial mapping size");
            break;
        }
        if (ce(cudaMemsetAsync(p, 0xab, gran, st->stream), e, ec) ||
            mynah_cuda_kv_vmm_resize(st, p, 2u * gran + 1u, &mapped, e, ec) != 0)
            break;
        if (mapped != 3u * gran) {
            set_error(e, ec, "KV VMM self-test: growth mapping size");
            break;
        }
        if (ce(cudaMemsetAsync(static_cast<unsigned char *>(p) + gran, 0xcd,
                               2u * gran, st->stream), e, ec) ||
            ce(cudaStreamSynchronize(st->stream), e, ec) ||
            ce(cudaMemcpy(host.data(), p, 3u * gran, cudaMemcpyDeviceToHost), e, ec))
            break;
        bool ok = true;
        for (size_t i = 0; i < 3u * gran && ok; ++i)
            ok = host[i] == (i < gran ? 0xabu : 0xcdu);
        if (!ok) {
            set_error(e, ec, "KV VMM self-test: bytes lost across growth in place");
            break;
        }
        if (mynah_cuda_kv_vmm_resize(st, p, gran, &mapped, e, ec) != 0) break;
        if (mapped != gran) {
            set_error(e, ec, "KV VMM self-test: trim mapping size");
            break;
        }
        if (ce(cudaMemcpy(host.data(), p, gran, cudaMemcpyDeviceToHost), e, ec)) break;
        for (size_t i = 0; i < gran && ok; ++i) ok = host[i] == 0xabu;
        if (!ok) {
            set_error(e, ec, "KV VMM self-test: bytes lost across a trim");
            break;
        }
        if (mynah_cuda_kv_vmm_resize(st, p, 5u * gran, &mapped, refused,
                                     sizeof(refused)) == 0) {
            set_error(e, ec, "KV VMM self-test: resize past the reservation accepted");
            break;
        }
        result = 0;
    } while (false);
    mynah_cuda_dev_free(st, static_cast<float *>(p)); /* routed to the VMM release */
    if (result == 0 && st->kv_vmm.live.load(std::memory_order_relaxed) != 0u) {
        set_error(e, ec, "KV VMM self-test: dev_free did not release the range");
        result = -1;
    }
    if (result != 0) return -1;
    return cuda_kv_vmm_layout_case(st, e, ec);
}
