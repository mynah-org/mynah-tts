#include "backend.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <new>
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

__global__ static void k_decoder_causal_columns(const float *window,
                                                float *columns, int channels,
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
    columns[index] = source >= 0 && source < window_len
        ? window[(size_t)channel * (size_t)window_len + (size_t)source]
        : 0.0f;
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

__global__ static void k_decoder_causal_columns_batch(
    float *const *windows, float *const *columns, int batch, int channels,
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
    columns[request][local] = source >= 0 && source < window_len
        ? windows[request][(size_t)channel * (size_t)window_len +
                           (size_t)source]
        : 0.0f;
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

struct cuda_backend_state;
static void destroy_graphs(cuda_backend_state *st);

struct cuda_backend_state {
    cublasHandle_t cublas;
    cudaStream_t stream;
    std::vector<cuda_cached_buffer> weights;
    std::vector<cuda_cached_fp16> weights_fp16;
    std::vector<cuda_cached_int8> weights_int8;
    /* Device scratch for activations (grows on demand). */
    float *dev_scratch;
    size_t dev_scratch_cap;   /* bytes */
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
    /* Pointer tables for the cross-request causal decoder.  Each table is
     * reused for one topology operation at a time; decoder state itself stays
     * in the per-request objects. */
    float **dev_decoder_ptr0;
    float **dev_decoder_ptr1;
    float **dev_decoder_ptr2;
    float **dev_decoder_ptr3;
    size_t batch_meta_cap;
    bool fast_math;
    bool graphs_enabled;
    std::vector<cuda_graph_entry> graph_cache;
    std::vector<cuda_pipeline_graph_entry> pipeline_graphs;
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
    std::atomic<unsigned long long> decoder_batch_frames;
    std::atomic<unsigned long long> decoder_failures;
    std::atomic<unsigned long long> resident_fallbacks;
    std::atomic<unsigned long long> matmul_calls;
    std::atomic<unsigned long long> matvec_calls;
    std::atomic<unsigned long long> q8_matmul_calls;
    std::atomic<unsigned long long> q8_rows;
    std::atomic<unsigned long long> q8_weight_uploads;
    std::atomic<unsigned long long> q8_weight_bytes;
    std::atomic<unsigned long long> q8_activation_bytes;
    bool q8_enabled;
    bool decoder_batch_enabled;
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
};

static constexpr size_t CUDA_BATCH_META_CAP = 64u;
/* A server can retain one decoder graph per live context in addition to the
 * width-bucketed backbone/flow graphs.  The condition-projection input graph
 * is a separate bucket because its input is already resident in `cuda_x` and
 * must not accidentally replay an H2D node from the ordinary bucket.  Keep
 * enough finite room for 128 request decoders plus the three 64-width bucket
 * families. */
static constexpr size_t CUDA_PIPELINE_GRAPH_CAP = 384u;

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

static bool cuda_env_enabled(const char *name, bool fallback) {
    const char *value = std::getenv(name);
    return value == nullptr ? fallback : std::strcmp(value, "0") != 0;
}

static bool cuda_decoder_batch_enabled(void) {
    return cuda_env_enabled("MYNAH_CUDA_DECODER_BATCH", true);
}

static bool cuda_graphs_enabled(void) {
    const char *value = std::getenv("MYNAH_CUDA_GRAPHS");
    return value == nullptr || std::strcmp(value, "0") != 0;
}

static cublasComputeType_t cuda_compute_type(const cuda_backend_state *st) {
    return st->fast_math ? CUBLAS_COMPUTE_32F_FAST_16F
                                    : CUBLAS_COMPUTE_32F;
}

static cublasGemmAlgo_t cuda_gemm_algo(const cuda_backend_state *st) {
    return st->fast_math ? CUBLAS_GEMM_DEFAULT_TENSOR_OP
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
        fabsf(causal_long_partial[0] - 15.0f) > 2.0e-4f ||
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
    const float input[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 2.0f};
    const float weight[12] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float bias[4] = {0.5f, -0.5f, 1.0f, 2.0f};
    std::vector<int8_t> qweight;
    std::vector<float> wscale;
    if (cuda_pack_weight_q8(weight, 4u, 3u, &qweight, &wscale, e, ec) != 0)
        return -1;
    int8_t qinput[6] = {0};
    float xscale[2] = {0.0f, 0.0f};
    for (size_t row = 0; row < 2u; ++row) {
        float amax = 0.0f;
        for (size_t col = 0; col < 3u; ++col)
            amax = std::fmax(amax, std::fabs(input[row * 3u + col]));
        xscale[row] = amax == 0.0f ? 0.0f : amax / 127.0f;
        const float inv = xscale[row] == 0.0f ? 0.0f : 127.0f / xscale[row];
        for (size_t col = 0; col < 3u; ++col)
            qinput[row * 3u + col] = (int8_t)cuda_q8_round(
                input[row * 3u + col] * inv);
    }
    float *d_input = nullptr;
    float *d_output = nullptr;
    if (ce(cudaMalloc((void **)&d_input, sizeof(input)), e, ec) != 0 ||
        ce(cudaMalloc((void **)&d_output, 8u * sizeof(float)), e, ec) != 0) {
        if (d_input != nullptr) cudaFree(d_input);
        if (d_output != nullptr) cudaFree(d_output);
        return -1;
    }
    int result = 0;
    do {
        if (ce(cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice),
               e, ec) != 0 ||
            mynah_cuda_matmul_q8_d2d(st, d_input, d_output, 2u, 3u, 4u,
                                     weight, bias, e, ec) != 0 ||
            ce(cudaStreamSynchronize(st->stream), e, ec) != 0) {
            result = -1;
            break;
        }
        float output[8] = {0.0f};
        if (ce(cudaMemcpy(output, d_output, sizeof(output),
                          cudaMemcpyDeviceToHost), e, ec) != 0) {
            result = -1;
            break;
        }
        for (size_t row = 0; row < 2u && result == 0; ++row) {
            for (size_t col = 0; col < 4u; ++col) {
                int32_t dot = 0;
                for (size_t k = 0; k < 3u; ++k)
                    dot += (int32_t)qinput[row * 3u + k] *
                           (int32_t)qweight[col * 3u + k];
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
    return cuda_q8_self_test(opaque, e, ec);
}

static void cuda_close(void *opaque) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (!st) return;
    destroy_graphs(st);
    for (auto &c : st->weights) cudaFree(c.device_pointer);
    for (auto &c : st->weights_fp16) cudaFree(c.device_ptr);
    for (auto &c : st->weights_int8) {
        cudaFree(c.device_q);
        cudaFree(c.device_scale);
    }
    if (st->dev_scratch) cudaFree(st->dev_scratch);
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
    if (st->dev_decoder_ptr0) cudaFree(st->dev_decoder_ptr0);
    if (st->dev_decoder_ptr1) cudaFree(st->dev_decoder_ptr1);
    if (st->dev_decoder_ptr2) cudaFree(st->dev_decoder_ptr2);
    if (st->dev_decoder_ptr3) cudaFree(st->dev_decoder_ptr3);
    cublasDestroy(st->cublas);
    cudaStreamDestroy(st->stream);
    delete st;
}

extern "C" int mynah_backend_cuda_open(void **state_out, mynah_backend_matmul_fn *matmul,
                                       mynah_backend_sgemm_fn *sgemm,
                                       mynah_backend_close_fn *close,
                                       mynah_backend_self_test_fn *self_test,
                                       char *e, size_t ec) {
    /* Mapped host staging is part of the FP32 bring-up path.  Device flags
     * must be set before any runtime call that can initialise the primary
     * context; otherwise cudaHostGetDevicePointer may be unavailable. */
    if (ce(cudaSetDeviceFlags(cudaDeviceMapHost), e, ec)) return -1;
    int dc = 0;
    if (ce(cudaGetDeviceCount(&dc), e, ec) || dc == 0) {
        if (dc == 0) set_error(e, ec, "no CUDA device"); return -1; }
    if (ce(cudaSetDevice(0), e, ec)) return -1;
    auto *st = new (std::nothrow) cuda_backend_state();
    if (!st) { set_error(e,ec,"oom"); return -1; }
    st->dev_scratch = nullptr; st->dev_scratch_cap = 0;
    st->dev_q8_activation = nullptr;
    st->dev_q8_activation_scale = nullptr;
    st->dev_q8_accum = nullptr;
    st->dev_q8_activation_cap = 0u;
    st->dev_q8_rows_cap = 0u;
    st->dev_q8_accum_cap = 0u;
    st->host_buf = nullptr; st->dev_buf = nullptr; st->host_buf_cap = 0;
    st->dev_argmax = nullptr;
    st->cublas_workspace = nullptr; st->cublas_workspace_cap = 0;
    st->dev_batch_k_cache = nullptr;
    st->dev_batch_v_cache = nullptr;
    st->dev_batch_positions = nullptr;
    st->dev_batch_cache_strides = nullptr;
    st->dev_decoder_ptr0 = nullptr;
    st->dev_decoder_ptr1 = nullptr;
    st->dev_decoder_ptr2 = nullptr;
    st->dev_decoder_ptr3 = nullptr;
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
    st->decoder_batch_frames.store(0ull, std::memory_order_relaxed);
    st->decoder_failures.store(0ull, std::memory_order_relaxed);
    st->resident_fallbacks.store(0ull, std::memory_order_relaxed);
    st->matmul_calls.store(0ull, std::memory_order_relaxed);
    st->matvec_calls.store(0ull, std::memory_order_relaxed);
    st->q8_matmul_calls.store(0ull, std::memory_order_relaxed);
    st->q8_rows.store(0ull, std::memory_order_relaxed);
    st->q8_weight_uploads.store(0ull, std::memory_order_relaxed);
    st->q8_weight_bytes.store(0ull, std::memory_order_relaxed);
    st->q8_activation_bytes.store(0ull, std::memory_order_relaxed);
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
    const cublasMath_t math_mode = st->fast_math
        ? CUBLAS_DEFAULT_MATH : CUBLAS_PEDANTIC_MATH;
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
    metrics->decoder_batch_frames =
        st->decoder_batch_frames.load(std::memory_order_relaxed);
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
    size_t free_bytes = 0u;
    size_t total_bytes = 0u;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        metrics->device_memory_bytes = (unsigned long long)total_bytes;
        metrics->device_memory_free_bytes = (unsigned long long)free_bytes;
    }
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

extern "C" void mynah_cuda_dev_free(void *opaque, float *dev_ptr) {
    (void)opaque;
    if (dev_ptr) cudaFree(dev_ptr);
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

/* Custom matvec kernel: out[n] = sum_k(in[k] * W[n*K + k]) + bias[n].
 * One thread per output element.  For 1×768 matvecs this beats cuBLAS
 * by eliminating launch + workspace overhead (~3μs vs ~15μs). */
__global__ static void k_matvec(const float *__restrict__ in,
                                const float *__restrict__ weight,
                                const float *__restrict__ bias,
                                float *__restrict__ out,
                                int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    const float *w = weight + (size_t)col * K;
    float sum = bias ? bias[col] : 0.0f;
    for (int k = 0; k < K; ++k) sum += in[k] * w[k];
    out[col] = sum;
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
    if (decoder->work_a != nullptr) cudaFree(decoder->work_a);
    if (decoder->work_b != nullptr) cudaFree(decoder->work_b);
    if (decoder->work_c != nullptr) cudaFree(decoder->work_c);
    if (decoder->columns != nullptr) cudaFree(decoder->columns);
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
        if (!decoder_mul(out_channels, op.max_full_len, &n) ||
            !decoder_mul(n, sizeof(float), &bytes) ||
            ce(cudaMalloc(&op.full, bytes), e, ec) ||
            ce(cudaMemset(op.full, 0, bytes), e, ec)) {
            decoder_free_op(&op);
            return false;
        }
    } else if (op.tail > 0u) {
        size_t window_len = 0u;
        if (!decoder_mul(in_channels, op.tail, &n) ||
            !decoder_mul(n, sizeof(float), &bytes) ||
            ce(cudaMalloc(&op.previous, bytes), e, ec) ||
            ce(cudaMemset(op.previous, 0, bytes), e, ec) ||
            !decoder_add(op.tail, max_in_len, &window_len) ||
            !decoder_mul(in_channels, window_len, &n) ||
            !decoder_mul(n, sizeof(float), &bytes) ||
            ce(cudaMalloc(&op.window, bytes), e, ec) ||
            ce(cudaMemset(op.window, 0, bytes), e, ec)) {
            decoder_free_op(&op);
            return false;
        }
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

static int decoder_conv1d(mynah_backend_decoder *decoder, cuda_decoder_op *op,
                          const float *input, float *output, size_t length,
                          char *e, size_t ec) {
    if (length == 0u || length > op->max_in_len || op->stride != 1 ||
        length % (size_t)op->stride != 0u)
        return -1;
    const size_t out_len = length / (size_t)op->stride;
    const size_t window_len = op->tail + length;
    const float *source = input;
    if (op->tail > 0u) {
        const size_t total = (size_t)op->in_channels * window_len;
        k_decoder_causal_window<<<((int)total + 255) / 256, 256,
                                  0, decoder->backend->stream>>>(
            op->previous, input, op->window, op->in_channels, (int)length,
            (int)op->tail);
        if (ce(cudaGetLastError(), e, ec)) return -1;
        source = op->window;
    }
    const size_t inner = (size_t)op->in_channels * (size_t)op->kernel;
    const size_t columns = inner * out_len;
    k_decoder_causal_columns<<<((int)columns + 255) / 256, 256,
                               0, decoder->backend->stream>>>(
        source, decoder->columns, op->in_channels, (int)out_len, op->kernel,
        op->dilation, op->stride, (int)op->tail);
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
    if (cbe(cublasGemmEx(decoder->backend->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
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
            op->window, op->previous, op->in_channels, (int)length,
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
    k_conv_transpose<<<((int)total + 255) / 256, 256, 0, decoder->backend->stream>>>(
        input, op->weight, op->bias, op->full, op->in_channels,
        op->out_channels, (int)length, (int)full_len, op->kernel, op->stride,
        op->groups);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (op->tail > 0u) {
        const size_t state = (size_t)op->out_channels * op->tail;
        k_decoder_convtr_fold_save<<<((int)state + 255) / 256, 256,
                                     0, decoder->backend->stream>>>(
            op->full, op->partial, op->bias, op->out_channels, (int)full_len,
            (int)op->tail);
        if (ce(cudaGetLastError(), e, ec)) return -1;
    }
    k_decoder_copy_prefix<<<((int)((size_t)op->out_channels * output_len) + 255) / 256,
                            256, 0, decoder->backend->stream>>>(
        op->full, output, op->out_channels, (int)full_len, (int)output_len);
    return ce(cudaGetLastError(), e, ec);
}

static int decoder_upload_ptrs(cuda_backend_state *backend, float **device,
                               float *const *host, size_t batch, char *e,
                               size_t ec) {
    if (backend == nullptr || device == nullptr || host == nullptr ||
        batch == 0u || batch > backend->batch_meta_cap) {
        set_error(e, ec, "invalid resident decoder pointer table");
        return -1;
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
           a->weight == b->weight && a->bias == b->bias;
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
        backend->dev_decoder_ptr0, backend->dev_decoder_ptr1, (int)batch,
        (int)elements, alpha);
    return ce(cudaGetLastError(), e, ec);
}

static int decoder_residual_batch(cuda_backend_state *backend,
                                  float *const *base, float *const *add,
                                  size_t batch, size_t elements, char *e,
                                  size_t ec) {
    size_t total = 0u;
    int blocks = 0;
    if (!decoder_mul(batch, elements, &total) ||
        !decoder_batch_launch_range(total, &blocks) ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, base, batch,
                            e, ec) != 0 ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, add, batch,
                            e, ec) != 0)
        return -1;
    k_decoder_residual_batch<<<blocks, 256, 0, backend->stream>>>(
        backend->dev_decoder_ptr0, backend->dev_decoder_ptr1, (int)batch,
        (int)elements);
    return ce(cudaGetLastError(), e, ec);
}

static int decoder_conv1d_batch(cuda_backend_state *backend,
                                cuda_decoder_op *const *ops,
                                float *const *inputs, float *const *outputs,
                                size_t batch, size_t length, char *e,
                                size_t ec) {
    if (backend == nullptr || ops == nullptr || inputs == nullptr ||
        outputs == nullptr || batch == 0u || batch > backend->batch_meta_cap ||
        ops[0] == nullptr || length == 0u ||
        length > ops[0]->max_in_len || ops[0]->stride != 1 ||
        length % (size_t)ops[0]->stride != 0u)
        return -1;
    cuda_decoder_op *op = ops[0];
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

    float *p0[CUDA_BATCH_META_CAP];
    float *p1[CUDA_BATCH_META_CAP];
    float *p2[CUDA_BATCH_META_CAP];
    float *p3[CUDA_BATCH_META_CAP];
    if (op->tail > 0u) {
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
            backend->dev_decoder_ptr0, backend->dev_decoder_ptr1,
            backend->dev_decoder_ptr2, (int)batch, op->in_channels, (int)length,
            (int)op->tail);
        if (ce(cudaGetLastError(), e, ec) != 0) return -1;
        for (size_t i = 0; i < batch; ++i) p0[i] = ops[i]->window;
    } else {
        for (size_t i = 0; i < batch; ++i) p0[i] = inputs[i];
    }
    for (size_t i = 0; i < batch; ++i) {
        if (ops[i] == nullptr || ops[i]->columns == nullptr) return -1;
        p1[i] = ops[i]->columns;
        p2[i] = outputs[i];
        p3[i] = op->weight;
    }
    int blocks = 0;
    if (!decoder_batch_launch_range(columns, &blocks) ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr0, p0, batch, e,
                            ec) != 0 ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr1, p1, batch, e,
                            ec) != 0)
        return -1;
    k_decoder_causal_columns_batch<<<blocks, 256, 0, backend->stream>>>(
        backend->dev_decoder_ptr0, backend->dev_decoder_ptr1, (int)batch,
        op->in_channels, (int)out_len, op->kernel, op->dilation, op->stride,
        (int)op->tail);
    if (ce(cudaGetLastError(), e, ec) != 0 ||
        decoder_upload_ptrs(backend, backend->dev_decoder_ptr2, p2, batch, e,
                            ec) != 0)
        return -1;
    if (!decoder_batch_launch_range(output_elements, &blocks)) return -1;
    k_decoder_bias_batch<<<blocks, 256, 0, backend->stream>>>(
        backend->dev_decoder_ptr2, op->bias, (int)batch, op->out_channels,
        (int)out_len);
    if (ce(cudaGetLastError(), e, ec) != 0) return -1;
    if (decoder_upload_ptrs(backend, backend->dev_decoder_ptr3, p3, batch, e,
                            ec) != 0 ||
        cbe(cublasSetStream(backend->cublas, backend->stream), e, ec) != 0)
        return -1;
    const float alpha = 1.0f;
    const float beta = 1.0f;
    if (cbe(cublasSgemmBatched(
                backend->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                (int)out_len, op->out_channels, (int)inner, &alpha,
                (const float *const *)backend->dev_decoder_ptr1, (int)out_len,
                (const float *const *)backend->dev_decoder_ptr3, (int)inner,
                &beta, backend->dev_decoder_ptr2, (int)out_len,
                (int)batch),
            e, ec) != 0)
        return -1;
    if (op->tail > 0u) {
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
            backend->dev_decoder_ptr0, backend->dev_decoder_ptr1, (int)batch,
            op->in_channels, (int)length, (int)op->tail);
        if (ce(cudaGetLastError(), e, ec) != 0) return -1;
    }
    return 0;
}

static int decoder_convtr_batch(cuda_backend_state *backend,
                                cuda_decoder_op *const *ops,
                                float *const *inputs, float *const *outputs,
                                size_t batch, size_t length, char *e,
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
    float *p0[CUDA_BATCH_META_CAP];
    float *p1[CUDA_BATCH_META_CAP];
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
    k_decoder_convtr_batch<<<blocks, 256, 0, backend->stream>>>(
        backend->dev_decoder_ptr0, backend->dev_decoder_ptr1, op->weight,
        op->bias, (int)batch, op->in_channels, op->out_channels, (int)length,
        (int)full_len, op->kernel, op->stride, op->groups);
    if (ce(cudaGetLastError(), e, ec) != 0) return -1;
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
            backend->dev_decoder_ptr0, backend->dev_decoder_ptr1, op->bias,
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
        backend->dev_decoder_ptr0, backend->dev_decoder_ptr1, (int)batch,
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
            return 1;
        for (size_t op = 0; op < op_count; ++op) {
            if (!decoder_ops_compatible(&first->ops[op], &decoder->ops[op]))
                return 1;
        }
    }
    if (cbe(cublasSetStream(backend->cublas, backend->stream), e, ec) != 0)
        return -1;

    float *current[CUDA_BATCH_META_CAP];
    float *other[CUDA_BATCH_META_CAP];
    float *scratch[CUDA_BATCH_META_CAP];
    float *destination[CUDA_BATCH_META_CAP];
    cuda_decoder_op *op_rows[CUDA_BATCH_META_CAP];
    for (size_t i = 0; i < batch; ++i)
        current[i] = const_cast<float *>(inputs[i]);

    size_t length = encoder_frames;
    size_t channels = first->dimension;
    for (size_t index = 0; index < op_count;) {
        cuda_decoder_op *op = &first->ops[index++];
        if (op->kind == CUDA_DECODER_RESBLOCK) {
            if (index + 1u >= op_count) {
                set_error(e, ec, "resident decoder residual topology is truncated");
                return -1;
            }
            index += 2u;
            const cuda_decoder_op *rb1 = &first->ops[index - 2u];
            for (size_t i = 0; i < batch; ++i) {
                op_rows[i] = &decoders[i]->ops[index - 2u];
                if (current[i] == decoders[i]->work_a)
                    other[i] = decoders[i]->work_b;
                else if (current[i] == decoders[i]->work_b)
                    other[i] = decoders[i]->work_a;
                else
                    other[i] = decoders[i]->work_a;
                scratch[i] = decoders[i]->work_c;
            }
            size_t elements = 0u;
            size_t hidden_elements = 0u;
            if (!decoder_mul(channels, length, &elements) ||
                !decoder_mul((size_t)rb1->out_channels, length,
                             &hidden_elements) ||
                elements > (size_t)INT_MAX ||
                hidden_elements > (size_t)INT_MAX ||
                decoder_elu_batch(backend, current, scratch, batch, elements,
                                  first->elu_alpha, e, ec) != 0 ||
                decoder_conv1d_batch(backend, op_rows, scratch, other, batch,
                                     length, e, ec) != 0)
                return -1;
            for (size_t i = 0; i < batch; ++i) {
                op_rows[i] = &decoders[i]->ops[index - 1u];
                scratch[i] = other[i];
            }
            if (decoder_elu_batch(backend, other, other, batch,
                                  hidden_elements,
                                  first->elu_alpha, e, ec) != 0 ||
                decoder_conv1d_batch(backend, op_rows, other, scratch, batch,
                                     length, e, ec) != 0 ||
                decoder_residual_batch(backend, current, scratch, batch,
                                       elements, e, ec) != 0)
                return -1;
            continue;
        }

        for (size_t i = 0; i < batch; ++i) {
            op_rows[i] = &decoders[i]->ops[index - 1u];
        }
        if (op->pre_elu) {
            size_t elements = 0u;
            if (!decoder_mul(channels, length, &elements) ||
                elements > (size_t)INT_MAX ||
                decoder_elu_batch(backend, current, current, batch, elements,
                                  first->elu_alpha, e, ec) != 0)
                return -1;
        }
        const bool last = index == op_count;
        for (size_t i = 0; i < batch; ++i) {
            if (last)
                destination[i] = outputs[i];
            else if (current[i] == decoders[i]->work_a)
                destination[i] = decoders[i]->work_b;
            else
                destination[i] = decoders[i]->work_a;
        }
        if (op->kind == CUDA_DECODER_CONV) {
            if (decoder_conv1d_batch(backend, op_rows, current, destination,
                                     batch, length, e, ec) != 0)
                return -1;
            channels = (size_t)op->out_channels;
        } else if (op->kind == CUDA_DECODER_CONVTR) {
            if (decoder_convtr_batch(backend, op_rows, current, destination,
                                     batch, length, e, ec) != 0)
                return -1;
            channels = (size_t)op->out_channels;
            if (!decoder_mul(length, (size_t)op->stride, &length))
                return -1;
        } else {
            set_error(e, ec, "resident decoder operation kind is invalid");
            return -1;
        }
        for (size_t i = 0; i < batch; ++i) current[i] = destination[i];
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
    try {
        if (decoder_build(decoder, desc, e, ec) != 0 ||
            decoder_alloc_workspace(decoder, e, ec) != 0) {
            decoder_destroy(decoder);
            return -1;
        }
    } catch (const std::bad_alloc &) {
        decoder_destroy(decoder);
        set_error(e, ec, "out of memory building resident decoder topology");
        return -1;
    }
    *out = decoder;
    return 0;
}

extern "C" void mynah_cuda_decoder_close(void *opaque,
                                           mynah_backend_decoder *decoder) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend != nullptr) (void)cudaStreamSynchronize(backend->stream);
    decoder_destroy(decoder);
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
    const int result = decoder_step_batch_impl(
        backend, decoders, dev_inputs, batch, encoder_frames, dev_outputs, e,
        ec);
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
}

extern "C" int mynah_cuda_decoder_note_step(
    void *opaque, mynah_backend_decoder *decoder) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || decoder == nullptr || decoder->backend != backend)
        return -1;
    backend->decoder_steps.fetch_add(1ull, std::memory_order_relaxed);
    return 0;
}

extern "C" int mynah_cuda_decoder_note_batch(void *opaque, size_t items,
                                               size_t frames) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || items == 0u || frames == 0u) return -1;
    if (!backend->decoder_batch_enabled) return 0;
    backend->decoder_batch_calls.fetch_add(1ull, std::memory_order_relaxed);
    backend->decoder_batch_items.fetch_add((unsigned long long)items,
                                           std::memory_order_relaxed);
    backend->decoder_batch_frames.fetch_add((unsigned long long)frames,
                                            std::memory_order_relaxed);
    return 0;
}

extern "C" int mynah_cuda_note_backbone_batch(void *opaque, size_t items) {
    auto *backend = static_cast<cuda_backend_state *>(opaque);
    if (backend == nullptr || items == 0u) return -1;
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
