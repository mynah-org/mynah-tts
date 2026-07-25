#include "backend.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

__global__ static void k_layer_norm(float *out, const float *in,
                                    const float *gain,
                                    int width, float eps, int nrows) {
    /* One thread per row: sequential accumulation matches CPU bit-order. */
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    const float *x = in + (size_t)row * width;
    float *y = out + (size_t)row * width;
    float mean = 0.0f;
    for (int d = 0; d < width; ++d) mean += x[d];
    mean /= (float)width;
    float var = 0.0f;
    for (int d = 0; d < width; ++d) { float dd = x[d] - mean; var += dd * dd; }
    float inv = 1.0f / sqrtf(var / (float)width + eps);
    for (int d = 0; d < width; ++d)
        y[d] = (x[d] - mean) * inv * gain[d];
}

__global__ static void k_softmax_causal(float *data, int cols, int valid) {
    int row = blockIdx.x;
    float *r = data + (size_t)row * cols;
    int v = valid < cols ? valid : cols;
    float mx = -1e30f;
    for (int i = threadIdx.x; i < v; i += blockDim.x) mx = fmaxf(mx, r[i]);
    for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, off));
    __shared__ float s_mx, s_sum;
    if (threadIdx.x == 0) s_mx = mx;
    __syncthreads();
    mx = s_mx;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < v; i += blockDim.x) { r[i] = expf(r[i] - mx); sum += r[i]; }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (threadIdx.x == 0) s_sum = sum;
    __syncthreads();
    float inv = 1.0f / s_sum;
    for (int i = threadIdx.x; i < v; i += blockDim.x) r[i] *= inv;
    for (int i = threadIdx.x + v; i < cols; i += blockDim.x) r[i] = 0.0f;
}

__global__ static void k_gelu(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = data[i];
        float c = 0.7978845608f * (x + 0.044715f * x * x * x);
        data[i] = 0.5f * x * (1.0f + tanhf(c));
    }
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

__global__ static void k_copy_strided(float *dst, const float *src,
                                      int dst_stride, int src_stride,
                                      int width, int rows) {
    int r = blockIdx.x;
    if (r >= rows) return;
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        dst[(size_t)r * dst_stride + c] = src[(size_t)r * src_stride + c];
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

struct cuda_backend_state {
    cublasHandle_t cublas;
    cudaStream_t stream;
    std::vector<cuda_cached_buffer> weights;
    std::vector<cuda_cached_fp16> weights_fp16;
    /* Device scratch for activations (grows on demand). */
    float *dev_scratch;
    size_t dev_scratch_cap;   /* bytes */
    /* Pinned host buffer for H2D/D2H. */
    float *host_buf;
    float *dev_buf;           /* mapped device pointer for host_buf */
    size_t host_buf_cap;
};

static void set_error(char *e, size_t c, const char *m) {
    if (e && c > 0) std::snprintf(e, c, "%s", m);
}
static int ce(cudaError_t r, char *e, size_t c) {
    if (r == cudaSuccess) return 0;
    std::snprintf(e, c, "CUDA: %s", cudaGetErrorString(r)); return -1;
}
static int cbe(cublasStatus_t s, char *e, size_t c) {
    if (s == CUBLAS_STATUS_SUCCESS) return 0;
    std::snprintf(e, c, "cuBLAS: %d", (int)s); return -1;
}

static int ensure_scratch(cuda_backend_state *st, size_t bytes, char *e, size_t ec) {
    if (st->dev_scratch_cap >= bytes) return 0;
    if (st->dev_scratch) cudaFree(st->dev_scratch);
    size_t cap = (bytes + (32u<<20) - 1u) & ~((32u<<20) - 1u);
    if (ce(cudaMalloc(&st->dev_scratch, cap), e, ec)) { st->dev_scratch = nullptr; st->dev_scratch_cap = 0; return -1; }
    st->dev_scratch_cap = cap;
    return 0;
}

static int ensure_host(cuda_backend_state *st, size_t bytes, char *e, size_t ec) {
    if (st->host_buf_cap >= bytes) return 0;
    if (st->host_buf) cudaFreeHost(st->host_buf);
    st->host_buf = nullptr; st->dev_buf = nullptr; st->host_buf_cap = 0;
    size_t cap = (bytes + (16u<<20) - 1u) & ~((16u<<20) - 1u);
    if (ce(cudaHostAlloc(&st->host_buf, cap, cudaHostAllocMapped), e, ec)) return -1;
    if (ce(cudaHostGetDevicePointer(&st->dev_buf, st->host_buf, 0), e, ec)) {
        cudaFreeHost(st->host_buf); st->host_buf = nullptr; return -1;
    }
    st->host_buf_cap = cap;
    return 0;
}

static int cached_weight(cuda_backend_state *st, const float *hp, size_t bytes,
                         float **dp, char *e, size_t ec) {
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
    for (auto &c : st->weights_fp16)
        if (c.host_pointer == hp && c.n == n) { *dp = c.device_ptr; return 0; }
    float *tmp = nullptr;
    half *d16 = nullptr;
    if (ce(cudaMalloc(&tmp, n * sizeof(float)), e, ec)) return -1;
    if (ce(cudaMalloc(&d16, n * sizeof(half)), e, ec)) { cudaFree(tmp); return -1; }
    cudaMemcpy(tmp, hp, n * sizeof(float), cudaMemcpyHostToDevice);
    k_f32_to_f16<<<((int)n+255)/256, 256>>>(tmp, d16, (int)n);
    cudaFree(tmp);
    st->weights_fp16.push_back({hp, n, d16});
    *dp = d16;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  matmul / sgemm (existing interface, with sync)                     */
/* ------------------------------------------------------------------ */

static int cuda_matmul(void *opaque, const float *input, float *output, size_t rows,
                       size_t iw, size_t ow, const float *weight, const float *bias,
                       char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    const size_t in_n = rows * iw;
    const size_t out_n = rows * ow;
    const size_t w_n = iw * ow;
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
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP), e, ec)) return -1;
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
    const size_t in_n = rows * iw;
    half *dw16 = nullptr;
    if (cached_weight_fp16(st, weight, iw * ow, &dw16, e, ec)) return -1;
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
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP), e, ec)) return -1;
    if (db) {
        const float one = 1.0f;
        static float *d_ones2 = nullptr; static size_t oc2 = 0;
        if (oc2 < rows) { if (d_ones2) cudaFree(d_ones2); size_t c2 = (rows+255)&~255;
            ce(cudaMalloc(&d_ones2, c2*4), e, ec); std::vector<float> h(c2,1.0f);
            cudaMemcpy(d_ones2,h.data(),c2*4,cudaMemcpyHostToDevice); oc2=c2; }
        cublasSger(st->cublas,(int)ow,(int)rows,&one,db,1,d_ones2,1,d_out,(int)ow);
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
                        CUBLAS_COMPUTE_32F_FAST_16F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP),e,ec)) return -1;
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
    float *dw = nullptr;
    if (cached_weight(st, weight, iw * ow * sizeof(float), &dw, e, ec)) return -1;
    float *db = nullptr;
    if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b0 = 0.0f;
    if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         (int)ow, (int)rows, (int)iw,
                         &a1, dw, CUDA_R_32F, (int)iw,
                         d_in, CUDA_R_32F, (int)iw,
                         &b0, d_out, CUDA_R_32F, (int)ow,
                         CUBLAS_COMPUTE_32F_FAST_16F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP), e, ec)) return -1;
    if (db) {
        const float one = 1.0f;
        static float *d_ones = nullptr; static size_t oc = 0;
        if (oc < rows) { if (d_ones) cudaFree(d_ones); size_t c2 = (rows+255)&~255;
            ce(cudaMalloc(&d_ones, c2*4), e, ec); std::vector<float> h(c2,1.0f);
            cudaMemcpy(d_ones,h.data(),c2*4,cudaMemcpyHostToDevice); oc=c2; }
        cublasSger(st->cublas,(int)ow,(int)rows,&one,db,1,d_ones,1,d_out,(int)ow);
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
    cublasSetStream(st->cublas, st->stream);
    cublasOperation_t oa = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t ob = ta ? CUBLAS_OP_T : CUBLAS_OP_N;
    /* Packed leading dimensions: caller provides actual data cols. */
    size_t ac = ta ? m : k;
    size_t bc = tb ? k : n;
    return cbe(cublasGemmEx(st->cublas, oa, ob,
                           (int)n, (int)m, (int)k, &alpha,
                           d_b, CUDA_R_32F, (int)bc,
                           d_a, CUDA_R_32F, (int)ac,
                           &beta, d_c, CUDA_R_32F, (int)n,
                           CUBLAS_COMPUTE_32F_FAST_16F,
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP), e, ec) ? -1 : 0;
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
    return ce(cudaMemcpyAsync(*dev_ptr, host, bytes, cudaMemcpyHostToDevice, st->stream), e, ec);
}

extern "C" int mynah_cuda_download(void *opaque, const float *dev_ptr, float *host,
                         size_t n, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    return ce(cudaMemcpyAsync(host, dev_ptr, n * sizeof(float),
                              cudaMemcpyDeviceToHost, st->stream), e, ec);
}

extern "C" int mynah_cuda_sync(void *opaque, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    return ce(cudaStreamSynchronize(st->stream), e, ec);
}

extern "C" int mynah_cuda_snake_dev(void *opaque, float *dev_data, const float *alpha,
                          size_t channels, size_t length, size_t snake_ch,
                          char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    /* Upload alpha to device (small, use host_buf). */
    if (ensure_host(st, channels * sizeof(float), e, ec)) return -1;
    std::memcpy(st->host_buf, alpha, channels * sizeof(float));
    int threads = 256;
    k_snake<<<(int)channels, threads, 0, st->stream>>>(
        dev_data, st->dev_buf, (int)channels, (int)length, (int)snake_ch);
    return ce(cudaGetLastError(), e, ec);
}

extern "C" int mynah_cuda_gelu_dev(void *opaque, float *data, size_t n,
                         char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t bytes = n * sizeof(float);
    if (ensure_scratch(st, bytes, e, ec)) return -1;
    float *d = st->dev_scratch;
    if (ce(cudaMemcpyAsync(d, data, bytes, cudaMemcpyHostToDevice, st->stream), e, ec)) return -1;
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    k_gelu<<<blocks, threads, 0, st->stream>>>(d, (int)n);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (ce(cudaMemcpyAsync(data, d, bytes, cudaMemcpyDeviceToHost, st->stream), e, ec)) return -1;
    return ce(cudaStreamSynchronize(st->stream), e, ec);
}

extern "C" int mynah_cuda_layer_norm_dev(void *opaque, const float *in, float *out,
                               const float *gain, const float *bias,
                               size_t rows, size_t width,
                               char *e, size_t ec) {
    (void)bias; /* Magpie layer_norm has no bias */
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t data_bytes = rows * width * sizeof(float);
    size_t gb = width * sizeof(float);
    if (ensure_scratch(st, data_bytes * 2, e, ec)) return -1;
    if (ensure_host(st, gb, e, ec)) return -1;
    float *d_in = st->dev_scratch;
    float *d_out = st->dev_scratch + rows * width;
    std::memcpy(st->host_buf, gain, gb);
    float *d_gain = st->dev_buf;
    if (ce(cudaMemcpyAsync(d_in, in, data_bytes, cudaMemcpyHostToDevice, st->stream), e, ec)) return -1;
    /* 1 thread per row, sequential accumulation for CPU bit-parity. */
    int threads = rows <= 256 ? (int)rows : 256;
    int blocks = ((int)rows + threads - 1) / threads;
    k_layer_norm<<<blocks, threads, 0, st->stream>>>(
        d_out, d_in, d_gain, (int)width, 1e-5f, (int)rows);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (ce(cudaMemcpyAsync(out, d_out, data_bytes, cudaMemcpyDeviceToHost, st->stream), e, ec)) return -1;
    return ce(cudaStreamSynchronize(st->stream), e, ec);
}

extern "C" int mynah_cuda_residual_add_dev(void *opaque, float *out, const float *in,
                                 size_t n, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    size_t bytes = n * sizeof(float);
    if (ensure_scratch(st, bytes * 2, e, ec)) return -1;
    float *d_out = st->dev_scratch;
    float *d_in = st->dev_scratch + n;
    if (ce(cudaMemcpyAsync(d_out, out, bytes, cudaMemcpyHostToDevice, st->stream), e, ec)) return -1;
    if (ce(cudaMemcpyAsync(d_in, in, bytes, cudaMemcpyHostToDevice, st->stream), e, ec)) return -1;
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    k_residual_add<<<blocks, threads, 0, st->stream>>>(d_out, d_in, (int)n);
    if (ce(cudaGetLastError(), e, ec)) return -1;
    if (ce(cudaMemcpyAsync(out, d_out, bytes, cudaMemcpyDeviceToHost, st->stream), e, ec)) return -1;
    return ce(cudaStreamSynchronize(st->stream), e, ec);
}

/* ------------------------------------------------------------------ */
/*  Self-test                                                          */
/* ------------------------------------------------------------------ */

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
    return 0;
}

static void cuda_close(void *opaque) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    if (!st) return;
    for (auto &c : st->weights) cudaFree(c.device_pointer);
    for (auto &c : st->weights_fp16) cudaFree(c.device_ptr);
    if (st->dev_scratch) cudaFree(st->dev_scratch);
    if (st->host_buf) cudaFreeHost(st->host_buf);
    cublasDestroy(st->cublas);
    cudaStreamDestroy(st->stream);
    delete st;
}

extern "C" int mynah_backend_cuda_open(void **state_out, mynah_backend_matmul_fn *matmul,
                                       mynah_backend_sgemm_fn *sgemm,
                                       mynah_backend_close_fn *close,
                                       mynah_backend_self_test_fn *self_test,
                                       char *e, size_t ec) {
    int dc = 0;
    if (ce(cudaGetDeviceCount(&dc), e, ec) || dc == 0) {
        if (dc == 0) set_error(e, ec, "no CUDA device"); return -1; }
    ce(cudaSetDevice(0), e, ec);
    cudaSetDeviceFlags(cudaDeviceMapHost);
    auto *st = new (std::nothrow) cuda_backend_state();
    if (!st) { set_error(e,ec,"oom"); return -1; }
    st->dev_scratch = nullptr; st->dev_scratch_cap = 0;
    st->host_buf = nullptr; st->dev_buf = nullptr; st->host_buf_cap = 0;
    if (ce(cudaStreamCreate(&st->stream), e, ec)) { delete st; return -1; }
    if (cublasCreate(&st->cublas) != CUBLAS_STATUS_SUCCESS) {
        set_error(e,ec,"cuBLAS init"); cudaStreamDestroy(st->stream); delete st; return -1; }
    cublasSetMathMode(st->cublas, CUBLAS_DEFAULT_MATH);
    *state_out = st;
    *matmul = cuda_matmul;
    *sgemm = cuda_sgemm;
    *close = cuda_close;
    *self_test = cuda_self_test;
    if (e && ec > 0) e[0] = '\0';
    return 0;
}

extern "C" int mynah_cuda_dev_alloc(void *opaque, size_t n, float **dev_ptr,
                          char *e, size_t ec) {
    (void)opaque;
    return ce(cudaMalloc(dev_ptr, n * sizeof(float)), e, ec);
}

extern "C" void mynah_cuda_dev_free(void *opaque, float *dev_ptr) {
    (void)opaque;
    if (dev_ptr) cudaFree(dev_ptr);
}

extern "C" int mynah_cuda_h2d(void *opaque, const float *host, float *dev_ptr,
                    size_t n, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    return ce(cudaMemcpyAsync(dev_ptr, host, n*sizeof(float),
                              cudaMemcpyHostToDevice, st->stream), e, ec);
}

extern "C" int mynah_cuda_d2h(void *opaque, const float *dev_ptr, float *host,
                    size_t n, char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
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
    half *dw16 = nullptr;
    if (cached_weight_fp16(st, weight, K * N, &dw16, e, ec)) return -1;
    float *db = nullptr;
    if (bias && cached_weight(st, bias, N * sizeof(float), &db, e, ec)) return -1;
    /* Convert FP32 input to FP16 in scratch. */
    if (ensure_scratch(st, K * sizeof(half), e, ec)) return -1;
    half *di16 = (half *)st->dev_scratch;
    k_f32_to_f16<<<((int)K+255)/256, 256, 0, st->stream>>>(d_in, di16, (int)K);
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b0 = 0.0f;
    if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         (int)N, 1, (int)K,
                         &a1, dw16, CUDA_R_16F, (int)K,
                         di16, CUDA_R_16F, (int)K,
                         &b0, d_out, CUDA_R_32F, (int)N,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP), e, ec)) return -1;
    if (db) {
        const float one = 1.0f;
        static float *d_one = nullptr;
        if (!d_one) { ce(cudaMalloc(&d_one, 4), e, ec); cudaMemcpy(d_one, &one, 4, cudaMemcpyHostToDevice); }
        cublasSaxpy(st->cublas, (int)N, &one, db, 1, d_out, 1);
    }
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
    /* Cache gain on device (same pointer = same layer, reused across steps). */
    float *d_gain = nullptr;
    if (cached_weight(st, gain, width * sizeof(float), &d_gain, e, ec)) return -1;
    k_layer_norm<<<(int)rows, 1, 0, st->stream>>>(
        dev_out, dev_in, d_gain, (int)width, 1e-5f, (int)rows);
    return ce(cudaGetLastError(), e, ec);
}

/* matmul device-to-device: input already on GPU, no host round-trip.
 * Converts FP32 device input → FP16, runs cuBLAS, output stays on device.
 * Does NOT sync. Caller syncs when needed. */
extern "C" int mynah_cuda_matmul_d2d(void *opaque, const float *d_in, float *d_out,
                           size_t rows, size_t iw, size_t ow,
                           const float *weight, const float *bias,
                           char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    const size_t in_n = rows * iw;
    half *dw16 = nullptr;
    if (cached_weight_fp16(st, weight, iw * ow, &dw16, e, ec)) return -1;
    float *db = nullptr;
    if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
    if (ensure_scratch(st, in_n * sizeof(half), e, ec)) return -1;
    half *di16 = (half *)st->dev_scratch;
    k_f32_to_f16<<<((int)in_n+255)/256, 256, 0, st->stream>>>(d_in, di16, (int)in_n);
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b0 = 0.0f;
    if (cbe(cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         (int)ow, (int)rows, (int)iw,
                         &a1, dw16, CUDA_R_16F, (int)iw,
                         di16, CUDA_R_16F, (int)iw,
                         &b0, d_out, CUDA_R_32F, (int)ow,
                         CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP), e, ec)) return -1;
    if (db) {
        const float one = 1.0f;
        static float *d_ones3 = nullptr; static size_t oc3 = 0;
        if (oc3 < rows) { if (d_ones3) cudaFree(d_ones3); size_t c2 = (rows+255)&~255;
            ce(cudaMalloc(&d_ones3, c2*4), e, ec); std::vector<float> h(c2,1.0f);
            cudaMemcpy(d_ones3,h.data(),c2*4,cudaMemcpyHostToDevice); oc3=c2; }
        cublasSger(st->cublas,(int)ow,(int)rows,&one,db,1,d_ones3,1,d_out,(int)ow);
    }
    return 0; /* no sync */
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
                         CUBLAS_COMPUTE_32F_FAST_16F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP), e, ec)) return -1;

    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;

    /* Download output via mapped buffer. */
    size_t out_mapped_need = out_count * sizeof(float);
    if (ensure_host(st, out_mapped_need, e, ec)) return -1;
    cudaMemcpy(st->host_buf, d_out, out_count * sizeof(float), cudaMemcpyDeviceToHost);
    std::memcpy(output, st->host_buf, out_count * sizeof(float));
    return 0;
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
struct cuda_graph_entry {
    size_t rows, iw, ow;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    bool valid;
};

#define GRAPH_CACHE_SIZE 16
static cuda_graph_entry g_graph_cache[GRAPH_CACHE_SIZE];
static int g_graph_count = 0;

static cuda_graph_entry *find_graph(size_t rows, size_t iw, size_t ow) {
    for (int i = 0; i < g_graph_count; ++i)
        if (g_graph_cache[i].rows == rows && g_graph_cache[i].iw == iw &&
            g_graph_cache[i].ow == ow)
            return &g_graph_cache[i];
    return nullptr;
}

/* Matmul with CUDA Graph: capture on first call, replay on subsequent.
 * Falls back to regular matmul if capture fails. */
extern "C" int mynah_cuda_matmul_graph(void *opaque, const float *input, float *output,
                             size_t rows, size_t iw, size_t ow,
                             const float *weight, const float *bias,
                             char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);

    /* Try to find a cached graph. */
    cuda_graph_entry *entry = find_graph(rows, iw, ow);
    if (entry && entry->valid) {
        /* Update input in mapped buffer, replay graph. */
        const size_t in_n = rows * iw;
        const size_t out_n = rows * ow;
        size_t mapped_need = (in_n + out_n) * sizeof(float);
        if (ensure_host(st, mapped_need, e, ec)) return -1;
        std::memcpy(st->host_buf, input, in_n * sizeof(float));
        cudaGraphLaunch(entry->exec, st->stream);
        if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
        std::memcpy(output, st->host_buf + in_n, out_n * sizeof(float));
        return 0;
    }

    /* First call: capture the graph. */
    const size_t in_n = rows * iw;
    const size_t out_n = rows * ow;
    const size_t w_n = iw * ow;
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

    /* Capture. Alpha/beta must be static (not stack) for graph capture. */
    static const float g_alpha = 1.0f, g_beta = 0.0f;
    cublasSetStream(st->cublas, st->stream);
    cudaStreamBeginCapture(st->stream, cudaStreamCaptureModeGlobal);
    k_f32_to_f16<<<((int)in_n+255)/256, 256, 0, st->stream>>>(st->dev_buf, di16, (int)in_n);
    cublasGemmEx(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                 (int)ow, (int)rows, (int)iw,
                 &g_alpha, dw16, CUDA_R_16F, (int)iw,
                 di16, CUDA_R_16F, (int)iw,
                 &g_beta, d_out_mapped, CUDA_R_32F, (int)ow,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (db) {
        k_bias_add<<<((int)(rows*ow)+255)/256, 256, 0, st->stream>>>(
            d_out_mapped, db, (int)rows, (int)ow);
    }
    cudaGraph_t graph;
    cudaError_t cap_err = cudaStreamEndCapture(st->stream, &graph);
    if (cap_err != cudaSuccess || graph == nullptr) {
        /* Capture failed — fall back to regular matmul. */
        return cuda_matmul(opaque, input, output, rows, iw, ow, weight, bias, e, ec);
    }

    cudaGraphExec_t exec;
    if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) {
        cudaGraphDestroy(graph);
        return cuda_matmul(opaque, input, output, rows, iw, ow, weight, bias, e, ec);
    }

    /* Cache the graph. */
    if (g_graph_count < GRAPH_CACHE_SIZE) {
        g_graph_cache[g_graph_count] = {rows, iw, ow, graph, exec, true};
        g_graph_count++;
    }

    /* Replay for this call. */
    cudaGraphLaunch(exec, st->stream);
    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    std::memcpy(output, st->host_buf + in_n, out_n * sizeof(float));
    return 0;
}
