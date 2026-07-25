#include "backend.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

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
                                    int width, float eps) {
    int row = blockIdx.x;
    const float *x = in + (size_t)row * width;
    float *y = out + (size_t)row * width;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) sum += x[i];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    __shared__ float s_mean, s_inv;
    if (threadIdx.x == 0) s_mean = sum / (float)width;
    __syncthreads();
    float mean = s_mean;
    float var = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float d = x[i] - mean;
        var += d * d;
    }
    for (int off = 16; off > 0; off >>= 1) var += __shfl_down_sync(0xffffffff, var, off);
    if (threadIdx.x == 0) s_inv = rsqrtf(var / (float)width + eps);
    __syncthreads();
    float inv = s_inv;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        y[i] = (x[i] - mean) * inv * gain[i];
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

struct cuda_backend_state {
    cublasHandle_t cublas;
    cudaStream_t stream;
    std::vector<cuda_cached_buffer> weights;
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

/* ------------------------------------------------------------------ */
/*  matmul / sgemm (existing interface, with sync)                     */
/* ------------------------------------------------------------------ */

static int cuda_matmul(void *opaque, const float *input, float *output, size_t rows,
                       size_t iw, size_t ow, const float *weight, const float *bias,
                       char *e, size_t ec) {
    auto *st = static_cast<cuda_backend_state *>(opaque);
    const size_t ib = rows * iw * sizeof(float);
    const size_t ob = rows * ow * sizeof(float);
    float *dw = nullptr;
    if (cached_weight(st, weight, iw * ow * sizeof(float), &dw, e, ec)) return -1;
    float *db = nullptr;
    if (bias && cached_weight(st, bias, ow * sizeof(float), &db, e, ec)) return -1;
    if (ensure_host(st, ib + ob, e, ec)) return -1;
    float *di = st->dev_buf;
    float *dout = st->dev_buf + rows * iw;
    std::memcpy(st->host_buf, input, ib);
    cublasSetStream(st->cublas, st->stream);
    const float a1 = 1.0f, b0 = 0.0f;
    if (cbe(cublasSgemm(st->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                        (int)ow, (int)rows, (int)iw, &a1,
                        dw, (int)iw, di, (int)iw, &b0, dout, (int)ow), e, ec)) return -1;
    if (db) {
        const float one = 1.0f;
        /* bias via Sger */
        static float *d_ones = nullptr; static size_t oc = 0;
        if (oc < rows) { if (d_ones) cudaFree(d_ones); size_t c2 = (rows+255)&~255;
            ce(cudaMalloc(&d_ones, c2*4), e, ec); std::vector<float> h(c2,1.0f);
            cudaMemcpy(d_ones,h.data(),c2*4,cudaMemcpyHostToDevice); oc=c2; }
        cublasSger(st->cublas,(int)ow,(int)rows,&one,db,1,d_ones,1,dout,(int)ow);
    }
    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    std::memcpy(output, st->host_buf + rows * iw, ob);
    return 0;
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
    for (size_t r=0;r<ar;r++) std::memcpy(ha+r*ac, a+r*lda, ac*4);
    for (size_t r=0;r<br;r++) std::memcpy(hb+r*bc, b+r*ldb, bc*4);
    if (beta!=0.0f) for (size_t r=0;r<m;r++) std::memcpy(hc+r*n, c+r*ldc, n*4);
    cublasSetStream(st->cublas, st->stream);
    cublasOperation_t oa = tb?CUBLAS_OP_T:CUBLAS_OP_N;
    cublasOperation_t ob = ta?CUBLAS_OP_T:CUBLAS_OP_N;
    if (cbe(cublasSgemm(st->cublas,oa,ob,(int)n,(int)m,(int)k,&alpha,
                        db2,(int)bc,da,(int)ac,&beta,dc,(int)n),e,ec)) return -1;
    if (ce(cudaStreamSynchronize(st->stream), e, ec)) return -1;
    for (size_t r=0;r<m;r++) std::memcpy(c+r*ldc, hc+r*n, n*4);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Device-side ops (CUDA implementations)                             */
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
    int threads = width <= 256 ? (int)width : 256;
    threads = ((threads + 31) / 32) * 32;
    if (threads > 1024) threads = 1024;
    k_layer_norm<<<(int)rows, threads, 0, st->stream>>>(
        d_out, d_in, d_gain, (int)width, 1e-5f);
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
