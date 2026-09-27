/* Resident-weight GEMM microbenchmark for the Pocket backbone step shapes.
 *
 * Offline tooling, not part of any build target:
 *
 *     nvcc -O2 -arch=sm_89 tools/cuda_gemm_bench.cu -lcublas -o build/cuda_gemm_bench
 *     build/cuda_gemm_bench [copies]
 *
 * For each (M, N x K) it times the four weight encodings the resident path
 * can use, each as the production code issues it (see gpu/cuda/backend_cuda.cu):
 *
 *   f32   cublasGemmEx FP32 x FP32, CUBLAS_COMPUTE_32F       (SIMT, default)
 *   tf32  same FP32 weights, CUBLAS_COMPUTE_32F_FAST_TF32    (tensor cores)
 *   bf16  activation f32->bf16 kernel + GemmEx BF16 x BF16 -> FP32 (tensor cores)
 *   int8  per-row activation quantize + GemmEx INT8 x INT8 -> INT32 + f32 epilogue
 *
 * The weight is rotated over `copies` distinct device buffers (default 24,
 * one per backbone layer) so the 48 MB L2 of an L4 cannot hold it between
 * calls: the number is the weight-streaming cost a real decode step pays.
 * The per-step estimate is layers x (qkv + out_proj + ffn1 + ffn2).
 * Each encoding's error is reported against the f32 result at the same M. */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t r_ = (x); if (r_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(r_)); \
    std::exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "%s:%d cuBLAS %d\n", __FILE__, __LINE__, (int)s_); \
    std::exit(1); } } while (0)

__device__ static uint16_t bf16_from_float(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
__global__ static void k_f32_to_bf16(const float *in, uint16_t *out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = bf16_from_float(in[i]);
}
/* Mirrors k_q8_quantize_rows / k_q8_epilogue in backend_cuda.cu. */
__global__ static void k_q8_quantize_rows(const float *in, int8_t *out,
                                          float *scales, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    __shared__ float red[256];
    float amax = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        amax = fmaxf(amax, fabsf(in[(size_t)row * cols + c]));
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const float scale = red[0] > 0.0f ? red[0] / 127.0f : 1.0f;
    const float inv = 1.0f / scale;
    if (threadIdx.x == 0) scales[row] = scale;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float v = in[(size_t)row * cols + c] * inv;
        int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        q = q > 127 ? 127 : (q < -127 ? -127 : q);
        out[(size_t)row * cols + c] = (int8_t)q;
    }
}
__global__ static void k_q8_epilogue(const int32_t *acc, const float *ascale,
                                     const float *wscale, float *out, int rows,
                                     int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int r = i / cols, c = i % cols;
    out[i] = (float)acc[i] * ascale[r] * wscale[c];
}

static float lcg(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return ((float)(*s >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

struct Shape { int n, k; const char *name; };

int main(int argc, char **argv) {
    const int copies = argc > 1 ? std::atoi(argv[1]) : 24;
    const int layers = 24;
    const int ms[] = {1, 8, 16, 32, 64};
    const Shape shapes[] = {{3072, 1024, "qkv 3072x1024"},
                            {1024, 1024, "oproj 1024x1024"},
                            {4096, 1024, "ffn1 4096x1024"},
                            {1024, 4096, "ffn2 1024x4096"}};
    const int iters = 200;
    cublasHandle_t h;
    CB(cublasCreate(&h));
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    CB(cublasSetStream(h, stream));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device %s, L2 %d MB, %d weight copies, %d iters\n", prop.name,
                prop.l2CacheSize >> 20, copies, iters);
    std::printf("%-17s %3s %9s %9s %9s %9s   %9s %9s %9s\n", "shape", "M",
                "f32 ms", "tf32 ms", "bf16 ms", "int8 ms", "tf32 err",
                "bf16 err", "int8 err");
    double step[5][4] = {{0}};
    for (const Shape &sh : shapes) {
        const size_t wn = (size_t)sh.n * sh.k;
        std::vector<float> host(wn);
        std::vector<int8_t> hq(wn);
        std::vector<float> hs(sh.n);
        std::vector<float *> w32(copies);
        std::vector<uint16_t *> w16(copies);
        std::vector<int8_t *> w8(copies);
        std::vector<float *> ws(copies);
        uint32_t seed = 12345u + (uint32_t)sh.n;
        for (int c = 0; c < copies; ++c) {
            for (size_t i = 0; i < wn; ++i) host[i] = 0.05f * lcg(&seed);
            for (int r = 0; r < sh.n; ++r) {
                float amax = 0.0f;
                for (int j = 0; j < sh.k; ++j) amax = fmaxf(amax, fabsf(host[(size_t)r * sh.k + j]));
                const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
                hs[r] = s;
                for (int j = 0; j < sh.k; ++j) {
                    const float v = host[(size_t)r * sh.k + j] / s;
                    int q = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
                    hq[(size_t)r * sh.k + j] = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
                }
            }
            CK(cudaMalloc(&w32[c], wn * 4));
            CK(cudaMalloc(&w16[c], wn * 2));
            CK(cudaMalloc(&w8[c], wn));
            CK(cudaMalloc(&ws[c], sh.n * 4));
            CK(cudaMemcpy(w32[c], host.data(), wn * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(w8[c], hq.data(), wn, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(ws[c], hs.data(), sh.n * 4, cudaMemcpyHostToDevice));
            k_f32_to_bf16<<<(int)((wn + 255) / 256), 256>>>(w32[c], w16[c], (int)wn);
        }
        CK(cudaDeviceSynchronize());
        const int mmax = 64;
        float *x, *y, *yref, *ascale;
        uint16_t *x16;
        int8_t *x8;
        int32_t *acc;
        CK(cudaMalloc(&x, (size_t)mmax * sh.k * 4));
        CK(cudaMalloc(&x16, (size_t)mmax * sh.k * 2));
        CK(cudaMalloc(&x8, (size_t)mmax * sh.k));
        CK(cudaMalloc(&ascale, mmax * 4));
        CK(cudaMalloc(&acc, (size_t)mmax * sh.n * 4));
        CK(cudaMalloc(&y, (size_t)mmax * sh.n * 4));
        CK(cudaMalloc(&yref, (size_t)mmax * sh.n * 4));
        {
            std::vector<float> hx((size_t)mmax * sh.k);
            for (auto &v : hx) v = lcg(&seed);
            CK(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
        }
        for (int mi = 0; mi < 5; ++mi) {
            const int M = ms[mi];
            const float one = 1.0f, zero = 0.0f;
            const int32_t ione = 1, izero = 0;
            auto run = [&](int mode, int c, float *out) {
                if (mode == 0 || mode == 1) {
                    CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, sh.n, M, sh.k, &one,
                                    w32[c], CUDA_R_32F, sh.k, x, CUDA_R_32F, sh.k,
                                    &zero, out, CUDA_R_32F, sh.n,
                                    mode == 0 ? CUBLAS_COMPUTE_32F
                                              : CUBLAS_COMPUTE_32F_FAST_TF32,
                                    CUBLAS_GEMM_DEFAULT));
                } else if (mode == 2) {
                    const int n = M * sh.k;
                    k_f32_to_bf16<<<(n + 255) / 256, 256, 0, stream>>>(x, x16, n);
                    CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, sh.n, M, sh.k, &one,
                                    w16[c], CUDA_R_16BF, sh.k, x16, CUDA_R_16BF, sh.k,
                                    &zero, out, CUDA_R_32F, sh.n, CUBLAS_COMPUTE_32F,
                                    CUBLAS_GEMM_DEFAULT));
                } else {
                    k_q8_quantize_rows<<<M, 256, 0, stream>>>(x, x8, ascale, M, sh.k);
                    CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, sh.n, M, sh.k, &ione,
                                    w8[c], CUDA_R_8I, sh.k, x8, CUDA_R_8I, sh.k,
                                    &izero, acc, CUDA_R_32I, sh.n, CUBLAS_COMPUTE_32I,
                                    CUBLAS_GEMM_DEFAULT));
                    const int n = M * sh.n;
                    k_q8_epilogue<<<(n + 255) / 256, 256, 0, stream>>>(acc, ascale, ws[c],
                                                                       out, M, sh.n);
                }
            };
            double ms_mode[4];
            double err[4] = {0, 0, 0, 0};
            run(0, 0, yref);
            CK(cudaStreamSynchronize(stream));
            std::vector<float> ref((size_t)M * sh.n), got((size_t)M * sh.n);
            CK(cudaMemcpy(ref.data(), yref, ref.size() * 4, cudaMemcpyDeviceToHost));
            for (int mode = 0; mode < 4; ++mode) {
                for (int i = 0; i < 20; ++i) run(mode, i % copies, y);
                CK(cudaEventRecord(e0, stream));
                for (int i = 0; i < iters; ++i) run(mode, i % copies, y);
                CK(cudaEventRecord(e1, stream));
                CK(cudaEventSynchronize(e1));
                float t = 0.0f;
                CK(cudaEventElapsedTime(&t, e0, e1));
                ms_mode[mode] = t / iters;
                run(mode, 0, y);
                CK(cudaStreamSynchronize(stream));
                CK(cudaMemcpy(got.data(), y, got.size() * 4, cudaMemcpyDeviceToHost));
                double num = 0, den = 0;
                for (size_t i = 0; i < got.size(); ++i) {
                    num += (got[i] - ref[i]) * (double)(got[i] - ref[i]);
                    den += (double)ref[i] * ref[i];
                }
                err[mode] = std::sqrt(num / (den > 0 ? den : 1));
                step[mi][mode] += ms_mode[mode] * layers;
            }
            std::printf("%-17s %3d %9.4f %9.4f %9.4f %9.4f   %9.2e %9.2e %9.2e\n",
                        sh.name, M, ms_mode[0], ms_mode[1], ms_mode[2], ms_mode[3],
                        err[1], err[2], err[3]);
        }
        for (int c = 0; c < copies; ++c) {
            cudaFree(w32[c]); cudaFree(w16[c]); cudaFree(w8[c]); cudaFree(ws[c]);
        }
        cudaFree(x); cudaFree(x16); cudaFree(x8); cudaFree(ascale);
        cudaFree(acc); cudaFree(y); cudaFree(yref);
    }
    std::printf("\nbackbone step GEMMs only (%d layers x 4 projections), ms:\n", layers);
    std::printf("%3s %9s %9s %9s %9s\n", "M", "f32", "tf32", "bf16", "int8");
    for (int mi = 0; mi < 5; ++mi)
        std::printf("%3d %9.3f %9.3f %9.3f %9.3f\n", ms[mi], step[mi][0], step[mi][1],
                    step[mi][2], step[mi][3]);
    cublasDestroy(h);
    return 0;
}
