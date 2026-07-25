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
 * CUDA backend for the GB10 Grace Blackwell (unified memory) and discrete
 * NVIDIA GPUs.  Uses cuBLAS for GEMM and caches weight matrices on the
 * device so the autoregressive loop never re-uploads them.
 *
 * On unified-memory platforms (GB10) the host pointers from mmap'd
 * safetensors are accessible from the GPU after a one-time migration;
 * on discrete GPUs the weights are explicitly copied once.
 */

struct cuda_cached_buffer {
    const void *host_pointer;
    size_t bytes;
    float *device_pointer;
};

struct cuda_backend_state {
    cublasHandle_t cublas;
    cudaStream_t stream;
    std::vector<cuda_cached_buffer> weights;
    float *io_buffer;          /* reusable device IO buffer */
    size_t io_capacity;        /* bytes */
    int unified_memory;        /* 1 if platform has unified memory */
};

static void set_error(char *error, size_t capacity, const char *message) {
    if (error != nullptr && capacity > 0) std::snprintf(error, capacity, "%s", message);
}

static int cuda_error(cudaError_t result, char *error, size_t error_capacity) {
    if (result == cudaSuccess) return 0;
    std::snprintf(error, error_capacity, "CUDA error: %s", cudaGetErrorString(result));
    return -1;
}

static int cublas_error(cublasStatus_t status, char *error, size_t error_capacity) {
    if (status == CUBLAS_STATUS_SUCCESS) return 0;
    std::snprintf(error, error_capacity, "cuBLAS error: %d", (int)status);
    return -1;
}

/* Ensure a device buffer of at least `bytes` is available, reusing the
 * pooled IO buffer when possible. */
static int ensure_io(cuda_backend_state *state, size_t bytes, char *error, size_t ec) {
    if (state->io_capacity >= bytes) return 0;
    if (state->io_buffer != nullptr) cudaFree(state->io_buffer);
    /* Round up to 16 MB granularity to avoid frequent reallocs. */
    size_t cap = (bytes + (16u << 20) - 1u) & ~((16u << 20) - 1u);
    if (cuda_error(cudaMalloc(&state->io_buffer, cap), error, ec) != 0) {
        state->io_buffer = nullptr;
        state->io_capacity = 0;
        return -1;
    }
    state->io_capacity = cap;
    return 0;
}

/* Look up or upload a weight/bias buffer.  On unified-memory platforms
 * we still copy once so the GPU page tables are populated; subsequent
 * calls hit the cache in O(n) linear scan (n is small: ~50 tensors). */
static int cached_weight(cuda_backend_state *state, const float *host_pointer,
                         size_t bytes, float **device_pointer,
                         char *error, size_t error_capacity) {
    for (const cuda_cached_buffer &cached : state->weights) {
        if (cached.host_pointer == host_pointer && cached.bytes == bytes) {
            *device_pointer = cached.device_pointer;
            return 0;
        }
    }
    float *device = nullptr;
    if (cuda_error(cudaMalloc(&device, bytes), error, error_capacity) != 0) return -1;
    if (cuda_error(cudaMemcpyAsync(device, host_pointer, bytes,
                                   cudaMemcpyHostToDevice, state->stream),
                   error, error_capacity) != 0) {
        cudaFree(device);
        return -1;
    }
    state->weights.push_back({host_pointer, bytes, device});
    *device_pointer = device;
    return 0;
}

/*
 * cuBLAS sgemm wrapper.
 *
 * We need:  output[rows, output_width] = input[rows, input_width] @ weight[output_width, input_width]^T + bias
 *
 * cuBLAS is column-major, so we compute the transpose:
 *   output^T = weight @ input^T
 * i.e. C(output_width, rows) = A(output_width, input_width) * B(input_width, rows)
 * where A = weight (row-major, treated as col-major transposed),
 *       B = input  (row-major, treated as col-major transposed).
 *
 * Using cublasSgemm with CUBLAS_OP_T for A and CUBLAS_OP_N for B:
 *   C = alpha * op(A) * op(B) + beta * C
 *   op(A) = A^T  (input_width x output_width) -> we pass weight as (output_width x input_width) with OP_T
 *   op(B) = B    (input_width x rows)
 *
 * Actually simpler: since both input and weight are row-major:
 *   output = input * weight^T
 * In column-major terms:
 *   output^T = weight * input^T
 *   C(ow, r) = A(ow, iw) * B(iw, r)
 *   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, ow, r, iw, &alpha, weight, ow, input, iw, &beta, output, ow)
 *
 * Wait — weight is stored as [output_width, input_width] row-major.
 * In column-major that's [input_width, output_width].
 * We want weight * input^T where weight is (ow x iw) and input^T is (iw x r).
 * Column-major: weight stored as (iw x ow), so we need OP_T on it to get (ow x iw).
 * input stored as (iw x r) in col-major (since row-major [r, iw] = col-major [iw, r]).
 *
 * cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, ow, r, iw, alpha, weight, iw, input, iw, beta, output, ow)
 */
static int cuda_matmul(void *opaque, const float *input, float *output, size_t rows,
                       size_t input_width, size_t output_width, const float *weight,
                       const float *bias, char *error, size_t error_capacity) {
    if (rows > UINT32_MAX || input_width > UINT32_MAX || output_width > UINT32_MAX) {
        set_error(error, error_capacity, "CUDA matmul dimensions are too large");
        return -1;
    }
    cuda_backend_state *state = static_cast<cuda_backend_state *>(opaque);

    const size_t input_bytes = rows * input_width * sizeof(float);
    const size_t weight_bytes = input_width * output_width * sizeof(float);
    const size_t output_bytes = rows * output_width * sizeof(float);
    const size_t bias_bytes = output_width * sizeof(float);

    /* Upload weights (cached after first call). */
    float *d_weight = nullptr;
    if (cached_weight(state, weight, weight_bytes, &d_weight, error, error_capacity) != 0)
        return -1;

    float *d_bias = nullptr;
    if (bias != nullptr &&
        cached_weight(state, bias, bias_bytes, &d_bias, error, error_capacity) != 0)
        return -1;

    /* IO buffer: input region + output region. */
    const size_t io_needed = input_bytes + output_bytes;
    if (ensure_io(state, io_needed, error, error_capacity) != 0) return -1;

    float *d_input = state->io_buffer;
    float *d_output = state->io_buffer + rows * input_width;

    /* Async H2D for input. */
    if (cuda_error(cudaMemcpyAsync(d_input, input, input_bytes,
                                   cudaMemcpyHostToDevice, state->stream),
                   error, error_capacity) != 0)
        return -1;

    /* cuBLAS sgemm: output^T = weight * input^T (see comment above). */
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSetStream(state->cublas, state->stream);
    cublasStatus_t status = cublasSgemm(
        state->cublas,
        CUBLAS_OP_T,     /* op(A) = weight^T: (ow x iw) */
        CUBLAS_OP_N,     /* op(B) = input^T as stored: (iw x r) */
        (int)output_width,
        (int)rows,
        (int)input_width,
        &alpha,
        d_weight, (int)input_width,   /* A: (iw x ow) in col-major, lda = iw */
        d_input, (int)input_width,    /* B: (iw x r) in col-major, ldb = iw */
        &beta,
        d_output, (int)output_width); /* C: (ow x r) in col-major, ldc = ow */
    if (cublas_error(status, error, error_capacity) != 0) return -1;

    /* Add bias if present: output[row, col] += bias[col].
     * Use a simple kernel or cublasSger.  For now, a small kernel. */
    if (d_bias != nullptr) {
        /* cublasSger: A = alpha * x * y^T + A
         * We want to add bias to each row.  Treat output as (ow x r) col-major.
         * bias is (ow x 1), ones is (r x 1).
         * A += bias * ones^T  =>  each column of A gets bias added.
         * But output is (ow x r) col-major = (r x ow) row-major.
         * Each column in col-major is a row in row-major... no.
         * Col-major (ow x r): column j has ow elements = output row j transposed.
         * We want output[row][col] += bias[col].
         * In col-major (ow x r): element (col, row) += bias[col].
         * So we add bias (ow x 1) to each column: A += bias * 1^T.
         * cublasSger(handle, ow, r, &alpha, bias, 1, ones, 1, A, ow).
         * We need a ones vector of length r.  Use a small static buffer. */
        /* Simpler: just do it with a tiny kernel or accept the overhead.
         * For now, copy bias addition to a custom approach. */
        /* Actually, let's use cublasSger with a ones vector. */
        static float *d_ones = nullptr;
        static size_t ones_capacity = 0;
        if (ones_capacity < rows) {
            if (d_ones) cudaFree(d_ones);
            size_t cap = (rows + 255u) & ~255u;
            if (cuda_error(cudaMalloc(&d_ones, cap * sizeof(float)), error, error_capacity) != 0)
                return -1;
            /* Fill with ones. */
            std::vector<float> h_ones(cap, 1.0f);
            cudaMemcpyAsync(d_ones, h_ones.data(), cap * sizeof(float),
                            cudaMemcpyHostToDevice, state->stream);
            ones_capacity = cap;
        }
        const float one = 1.0f;
        status = cublasSger(state->cublas,
                            (int)output_width, (int)rows,
                            &one,
                            d_bias, 1,
                            d_ones, 1,
                            d_output, (int)output_width);
        if (cublas_error(status, error, error_capacity) != 0) return -1;
    }

    /* D2H for output. */
    if (cuda_error(cudaMemcpyAsync(output, d_output, output_bytes,
                                   cudaMemcpyDeviceToHost, state->stream),
                   error, error_capacity) != 0)
        return -1;
    if (cuda_error(cudaStreamSynchronize(state->stream), error, error_capacity) != 0)
        return -1;

    return 0;
}

/*
 * Generic sgemm via cuBLAS.  Mirrors cblas_sgemm row-major semantics:
 *   C[m,n] = alpha * op(A) * op(B) + beta * C
 *
 * cuBLAS is column-major, so we compute the equivalent:
 *   C^T = alpha * op(B)^T * op(A)^T + beta * C^T
 * which in cuBLAS column-major terms is:
 *   cublasSgemm(handle, opB, opA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)
 */
static int cuda_sgemm(void *opaque, int trans_a, int trans_b,
                      size_t m, size_t n, size_t k,
                      float alpha,
                      const float *a, size_t lda,
                      const float *b, size_t ldb,
                      float beta,
                      float *c, size_t ldc,
                      char *error, size_t error_capacity) {
    if (m > UINT32_MAX || n > UINT32_MAX || k > UINT32_MAX ||
        lda > UINT32_MAX || ldb > UINT32_MAX || ldc > UINT32_MAX) {
        set_error(error, error_capacity, "CUDA sgemm dimensions are too large");
        return -1;
    }
    cuda_backend_state *state = static_cast<cuda_backend_state *>(opaque);

    /* Row-major: A stored as (trans_a ? k×m : m×k) with row stride lda.
     * The pointer may be an offset into a larger buffer (e.g. qkv + head*hw),
     * so we copy only the actual columns and pack them on the device. */
    const size_t a_rows = trans_a ? k : m;
    const size_t a_cols = trans_a ? m : k;  /* actual data columns per row */
    const size_t b_rows = trans_b ? n : k;
    const size_t b_cols = trans_b ? k : n;

    /* Packed device layout: no stride gaps. */
    const size_t a_packed = a_rows * a_cols;
    const size_t b_packed = b_rows * b_cols;
    const size_t c_packed = m * n;
    const size_t io_needed = (a_packed + b_packed + c_packed) * sizeof(float);
    if (ensure_io(state, io_needed, error, error_capacity) != 0) return -1;

    float *d_a = state->io_buffer;
    float *d_b = d_a + a_packed;
    float *d_c = d_b + b_packed;

    cublasSetStream(state->cublas, state->stream);

    /* Copy with src pitch = lda/ldb, dst pitch = packed (a_cols/b_cols). */
    if (cuda_error(cudaMemcpy2DAsync(d_a, a_cols * sizeof(float),
                                     a, lda * sizeof(float),
                                     a_cols * sizeof(float), a_rows,
                                     cudaMemcpyHostToDevice, state->stream),
                   error, error_capacity) != 0) return -1;
    if (cuda_error(cudaMemcpy2DAsync(d_b, b_cols * sizeof(float),
                                     b, ldb * sizeof(float),
                                     b_cols * sizeof(float), b_rows,
                                     cudaMemcpyHostToDevice, state->stream),
                   error, error_capacity) != 0) return -1;
    if (beta != 0.0f) {
        if (cuda_error(cudaMemcpy2DAsync(d_c, n * sizeof(float),
                                         c, ldc * sizeof(float),
                                         n * sizeof(float), m,
                                         cudaMemcpyHostToDevice, state->stream),
                       error, error_capacity) != 0) return -1;
    }

    /* cuBLAS column-major trick for row-major C = alpha*op(A)*op(B) + beta*C:
     *   cublasSgemm(h, opB, opA, n, m, k, alpha, B, ldb_packed, A, lda_packed, beta, C, ldc_packed)
     * With packed data: lda_packed = a_cols, ldb_packed = b_cols, ldc_packed = n. */
    const cublasOperation_t opA = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t opB = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int lda_packed = (int)a_cols;
    const int ldb_packed = (int)b_cols;
    cublasStatus_t status = cublasSgemm(
        state->cublas, opA, opB,
        (int)n, (int)m, (int)k,
        &alpha,
        d_b, ldb_packed,
        d_a, lda_packed,
        &beta,
        d_c, (int)n);
    if (cublas_error(status, error, error_capacity) != 0) return -1;

    /* Copy result back with dst pitch = ldc, src pitch = n (packed). */
    if (cuda_error(cudaMemcpy2DAsync(c, ldc * sizeof(float),
                                     d_c, n * sizeof(float),
                                     n * sizeof(float), m,
                                     cudaMemcpyDeviceToHost, state->stream),
                   error, error_capacity) != 0) return -1;
    if (cuda_error(cudaStreamSynchronize(state->stream), error, error_capacity) != 0)
        return -1;
    return 0;
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
        if (std::fabs(output[i] - expected[i]) > 1.0e-4f) {
            std::snprintf(error, error_capacity,
                          "CUDA matmul self-test mismatch at %zu: got %f want %f",
                          i, output[i], expected[i]);
            return -1;
        }
    }
    /* Test sgemm: same computation via the generic interface.
     * C[2,4] = A[2,3] * B[4,3]^T + bias (bias added separately). */
    float sgemm_out[8] = {0};
    if (cuda_sgemm(opaque, 0, 1, 2u, 4u, 3u, 1.0f,
                   input, 3u, weight, 3u, 0.0f,
                   sgemm_out, 4u, error, error_capacity) != 0) return -1;
    /* sgemm_out should equal expected - bias (no bias in sgemm). */
    const float expected_nobias[8] = {1.0f, 2.0f, 3.0f, 6.0f, -1.0f, 0.5f, 2.0f, 1.5f};
    for (size_t i = 0; i < 8u; ++i) {
        if (std::fabs(sgemm_out[i] - expected_nobias[i]) > 1.0e-4f) {
            std::snprintf(error, error_capacity,
                          "CUDA sgemm self-test mismatch at %zu: got %f want %f",
                          i, sgemm_out[i], expected_nobias[i]);
            return -1;
        }
    }
    return 0;
}

static void cuda_close(void *opaque) {
    cuda_backend_state *state = static_cast<cuda_backend_state *>(opaque);
    if (state == nullptr) return;
    for (const cuda_cached_buffer &cached : state->weights)
        cudaFree(cached.device_pointer);
    if (state->io_buffer != nullptr) cudaFree(state->io_buffer);
    cublasDestroy(state->cublas);
    cudaStreamDestroy(state->stream);
    delete state;
}

extern "C" int mynah_backend_cuda_open(void **state_out, mynah_backend_matmul_fn *matmul,
                                       mynah_backend_sgemm_fn *sgemm,
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

    /* Detect unified memory (GB10 Grace Blackwell). */
    int unified = 0;
    cudaDeviceGetAttribute(&unified, cudaDevAttrPageableMemoryAccess, 0);

    cuda_backend_state *state = new (std::nothrow) cuda_backend_state();
    if (state == nullptr) {
        set_error(error, error_capacity, "out of memory creating CUDA backend");
        return -1;
    }
    state->io_buffer = nullptr;
    state->io_capacity = 0;
    state->unified_memory = unified;

    if (cuda_error(cudaStreamCreate(&state->stream), error, error_capacity) != 0) {
        delete state;
        return -1;
    }
    cublasStatus_t cs = cublasCreate(&state->cublas);
    if (cs != CUBLAS_STATUS_SUCCESS) {
        std::snprintf(error, error_capacity, "cuBLAS init failed: %d", (int)cs);
        cudaStreamDestroy(state->stream);
        delete state;
        return -1;
    }
    /* Use tensor cores when available (TF32 on Ampere+, FP32 on Blackwell). */
    cublasSetMathMode(state->cublas, CUBLAS_DEFAULT_MATH);

    *state_out = state;
    *matmul = cuda_matmul;
    *sgemm = cuda_sgemm;
    *close = cuda_close;
    *self_test = cuda_self_test;
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    return 0;
}
