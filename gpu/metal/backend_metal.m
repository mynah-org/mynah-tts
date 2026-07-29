#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t rows;
    uint32_t input_width;
    uint32_t output_width;
} metal_matmul_params;

@class MynahMetalMPSMatmul;

/* Pooled reusable buffers eliminate per-call allocation, which is the
 * dominant overhead in the per-op dispatch model.  Weight buffers are
 * cached by host pointer; activation/IO buffers are pre-allocated at
 * maximum size and reused. */
@interface MynahMetalState : NSObject
@property(nonatomic, strong) NSMutableArray *params_buffers;
@property(nonatomic, assign) NSUInteger params_cursor;
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> matmul_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> tiled_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> row_tiled_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> row_simd_pipeline;
@property(nonatomic, strong) NSMutableArray *weight_cache;
@property(nonatomic, strong) NSMutableArray *mps_matmul_cache;
@property(nonatomic, assign) BOOL use_mps;
@property(nonatomic, assign) BOOL use_simd_matvec;
@property(nonatomic, strong) id<MTLBuffer> scalar_buffer;
@property(nonatomic, strong) id<MTLBuffer> io_buffer;      /* reusable input/output */
@property(nonatomic, strong) id<MTLBuffer> params_buffer;   /* reusable params */
@property(nonatomic, strong) id<MTLBuffer> conv_columns_buffer;
@property(nonatomic, assign) NSUInteger io_capacity;
@property(nonatomic, assign) NSUInteger conv_columns_capacity;
- (id<MTLBuffer>)bufferForHostPointer:(const void *)pointer length:(NSUInteger)length;
- (id<MTLBuffer>)ioBufferWithLength:(NSUInteger)length;
- (id<MTLBuffer>)convColumnsBufferWithLength:(NSUInteger)length;
- (MynahMetalMPSMatmul *)mpsMatmulForRows:(NSUInteger)rows
                              inputWidth:(NSUInteger)input_width
                             outputWidth:(NSUInteger)output_width;
@end

@interface MynahMetalCachedBuffer : NSObject
@property(nonatomic, assign) const void *host_pointer;
@property(nonatomic, assign) NSUInteger length;
@property(nonatomic, strong) id<MTLBuffer> buffer;
@end

@interface MynahMetalMPSMatmul : NSObject
@property(nonatomic, assign) NSUInteger rows;
@property(nonatomic, assign) NSUInteger input_width;
@property(nonatomic, assign) NSUInteger output_width;
@property(nonatomic, strong) MPSMatrixMultiplication *kernel;
@property(nonatomic, strong) MPSMatrixDescriptor *input_descriptor;
@property(nonatomic, strong) MPSMatrixDescriptor *weight_descriptor;
@property(nonatomic, strong) MPSMatrixDescriptor *output_descriptor;
@end

@implementation MynahMetalMPSMatmul
@end

@implementation MynahMetalCachedBuffer
@end

@implementation MynahMetalState
- (MynahMetalMPSMatmul *)mpsMatmulForRows:(NSUInteger)rows
                              inputWidth:(NSUInteger)input_width
                             outputWidth:(NSUInteger)output_width {
    for (MynahMetalMPSMatmul *cached in self.mps_matmul_cache) {
        if (cached.rows == rows && cached.input_width == input_width &&
            cached.output_width == output_width) return cached;
    }
    MPSMatrixDescriptor *input_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:input_width
                                              rowBytes:input_width * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *weight_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:output_width columns:input_width
                                              rowBytes:input_width * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *output_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:output_width
                                              rowBytes:output_width * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrixMultiplication *kernel =
        [[MPSMatrixMultiplication alloc] initWithDevice:self.device
                                         transposeLeft:NO transposeRight:YES
                                            resultRows:rows resultColumns:output_width
                                       interiorColumns:input_width alpha:1.0 beta:1.0];
    if (input_descriptor == nil || weight_descriptor == nil ||
        output_descriptor == nil || kernel == nil) return nil;
    MynahMetalMPSMatmul *cached = [MynahMetalMPSMatmul new];
    cached.rows = rows;
    cached.input_width = input_width;
    cached.output_width = output_width;
    cached.kernel = kernel;
    cached.input_descriptor = input_descriptor;
    cached.weight_descriptor = weight_descriptor;
    cached.output_descriptor = output_descriptor;
    if (self.mps_matmul_cache == nil) self.mps_matmul_cache = [NSMutableArray array];
    [self.mps_matmul_cache addObject:cached];
    return cached;
}

- (id<MTLBuffer>)bufferForHostPointer:(const void *)pointer length:(NSUInteger)length {
    for (MynahMetalCachedBuffer *cached in self.weight_cache) {
        if (cached.host_pointer == pointer && cached.length == length) return cached.buffer;
    }
    id<MTLBuffer> buffer = [self.device newBufferWithBytes:pointer length:length
                                                    options:MTLResourceStorageModeShared];
    if (buffer == nil) return nil;
    MynahMetalCachedBuffer *cached = [MynahMetalCachedBuffer new];
    cached.host_pointer = pointer;
    cached.length = length;
    cached.buffer = buffer;
    [self.weight_cache addObject:cached];
    return buffer;
}

- (id<MTLBuffer>)ioBufferWithLength:(NSUInteger)length {
    if (self.io_buffer == nil || self.io_capacity < length) {
        NSUInteger cap = length;
        if (cap < 4u * 1024u * 1024u) cap = 4u * 1024u * 1024u; /* 4 MB floor */
        self.io_buffer = [self.device newBufferWithLength:cap
                                                  options:MTLResourceStorageModeShared];
        self.io_capacity = cap;
    }
    return self.io_buffer;
}

- (id<MTLBuffer>)convColumnsBufferWithLength:(NSUInteger)length {
    if (self.conv_columns_buffer == nil || self.conv_columns_capacity < length) {
        self.conv_columns_buffer = [self.device newBufferWithLength:length
                                                               options:MTLResourceStorageModeShared];
        self.conv_columns_capacity = self.conv_columns_buffer == nil ? 0 : length;
    }
    return self.conv_columns_buffer;
}
@end

static void set_error(char *error, size_t capacity, NSString *message) {
    if (error == NULL || capacity == 0) return;
    snprintf(error, capacity, "%s", message.UTF8String == NULL ? "Metal error" :
             message.UTF8String);
}

/* Keep the runtime shader in sync with gpu/metal/matmul.metal.  This version
 * uses cooperative threadgroup tiles instead of rereading global memory for
 * every output element. */
static NSString *metal_shader_source_cooperative(void) {
    return
    @"#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct MatmulParams { uint rows; uint input_width; uint output_width; };\n"
    "constant uint TILE_M = 4; constant uint TILE_N = 64; constant uint TILE_K = 32;\n"
    "kernel void mynah_matmul_row_simd(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]], device const float *bias [[buffer(2)]], device float *output [[buffer(3)]], constant MatmulParams &p [[buffer(4)]], uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sgid [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]]) {\n"
    " uint row=tgid.x*nsg+sgid; if(row>=p.output_width) return; float acc=0.0f; uint n4=p.input_width/4u; device const float4 *in4=(device const float4 *)input; device const float4 *w4=(device const float4 *)(weight+(ulong)row*p.input_width);\n"
    " for(uint c=lane;c<n4;c+=32u) acc+=dot(w4[c],in4[c]); for(uint k=n4*4u+lane;k<p.input_width;k+=32u) acc+=input[k]*weight[(ulong)row*p.input_width+k]; acc=simd_sum(acc); if(lane==0) output[row]=acc+(bias==nullptr?0.0f:bias[row]);\n"
    "}\n"
    "kernel void mynah_matmul_row_tiled(device const float *input [[buffer(0)]],\n"
    " device const float *weight [[buffer(1)]], device const float *bias [[buffer(2)]],\n"
    " device float *output [[buffer(3)]], constant MatmulParams &p [[buffer(4)]],\n"
    " uint2 gid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {\n"
    " const uint col = gid.x * 64u + tid;\n"
    " threadgroup float input_tile[TILE_K]; float acc = 0.0f;\n"
    " for (uint k0 = 0; k0 < p.input_width; k0 += TILE_K) {\n"
    "  if (tid < TILE_K) input_tile[tid] = k0 + tid < p.input_width ? input[k0 + tid] : 0.0f;\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  if (col < p.output_width) {\n"
    "   const uint base = col * p.input_width + k0;\n"
    "   for (uint k = 0; k < TILE_K && k0 + k < p.input_width; ++k)\n"
    "    acc += input_tile[k] * weight[base + k];\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    " }\n"
    " if (col < p.output_width) output[col] = acc + (bias == nullptr ? 0.0f : bias[col]);\n"
    "}\n"
    "kernel void mynah_matmul_tiled(device const float *input [[buffer(0)]],\n"
    " device const float *weight [[buffer(1)]], device const float *bias [[buffer(2)]],\n"
    " device float *output [[buffer(3)]], constant MatmulParams &p [[buffer(4)]],\n"
    " uint2 gid [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]]) {\n"
    " const uint row_start = gid.y * TILE_M; const uint col_start = gid.x * TILE_N;\n"
    " const uint tx = lid.x; const uint ty = lid.y; const uint col = col_start + tx;\n"
    " const uint row = row_start + ty; const uint linear = ty * TILE_N + tx;\n"
    " threadgroup float input_tile[TILE_M * TILE_K];\n"
    " threadgroup float weight_tile[TILE_N * TILE_K]; float acc = 0.0f;\n"
    " for (uint k0 = 0; k0 < p.input_width; k0 += TILE_K) {\n"
    "  for (uint i = linear; i < TILE_M * TILE_K; i += TILE_M * TILE_N) {\n"
    "   const uint tile_row = i / TILE_K; const uint tile_k = i - tile_row * TILE_K;\n"
    "   const uint src_row = row_start + tile_row; const uint src_k = k0 + tile_k;\n"
    "   input_tile[i] = (src_row < p.rows && src_k < p.input_width) ? input[src_row * p.input_width + src_k] : 0.0f;\n"
    "  }\n"
    "  for (uint i = linear; i < TILE_N * TILE_K; i += TILE_M * TILE_N) {\n"
    "   const uint tile_col = i / TILE_K; const uint tile_k = i - tile_col * TILE_K;\n"
    "   const uint src_col = col_start + tile_col; const uint src_k = k0 + tile_k;\n"
    "   weight_tile[i] = (src_col < p.output_width && src_k < p.input_width) ? weight[src_col * p.input_width + src_k] : 0.0f;\n"
    "  }\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    "  for (uint k = 0; k < TILE_K; ++k) acc += input_tile[ty * TILE_K + k] * weight_tile[tx * TILE_K + k];\n"
    "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    " }\n"
    " if (row < p.rows && col < p.output_width) output[row * p.output_width + col] = acc + (bias == nullptr ? 0.0f : bias[col]);\n"
    "}\n"
    "kernel void mynah_matmul(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]],\n"
    " device const float *bias [[buffer(2)]], device float *output [[buffer(3)]], constant MatmulParams &p [[buffer(4)]],\n"
    " uint index [[thread_position_in_grid]]) {\n"
    " const uint total = p.rows * p.output_width; if (index >= total) return;\n"
    " const uint row = index / p.output_width; const uint column = index - row * p.output_width;\n"
    " const device float *input_row = input + row * p.input_width; const device float *weight_row = weight + column * p.input_width;\n"
    " float value = bias == nullptr ? 0.0f : bias[column];\n"
    " for (uint i = 0; i < p.input_width; ++i) value += input_row[i] * weight_row[i];\n"
    " output[index] = value;\n"
    "}\n";
}

static int metal_matmul(void *opaque, const float *input, float *output, size_t rows,
                        size_t input_width, size_t output_width, const float *weight,
                        const float *bias, char *error, size_t error_capacity) {
    @autoreleasepool {
        MynahMetalState *state = (__bridge MynahMetalState *)opaque;
        if (rows > UINT32_MAX || input_width > UINT32_MAX || output_width > UINT32_MAX) {
            set_error(error, error_capacity, @"Metal matmul dimensions are too large");
            return -1;
        }
        const NSUInteger input_bytes = rows * input_width * sizeof(float);
        const NSUInteger weight_bytes = input_width * output_width * sizeof(float);
        const NSUInteger output_bytes = rows * output_width * sizeof(float);
        const NSUInteger bias_bytes = output_width * sizeof(float);

        /* Weight and bias buffers are cached across calls (resident). */
        id<MTLBuffer> weight_buffer = [state bufferForHostPointer:weight length:weight_bytes];
        id<MTLBuffer> bias_buffer = bias == NULL ? nil :
            [state bufferForHostPointer:bias length:bias_bytes];

        /* IO buffer is pooled: write input directly into shared memory,
         * dispatch, then read output from the same buffer region. */
        const NSUInteger io_needed = input_bytes + output_bytes;
        id<MTLBuffer> io = [state ioBufferWithLength:io_needed];
        if (io == nil || weight_buffer == nil || (bias != NULL && bias_buffer == nil)) {
            set_error(error, error_capacity, @"Metal could not allocate matmul buffers");
            return -1;
        }
        float *io_ptr = (float *)io.contents;
        memcpy(io_ptr, input, input_bytes);

        /* MPS uses the existing resident weight buffers and the pooled IO
         * buffer.  Seed C with the bias so GEMM computes A*B^T + C in one
         * MPS operation without a second bias-dispatch kernel. */
        if (state.use_mps && rows > 1u && input_width > 0 && output_width > 0) {
            MynahMetalMPSMatmul *mps =
                [state mpsMatmulForRows:rows inputWidth:input_width outputWidth:output_width];
            if (mps != nil) {
                float *result = io_ptr + input_bytes / sizeof(float);
                if (bias != NULL) {
                    for (NSUInteger row = 0; row < rows; ++row)
                        memcpy(result + row * output_width, bias, bias_bytes);
                } else {
                    memset(result, 0, output_bytes);
                }
                MPSMatrix *left = [[MPSMatrix alloc] initWithBuffer:io offset:0
                                                            descriptor:mps.input_descriptor];
                MPSMatrix *right = [[MPSMatrix alloc] initWithBuffer:weight_buffer
                                                                 descriptor:mps.weight_descriptor];
                MPSMatrix *destination = [[MPSMatrix alloc] initWithBuffer:io
                                                                      offset:input_bytes
                                                                  descriptor:mps.output_descriptor];
                id<MTLCommandBuffer> command = [state.queue commandBuffer];
                if (left != nil && right != nil && destination != nil && command != nil) {
                    [mps.kernel encodeToCommandBuffer:command leftMatrix:left
                                         rightMatrix:right resultMatrix:destination];
                    [command commit];
                    [command waitUntilCompleted];
                    if (command.status != MTLCommandBufferStatusCompleted) {
                        set_error(error, error_capacity, command.error.localizedDescription);
                        return -1;
                    }
                    memcpy(output, result, output_bytes);
                    return 0;
                }
            }
        }

        metal_matmul_params params = {(uint32_t)rows, (uint32_t)input_width,
                                      (uint32_t)output_width};
        if (state.params_buffer == nil) {
            state.params_buffer = [state.device newBufferWithLength:sizeof(params)
                                                            options:MTLResourceStorageModeShared];
        }
        memcpy(state.params_buffer.contents, &params, sizeof(params));

        id<MTLCommandBuffer> command = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];

        /* Use the tiled kernel for larger outputs, simple for tiny. */
        const NSUInteger total = rows * output_width;
        if (rows == 1u && state.use_simd_matvec && state.row_simd_pipeline != nil &&
            state.row_simd_pipeline.maxTotalThreadsPerThreadgroup >= 32u) {
            [encoder setComputePipelineState:state.row_simd_pipeline];
            [encoder setBuffer:io offset:0 atIndex:0];
            [encoder setBuffer:weight_buffer offset:0 atIndex:1];
            [encoder setBuffer:bias_buffer offset:0 atIndex:2];
            [encoder setBuffer:io offset:input_bytes atIndex:3];
            [encoder setBuffer:state.params_buffer offset:0 atIndex:4];
            const NSUInteger nsg = MIN((NSUInteger)8,
                                       state.row_simd_pipeline.maxTotalThreadsPerThreadgroup / 32u);
            [encoder dispatchThreadgroups:MTLSizeMake((output_width + nsg - 1u) / nsg, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
        } else if (rows == 1u && output_width >= 64u && state.row_tiled_pipeline != nil &&
            state.row_tiled_pipeline.maxTotalThreadsPerThreadgroup >= 64u) {
            [encoder setComputePipelineState:state.row_tiled_pipeline];
            [encoder setBuffer:io offset:0 atIndex:0];
            [encoder setBuffer:weight_buffer offset:0 atIndex:1];
            [encoder setBuffer:bias_buffer offset:0 atIndex:2];
            [encoder setBuffer:io offset:input_bytes atIndex:3];
            [encoder setBuffer:state.params_buffer offset:0 atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake((output_width + 63u) / 64u, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        } else if (output_width >= 64u && state.tiled_pipeline != nil) {
            [encoder setComputePipelineState:state.tiled_pipeline];
            [encoder setBuffer:io offset:0 atIndex:0];
            [encoder setBuffer:weight_buffer offset:0 atIndex:1];
            [encoder setBuffer:bias_buffer offset:0 atIndex:2];
            [encoder setBuffer:io offset:input_bytes atIndex:3];
            [encoder setBuffer:state.params_buffer offset:0 atIndex:4];
            const NSUInteger tgx = MIN(64u, state.tiled_pipeline.maxTotalThreadsPerThreadgroup);
            const NSUInteger tgy = MIN(4u, state.tiled_pipeline.maxTotalThreadsPerThreadgroup / tgx);
            [encoder dispatchThreadgroups:MTLSizeMake((output_width + 63u) / 64u,
                                                       (rows + 3u) / 4u, 1)
                     threadsPerThreadgroup:MTLSizeMake(tgx, tgy, 1)];
        } else {
            [encoder setComputePipelineState:state.matmul_pipeline];
            [encoder setBuffer:io offset:0 atIndex:0];
            [encoder setBuffer:weight_buffer offset:0 atIndex:1];
            [encoder setBuffer:bias_buffer offset:0 atIndex:2];
            [encoder setBuffer:io offset:input_bytes atIndex:3];
            [encoder setBuffer:state.params_buffer offset:0 atIndex:4];
            const NSUInteger tw = MIN((NSUInteger)256,
                                       state.matmul_pipeline.maxTotalThreadsPerThreadgroup);
            [encoder dispatchThreadgroups:MTLSizeMake((total + tw - 1u) / tw, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            set_error(error, error_capacity, command.error.localizedDescription);
            return -1;
        }
        memcpy(output, io_ptr + input_bytes / sizeof(float), output_bytes);
        return 0;
    }
}

static int metal_self_test(void *opaque, char *error, size_t error_capacity) {
    const float input[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 2.0f};
    const float weight[12] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float bias[4] = {0.5f, -0.5f, 1.0f, 2.0f};
    const float expected[8] = {1.5f, 1.5f, 4.0f, 8.0f, -0.5f, 0.0f, 3.0f, 3.5f};
    float output[8] = {0};
    float tiled_weight[128] = {0};
    float tiled_bias[64] = {0};
    float tiled_output[64] = {0};
    if (metal_matmul(opaque, input, output, 2u, 3u, 4u, weight, bias,
                     error, error_capacity) != 0) return -1;
    for (size_t i = 0; i < 8u; ++i) {
        if (fabsf(output[i] - expected[i]) > 1.0e-5f) {
            snprintf(error, error_capacity, "Metal backend self-test mismatch at %zu", i);
            return -1;
        }
    }
    for (size_t column = 0; column < 64u; ++column) {
        tiled_weight[column * 2u] = 1.0f;
        tiled_bias[column] = (float)column;
    }
    if (metal_matmul(opaque, input, tiled_output, 1u, 2u, 64u,
                     tiled_weight, tiled_bias, error, error_capacity) != 0) {
        return -1;
    }
    for (size_t column = 0; column < 64u; ++column) {
        if (fabsf(tiled_output[column] - (1.0f + (float)column)) > 1.0e-5f) {
            snprintf(error, error_capacity,
                     "Metal tiled self-test mismatch at %zu", column);
            return -1;
        }
    }
    return 0;
}

static void metal_close(void *opaque) {
    MynahMetalState *state = (__bridge_transfer MynahMetalState *)opaque;
    (void)state;
}

int mynah_backend_metal_open(void **state_out, mynah_backend_matmul_fn *matmul,
                             mynah_backend_sgemm_fn *sgemm,
                             mynah_backend_close_fn *close,
                             mynah_backend_self_test_fn *self_test,
                             char *error, size_t error_capacity) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            set_error(error, error_capacity, @"Metal device is unavailable");
            return -1;
        }
        NSError *library_error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:metal_shader_source_cooperative()
                                                        options:nil error:&library_error];
        if (library == nil) {
            NSString *message = [NSString stringWithFormat:@"Metal shader compilation failed: %@",
                                  library_error.localizedDescription];
            set_error(error, error_capacity, message);
            return -1;
        }
        id<MTLFunction> simple_fn = [library newFunctionWithName:@"mynah_matmul"];
        id<MTLFunction> tiled_fn = [library newFunctionWithName:@"mynah_matmul_tiled"];
        NSError *pipeline_error = nil;
        id<MTLComputePipelineState> simple_pipeline =
            [device newComputePipelineStateWithFunction:simple_fn error:&pipeline_error];
        if (simple_pipeline == nil) {
            set_error(error, error_capacity, pipeline_error.localizedDescription);
            return -1;
        }
        id<MTLComputePipelineState> tiled_pipeline = nil;
        if (tiled_fn != nil) {
            tiled_pipeline = [device newComputePipelineStateWithFunction:tiled_fn
                                                                   error:&pipeline_error];
        }
        id<MTLComputePipelineState> row_tiled_pipeline = nil;
        id<MTLFunction> row_tiled_fn = [library newFunctionWithName:@"mynah_matmul_row_tiled"];
        if (row_tiled_fn != nil) {
            row_tiled_pipeline = [device newComputePipelineStateWithFunction:row_tiled_fn
                                                                         error:&pipeline_error];
        }
        id<MTLComputePipelineState> row_simd_pipeline = nil;
        id<MTLFunction> row_simd_fn = [library newFunctionWithName:@"mynah_matmul_row_simd"];
        if (row_simd_fn != nil) {
            row_simd_pipeline = [device newComputePipelineStateWithFunction:row_simd_fn
                                                                        error:&pipeline_error];
        }
        MynahMetalState *state = [MynahMetalState new];
        state.device = device;
        state.queue = [device newCommandQueue];
        state.matmul_pipeline = simple_pipeline;
        state.tiled_pipeline = tiled_pipeline;
        state.row_tiled_pipeline = row_tiled_pipeline;
        state.row_simd_pipeline = row_simd_pipeline;
        const char *simd_env = getenv("MYNAH_METAL_SIMD_MATVEC");
        state.use_simd_matvec = simd_env == NULL || strcmp(simd_env, "0") != 0;
        state.weight_cache = [NSMutableArray array];
        state.mps_matmul_cache = [NSMutableArray array];
        const char *mps_env = getenv("MYNAH_METAL_MPS");
        state.use_mps = mps_env == NULL || strcmp(mps_env, "0") != 0;
        if (state.queue == nil) {
            set_error(error, error_capacity, @"Metal could not create a command queue");
            return -1;
        }
        *state_out = (__bridge_retained void *)state;
        *matmul = metal_matmul;
        *sgemm = NULL; /* falls back to CPU sgemm in backend.c */
        *close = metal_close;
        *self_test = metal_self_test;
        if (error != NULL && error_capacity > 0) error[0] = '\0';
        return 0;
    }
}
