#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

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

/* Pooled reusable buffers eliminate per-call allocation, which is the
 * dominant overhead in the per-op dispatch model.  Weight buffers are
 * cached by host pointer; activation/IO buffers are pre-allocated at
 * maximum size and reused. */
@interface MynahMetalState : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> matmul_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> tiled_pipeline;
@property(nonatomic, strong) NSMutableArray *weight_cache;
@property(nonatomic, strong) id<MTLBuffer> io_buffer;      /* reusable input/output */
@property(nonatomic, strong) id<MTLBuffer> params_buffer;   /* reusable params */
@property(nonatomic, assign) NSUInteger io_capacity;
- (id<MTLBuffer>)bufferForHostPointer:(const void *)pointer length:(NSUInteger)length;
- (id<MTLBuffer>)ioBufferWithLength:(NSUInteger)length;
@end

@interface MynahMetalCachedBuffer : NSObject
@property(nonatomic, assign) const void *host_pointer;
@property(nonatomic, assign) NSUInteger length;
@property(nonatomic, strong) id<MTLBuffer> buffer;
@end

@implementation MynahMetalCachedBuffer
@end

@implementation MynahMetalState
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
@end

static void set_error(char *error, size_t capacity, NSString *message) {
    if (error == NULL || capacity == 0) return;
    snprintf(error, capacity, "%s", message.UTF8String == NULL ? "Metal error" :
             message.UTF8String);
}

static NSString *metal_shader_source(void) {
    return
@"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct MatmulParams { uint rows; uint input_width; uint output_width; };\n"
"\n"
"kernel void mynah_matmul_tiled(device const float *input   [[buffer(0)]],\n"
"                               device const float *weight  [[buffer(1)]],\n"
"                               device const float *bias    [[buffer(2)]],\n"
"                               device float       *output  [[buffer(3)]],\n"
"                               constant MatmulParams &p    [[buffer(4)]],\n"
"                               uint2 gid [[threadgroup_position_in_grid]],\n"
"                               uint2 lid [[thread_position_in_threadgroup]],\n"
"                               uint2 tgs [[threads_per_threadgroup]]) {\n"
"    const uint TILE_M = 4; const uint TILE_N = 64;\n"
"    const uint row_start = gid.y * TILE_M;\n"
"    const uint col_start = gid.x * TILE_N;\n"
"    const uint col = col_start + lid.x;\n"
"    const uint row = row_start + lid.y;\n"
"    float acc = 0.0f;\n"
"    if (row < p.rows && col < p.output_width) {\n"
"        for (uint k = 0; k < p.input_width; ++k)\n"
"            acc += input[row * p.input_width + k] * weight[col * p.input_width + k];\n"
"        if (bias != nullptr) acc += bias[col];\n"
"        output[row * p.output_width + col] = acc;\n"
"    }\n"
"}\n"
"\n"
"kernel void mynah_matmul(device const float *input [[buffer(0)]],\n"
" device const float *weight [[buffer(1)]], device const float *bias [[buffer(2)]],\n"
" device float *output [[buffer(3)]], constant MatmulParams &p [[buffer(4)]],\n"
" uint index [[thread_position_in_grid]]) {\n"
" uint total = p.rows * p.output_width; if (index >= total) return;\n"
" uint row = index / p.output_width; uint column = index - row * p.output_width;\n"
" device const float *in = input + row * p.input_width;\n"
" device const float *w = weight + column * p.input_width;\n"
" float value = bias == nullptr ? 0.0f : bias[column];\n"
" for (uint i = 0; i < p.input_width; ++i) value += in[i] * w[i];\n"
" output[index] = value; }\n";
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
        if (output_width >= 64u && state.tiled_pipeline != nil) {
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
    if (metal_matmul(opaque, input, output, 2u, 3u, 4u, weight, bias,
                     error, error_capacity) != 0) return -1;
    for (size_t i = 0; i < 8u; ++i) {
        if (fabsf(output[i] - expected[i]) > 1.0e-5f) {
            snprintf(error, error_capacity, "Metal backend self-test mismatch at %zu", i);
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
        id<MTLLibrary> library = [device newLibraryWithSource:metal_shader_source()
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
        MynahMetalState *state = [MynahMetalState new];
        state.device = device;
        state.queue = [device newCommandQueue];
        state.matmul_pipeline = simple_pipeline;
        state.tiled_pipeline = tiled_pipeline;
        state.weight_cache = [NSMutableArray array];
        if (state.queue == nil) {
            set_error(error, error_capacity, @"Metal could not create a command queue");
            return -1;
        }
        *state_out = (__bridge_retained void *)state;
        *matmul = metal_matmul;
        *close = metal_close;
        *self_test = metal_self_test;
        if (error != NULL && error_capacity > 0) error[0] = '\0';
        return 0;
    }
}
