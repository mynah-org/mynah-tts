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

@interface MynahMetalState : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> matmul_pipeline;
@property(nonatomic, strong) NSMutableArray *weight_cache;
- (id<MTLBuffer>)bufferForHostPointer:(const void *)pointer length:(NSUInteger)length;
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
@end

static void set_error(char *error, size_t capacity, NSString *message) {
    if (error == NULL || capacity == 0) return;
    snprintf(error, capacity, "%s", message.UTF8String == NULL ? "Metal error" :
             message.UTF8String);
}

static NSString *metal_shader_source(void) {
    return @"#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "struct MatmulParams { uint rows; uint input_width; uint output_width; };\n"
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
        if (rows > UINT32_MAX || input_width > UINT32_MAX || output_width > UINT32_MAX ||
            rows > SIZE_MAX / output_width || rows > UINT32_MAX / output_width ||
            input_width > SIZE_MAX / output_width) {
            set_error(error, error_capacity, @"Metal matmul dimensions are too large");
            return -1;
        }
        const NSUInteger input_bytes = rows * input_width * sizeof(float);
        const NSUInteger weight_bytes = input_width * output_width * sizeof(float);
        const NSUInteger output_bytes = rows * output_width * sizeof(float);
        const NSUInteger bias_bytes = output_width * sizeof(float);
        id<MTLBuffer> input_buffer = [state.device newBufferWithBytes:input
                                                                  length:input_bytes
                                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> weight_buffer = [state bufferForHostPointer:weight length:weight_bytes];
        id<MTLBuffer> bias_buffer = bias == NULL ? nil :
            [state bufferForHostPointer:bias length:bias_bytes];
        id<MTLBuffer> output_buffer = [state.device newBufferWithLength:output_bytes
                                                                   options:MTLResourceStorageModeShared];
        if (input_buffer == nil || weight_buffer == nil || output_buffer == nil ||
            (bias != NULL && bias_buffer == nil)) {
            set_error(error, error_capacity, @"Metal could not allocate matmul buffers");
            return -1;
        }
        metal_matmul_params params = {(uint32_t)rows, (uint32_t)input_width,
                                      (uint32_t)output_width};
        id<MTLBuffer> params_buffer = [state.device newBufferWithBytes:&params
                                                                   length:sizeof(params)
                                                                  options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> command = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:state.matmul_pipeline];
        [encoder setBuffer:input_buffer offset:0 atIndex:0];
        [encoder setBuffer:weight_buffer offset:0 atIndex:1];
        [encoder setBuffer:bias_buffer offset:0 atIndex:2];
        [encoder setBuffer:output_buffer offset:0 atIndex:3];
        [encoder setBuffer:params_buffer offset:0 atIndex:4];
        const NSUInteger thread_width = MIN((NSUInteger)256,
                                             state.matmul_pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreadgroups:MTLSizeMake((rows * output_width + thread_width - 1u) /
                                                   thread_width, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(thread_width, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            set_error(error, error_capacity, command.error.localizedDescription);
            return -1;
        }
        memcpy(output, output_buffer.contents, output_bytes);
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
        id<MTLFunction> function = [library newFunctionWithName:@"mynah_matmul"];
        NSError *pipeline_error = nil;
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (pipeline == nil) {
            set_error(error, error_capacity, pipeline_error.localizedDescription);
            return -1;
        }
        MynahMetalState *state = [MynahMetalState new];
        state.device = device;
        state.queue = [device newCommandQueue];
        state.matmul_pipeline = pipeline;
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
