#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

#include "backend.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* The main Metal translation unit owns this state class.  This declaration
 * deliberately mirrors only the stable resident-buffer surface. */
@interface MynahMetalCachedBuffer : NSObject
@property(nonatomic, assign) const void *host_pointer;
@property(nonatomic, assign) NSUInteger length;
@property(nonatomic, strong) id<MTLBuffer> buffer;
@end

@interface MynahMetalState : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> matmul_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> tiled_pipeline;
@property(nonatomic, strong) NSMutableArray *weight_cache;
@property(nonatomic, strong) id<MTLBuffer> params_buffer;
- (id<MTLBuffer>)ioBufferWithLength:(NSUInteger)length;
- (id<MTLBuffer>)bufferForHostPointer:(const void *)pointer length:(NSUInteger)length;
@end

typedef struct {
    uint32_t rows;
    uint32_t input_width;
    uint32_t output_width;
} metal_ops_matmul_params;

typedef struct {
    uint32_t rows;
    uint32_t width;
    float epsilon;
} metal_ops_norm_params;

static const void *k_metal_ops_library = &k_metal_ops_library;
static const void *k_metal_ops_last_command = &k_metal_ops_last_command;
static const void *k_metal_ops_gelu_pipeline = &k_metal_ops_gelu_pipeline;
static const void *k_metal_ops_residual_pipeline = &k_metal_ops_residual_pipeline;
static const void *k_metal_ops_norm_pipeline = &k_metal_ops_norm_pipeline;
static const void *k_metal_ops_snake_pipeline = &k_metal_ops_snake_pipeline;
static const void *k_metal_ops_im2col_pipeline = &k_metal_ops_im2col_pipeline;
static const void *k_metal_ops_conv_pipeline = &k_metal_ops_conv_pipeline;

static void metal_ops_error(char *error, size_t capacity, NSString *message) {
    if (error == NULL || capacity == 0) return;
    const char *text = message.UTF8String;
    snprintf(error, capacity, "%s", text == NULL ? "Metal operation failed" : text);
}

static id<MTLBuffer> metal_ops_buffer(MynahMetalState *state,
                                       const void *pointer, size_t bytes) {
    if (pointer == NULL || bytes > NSUIntegerMax) return nil;
    return [state bufferForHostPointer:pointer length:(NSUInteger)bytes];
}

static id<MTLComputePipelineState> metal_ops_pipeline(MynahMetalState *state,
                                                        NSString *name,
                                                        char *error,
                                                        size_t capacity) {
    id<MTLLibrary> library = objc_getAssociatedObject(state, k_metal_ops_library);
    if (library == nil) {
        NSString *source =
        @"#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "struct NormParams { uint rows; uint width; float epsilon; };\n"
        "kernel void mynah_ops_gelu(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { float v=x[i]; float c=0.7978845608f*(v+0.044715f*v*v*v); x[i]=0.5f*v*(1.0f+tanh(c)); }\n"
        "kernel void mynah_ops_residual(device float *out [[buffer(0)]], device const float *in [[buffer(1)]], uint i [[thread_position_in_grid]]) { out[i] += in[i]; }\n"
        "struct SnakeParams { uint channels; uint length; uint snake_channels; };\n"
        "kernel void mynah_ops_snake(device float *x [[buffer(0)]], device const float *alpha [[buffer(1)]], constant SnakeParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) { uint ch=i/p.length; if(ch<p.snake_channels){float a=alpha[ch];float v=x[i];float s=sin(a*v);x[i]=v+s*s/(a+1.0e-9f);} else if(x[i]<0.0f) x[i]*=0.01f; }\n"
        "struct ConvParams { uint in_channels; uint out_channels; uint length; uint kernel_size; uint dilation; };\n"
        "kernel void mynah_ops_im2col(device const float *input [[buffer(0)]], device float *columns [[buffer(1)]], constant ConvParams &p [[buffer(2)]], uint index [[thread_position_in_grid]]) { uint t=index%p.length; uint ik=index/p.length; uint k=ik%p.kernel_size; uint ch=ik/p.kernel_size; int src=int(t)-int((p.kernel_size-1u-k)*p.dilation); columns[index]=(src>=0)?input[ch*p.length+uint(src)]:0.0f; }\n"
        "kernel void mynah_ops_conv(device const float *columns [[buffer(0)]], device const float *weight [[buffer(1)]], device const float *bias [[buffer(2)]], device float *output [[buffer(3)]], constant ConvParams &p [[buffer(4)]], uint index [[thread_position_in_grid]]) { uint t=index%p.length; uint o=index/p.length; float v=bias==nullptr?0.0f:bias[o]; for(uint ch=0;ch<p.in_channels;ch++) for(uint k=0;k<p.kernel_size;k++) v+=weight[o*(p.in_channels*p.kernel_size)+ch*p.kernel_size+k]*columns[(ch*p.kernel_size+k)*p.length+t]; output[index]=v; }\n"
        "kernel void mynah_ops_layer_norm(device const float *in [[buffer(0)]], device float *out [[buffer(1)]], device const float *gain [[buffer(2)]], device const float *bias [[buffer(3)]], constant NormParams &p [[buffer(4)]], uint row [[thread_position_in_grid]]) {\n"
        " if (row >= p.rows) return; device const float *x=in+row*p.width; device float *y=out+row*p.width; float mean=0.0f; for(uint i=0;i<p.width;i++) mean+=x[i]; mean/=float(p.width); float var=0.0f; for(uint i=0;i<p.width;i++){float d=x[i]-mean;var+=d*d;} float inv=rsqrt(var/float(p.width)+p.epsilon); for(uint i=0;i<p.width;i++) y[i]=(x[i]-mean)*inv*gain[i]+(bias==nullptr?0.0f:bias[i]); }\n";
        NSError *compile_error = nil;
        library = [state.device newLibraryWithSource:source options:nil error:&compile_error];
        if (library == nil) {
            metal_ops_error(error, capacity, [NSString stringWithFormat:@"Metal ops shader compilation failed: %@",
                                                            compile_error.localizedDescription]);
            return nil;
        }
        objc_setAssociatedObject(state, k_metal_ops_library, library,
                                 OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    const void *assoc_key = k_metal_ops_norm_pipeline;
    if ([name isEqualToString:@"mynah_ops_gelu"]) assoc_key = k_metal_ops_gelu_pipeline;
    else if ([name isEqualToString:@"mynah_ops_residual"]) assoc_key = k_metal_ops_residual_pipeline;
    else if ([name isEqualToString:@"mynah_ops_snake"]) assoc_key = k_metal_ops_snake_pipeline;
    else if ([name isEqualToString:@"mynah_ops_im2col"]) assoc_key = k_metal_ops_im2col_pipeline;
    else if ([name isEqualToString:@"mynah_ops_conv"]) assoc_key = k_metal_ops_conv_pipeline;
    id<MTLComputePipelineState> pipeline = objc_getAssociatedObject(state, assoc_key);
    if (pipeline == nil) {
        id<MTLFunction> function = [library newFunctionWithName:name];
        NSError *pipeline_error = nil;
        pipeline = [state.device newComputePipelineStateWithFunction:function error:&pipeline_error];
        if (pipeline == nil) {
            metal_ops_error(error, capacity, [NSString stringWithFormat:@"Metal ops pipeline failed: %@",
                                                            pipeline_error.localizedDescription]);
            return nil;
        }
        objc_setAssociatedObject(state, assoc_key, pipeline,
                                 OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    return pipeline;
}

static int metal_ops_commit(MynahMetalState *state, id<MTLCommandBuffer> command,
                            char *error, size_t capacity) {
    [command addCompletedHandler:^(id<MTLCommandBuffer> completed) {
        (void)completed;
    }];
    objc_setAssociatedObject(state, k_metal_ops_last_command, command,
                             OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    [command commit];
    (void)error;
    (void)capacity;
    return 0;
}

static int metal_ops_sync_state(MynahMetalState *state, char *error, size_t capacity) {
    id<MTLCommandBuffer> command = objc_getAssociatedObject(state, k_metal_ops_last_command);
    if (command == nil) return 0;
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) {
        metal_ops_error(error, capacity, command.error.localizedDescription);
        return -1;
    }
    objc_setAssociatedObject(state, k_metal_ops_last_command, nil,
                             OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    return 0;
}

static int metal_ops_snake(MynahMetalState *state, float *data, const float *alpha,
                           size_t channels, size_t length, size_t snake_channels,
                           char *error, size_t capacity) {
    if (channels == 0 || length == 0 || channels > UINT32_MAX ||
        length > UINT32_MAX || snake_channels > UINT32_MAX ||
        channels > SIZE_MAX / length) return -1;
    id<MTLBuffer> dx = metal_ops_buffer(state, data, channels * length * sizeof(float));
    id<MTLBuffer> da = metal_ops_buffer(state, alpha, channels * sizeof(float));
    id<MTLComputePipelineState> pipeline = metal_ops_pipeline(state, @"mynah_ops_snake", error, capacity);
    if (dx == nil || da == nil || pipeline == nil) return -1;
    if (state.params_buffer == nil || state.params_buffer.length < 3u * sizeof(uint32_t))
        state.params_buffer = [state.device newBufferWithLength:3u * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
    uint32_t params[3] = {(uint32_t)channels, (uint32_t)length, (uint32_t)snake_channels};
    memcpy(state.params_buffer.contents, params, sizeof(params));
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dx offset:0 atIndex:0];
    [encoder setBuffer:da offset:0 atIndex:1];
    [encoder setBuffer:state.params_buffer offset:0 atIndex:2];
    NSUInteger threads = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(channels * length, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    return metal_ops_commit(state, command, error, capacity);
}

static int metal_ops_conv1d(MynahMetalState *state, const float *input, float *output,
                            int in_channels, int out_channels, int length, int kernel,
                            int dilation, const float *weight, const float *bias,
                            char *error, size_t capacity) {
    if (in_channels <= 0 || out_channels <= 0 || length <= 0 || kernel <= 0 || dilation <= 0)
        return -1;
    if ((size_t)in_channels > SIZE_MAX / (size_t)kernel) return -1;
    size_t inner = (size_t)in_channels * (size_t)kernel;
    if ((size_t)length > SIZE_MAX / (size_t)in_channels) return -1;
    size_t input_count = (size_t)in_channels * (size_t)length;
    if ((size_t)length > SIZE_MAX / inner ||
        (size_t)out_channels > SIZE_MAX / (size_t)length) return -1;
    size_t column_count = inner * (size_t)length;
    size_t output_count = (size_t)out_channels * (size_t)length;
    if (input_count > SIZE_MAX - column_count ||
        input_count + column_count > SIZE_MAX - output_count ||
        input_count + column_count + output_count > SIZE_MAX / sizeof(float)) return -1;
    if (inner > UINT32_MAX || (size_t)out_channels > UINT32_MAX ||
        (size_t)length > UINT32_MAX || (size_t)kernel > UINT32_MAX ||
        (size_t)dilation > UINT32_MAX) return -1;
    NSUInteger bytes = (input_count + column_count + output_count) * sizeof(float);
    id<MTLBuffer> io = [state ioBufferWithLength:bytes];
    id<MTLBuffer> dw = metal_ops_buffer(state, weight, inner * (size_t)out_channels * sizeof(float));
    id<MTLBuffer> db = bias == NULL ? nil : metal_ops_buffer(state, bias, (size_t)out_channels * sizeof(float));
    if (io == nil || dw == nil || (bias != NULL && db == nil)) return -1;
    float *io_ptr = (float *)io.contents;
    memcpy(io_ptr, input, input_count * sizeof(float));
    if (state.params_buffer == nil || state.params_buffer.length < 5u * sizeof(uint32_t))
        state.params_buffer = [state.device newBufferWithLength:5u * sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
    uint32_t params[5] = {(uint32_t)in_channels, (uint32_t)out_channels,
                          (uint32_t)length, (uint32_t)kernel, (uint32_t)dilation};
    memcpy(state.params_buffer.contents, params, sizeof(params));
    id<MTLComputePipelineState> im2col = metal_ops_pipeline(state, @"mynah_ops_im2col", error, capacity);
    id<MTLComputePipelineState> conv = metal_ops_pipeline(state, @"mynah_ops_conv", error, capacity);
    if (im2col == nil || conv == nil) return -1;
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:im2col];
    [encoder setBuffer:io offset:0 atIndex:0];
    [encoder setBuffer:io offset:input_count * sizeof(float) atIndex:1];
    [encoder setBuffer:state.params_buffer offset:0 atIndex:2];
    NSUInteger threads = MIN((NSUInteger)256, im2col.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(column_count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:conv];
    [encoder setBuffer:io offset:input_count * sizeof(float) atIndex:0];
    [encoder setBuffer:dw offset:0 atIndex:1];
    [encoder setBuffer:db offset:0 atIndex:2];
    [encoder setBuffer:io offset:(input_count + column_count) * sizeof(float) atIndex:3];
    [encoder setBuffer:state.params_buffer offset:0 atIndex:4];
    threads = MIN((NSUInteger)256, conv.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(output_count, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    if (metal_ops_commit(state, command, error, capacity) != 0) return -1;
    if (metal_ops_sync_state(state, error, capacity) != 0) return -1;
    memcpy(output, io_ptr + input_count + column_count, output_count * sizeof(float));
    return 0;
}

static int metal_ops_dispatch_vector(MynahMetalState *state, NSString *name,
                                      float *data, const float *other, size_t n,
                                      char *error, size_t capacity) {
    id<MTLBuffer> out = metal_ops_buffer(state, data, n * sizeof(float));
    id<MTLBuffer> in = other == NULL ? nil : metal_ops_buffer(state, other, n * sizeof(float));
    id<MTLComputePipelineState> pipeline = metal_ops_pipeline(state, name, error, capacity);
    if (out == nil || (other != NULL && in == nil)) {
        metal_ops_error(error, capacity, @"Metal resident buffer lookup failed");
        return -1;
    }
    if (pipeline == nil) {
        return -1;
    }
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:out offset:0 atIndex:0];
    if (in != nil) [encoder setBuffer:in offset:0 atIndex:1];
    NSUInteger threads = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    if (metal_ops_commit(state, command, error, capacity) != 0) return -1;
    /* Host callers pass ordinary malloc'd arrays.  bufferForHostPointer: uses
     * a private shared MTLBuffer for those arrays, so the GPU result must be
     * copied back before returning.  Resident callers pass buffer.contents;
     * in that case the pointers are identical and the copy is unnecessary. */
    if (out.contents != data) {
        if (metal_ops_sync_state(state, error, capacity) != 0) return -1;
        memcpy(data, out.contents, n * sizeof(float));
    }
    return 0;
}

static int metal_ops_layer_norm(MynahMetalState *state, const float *input,
                                 float *output, const float *gain,
                                 const float *bias, size_t rows, size_t width,
                                 char *error, size_t capacity) {
    id<MTLBuffer> in = metal_ops_buffer(state, input, rows * width * sizeof(float));
    id<MTLBuffer> out = metal_ops_buffer(state, output, rows * width * sizeof(float));
    id<MTLBuffer> dg = metal_ops_buffer(state, gain, width * sizeof(float));
    id<MTLBuffer> db = bias == NULL ? nil : metal_ops_buffer(state, bias, width * sizeof(float));
    id<MTLComputePipelineState> pipeline = metal_ops_pipeline(state, @"mynah_ops_layer_norm", error, capacity);
    if (rows > UINT32_MAX || width > UINT32_MAX || in == nil || out == nil || dg == nil || pipeline == nil) {
        metal_ops_error(error, capacity, @"Metal layer norm buffer lookup failed");
        return -1;
    }
    metal_ops_norm_params params = {(uint32_t)rows, (uint32_t)width, 1.0e-5f};
 id<MTLBuffer> params_buffer = [state.device newBufferWithBytes:&params
                                                           length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
 if (params_buffer == nil) {
 metal_ops_error(error, capacity, @"Metal layer norm parameter buffer allocation failed");
 return -1;
 }
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBuffer:dg offset:0 atIndex:2];
    [encoder setBuffer:db offset:0 atIndex:3];
 [encoder setBuffer:params_buffer offset:0 atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(rows, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup), 1, 1)];
    [encoder endEncoding];
    return metal_ops_commit(state, command, error, capacity);
}

static int metal_ops_matmul(MynahMetalState *state, const float *input,
                             float *output, size_t rows, size_t iw, size_t ow,
                             const float *weight, const float *bias,
                             char *error, size_t capacity) {
    if (rows > UINT32_MAX || iw > UINT32_MAX || ow > UINT32_MAX) {
        metal_ops_error(error, capacity, @"Metal matmul dimensions are too large");
        return -1;
    }
    id<MTLBuffer> in = metal_ops_buffer(state, input, rows * iw * sizeof(float));
    id<MTLBuffer> out = metal_ops_buffer(state, output, rows * ow * sizeof(float));
    id<MTLBuffer> dw = metal_ops_buffer(state, weight, iw * ow * sizeof(float));
    id<MTLBuffer> db = bias == NULL ? nil : metal_ops_buffer(state, bias, ow * sizeof(float));
    if (in == nil || out == nil || dw == nil || (bias != NULL && db == nil)) {
        metal_ops_error(error, capacity, @"Metal matmul resident buffer lookup failed");
        return -1;
    }
    metal_ops_matmul_params params = {(uint32_t)rows, (uint32_t)iw, (uint32_t)ow};
 id<MTLBuffer> params_buffer = [state.device newBufferWithBytes:&params
                                                           length:sizeof(params)
                                                          options:MTLResourceStorageModeShared];
 if (params_buffer == nil) {
 metal_ops_error(error, capacity, @"Metal matmul parameter buffer allocation failed");
 return -1;
 }
    id<MTLComputePipelineState> pipeline = ow >= 64u ? state.tiled_pipeline : state.matmul_pipeline;
    if (pipeline == nil) {
        metal_ops_error(error, capacity, @"Metal matmul pipeline unavailable");
        return -1;
    }
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:dw offset:0 atIndex:1];
    [encoder setBuffer:db offset:0 atIndex:2];
    [encoder setBuffer:out offset:0 atIndex:3];
 [encoder setBuffer:params_buffer offset:0 atIndex:4];
    if (pipeline == state.tiled_pipeline) {
        NSUInteger tgx = MIN((NSUInteger)64, pipeline.maxTotalThreadsPerThreadgroup);
        NSUInteger tgy = MIN((NSUInteger)4, pipeline.maxTotalThreadsPerThreadgroup / tgx);
        [encoder dispatchThreadgroups:MTLSizeMake((ow + 63u) / 64u, (rows + 3u) / 4u, 1)
                 threadsPerThreadgroup:MTLSizeMake(tgx, tgy, 1)];
    } else {
        NSUInteger threads = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(rows * ow, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    }
    [encoder endEncoding];
    return metal_ops_commit(state, command, error, capacity);
}

static int metal_ops_alloc(MynahMetalState *state, size_t n, float **out,
                            char *error, size_t capacity) {
    if (out == NULL || n > SIZE_MAX / sizeof(float)) return -1;
    NSUInteger bytes = n * sizeof(float);
    id<MTLBuffer> buffer = [state.device newBufferWithLength:bytes
                                                     options:MTLResourceStorageModeShared];
    if (buffer == nil) {
        metal_ops_error(error, capacity, @"Metal device buffer allocation failed");
        return -1;
    }
    MynahMetalCachedBuffer *cached = [MynahMetalCachedBuffer new];
    cached.host_pointer = buffer.contents;
    cached.length = bytes;
    cached.buffer = buffer;
    [state.weight_cache addObject:cached];
    *out = (float *)buffer.contents;
    return 0;
}

int mynah_metal_upload(void *opaque, const float *host, size_t n, float **out,
                       char *error, size_t capacity) {
    MynahMetalState *state = (__bridge MynahMetalState *)opaque;
    if (n > SIZE_MAX / sizeof(float)) return -1;
    id<MTLBuffer> buffer = [state ioBufferWithLength:n * sizeof(float)];
    if (buffer == nil) {
        metal_ops_error(error, capacity, @"Metal upload buffer allocation failed");
        return -1;
    }
    const void *pointer = buffer.contents;
    BOOL cached = NO;
    for (MynahMetalCachedBuffer *entry in state.weight_cache) {
        if (entry.host_pointer == pointer && entry.length >= n * sizeof(float)) {
            cached = YES;
            break;
        }
    }
    if (!cached) {
        MynahMetalCachedBuffer *entry = [MynahMetalCachedBuffer new];
        entry.host_pointer = pointer;
        entry.length = n * sizeof(float);
        entry.buffer = buffer;
        [state.weight_cache addObject:entry];
    }
    *out = (float *)pointer;
    memcpy(*out, host, n * sizeof(float));
    return 0;
}

int mynah_metal_download(void *opaque, const float *device, float *host, size_t n,
                         char *error, size_t capacity) {
    MynahMetalState *state = (__bridge MynahMetalState *)opaque;
    if (metal_ops_sync_state(state, error, capacity) != 0) return -1;
    id<MTLBuffer> buffer = metal_ops_buffer(state, device, n * sizeof(float));
    if (buffer == nil) return -1;
    memcpy(host, buffer.contents, n * sizeof(float));
    return 0;
}

int mynah_metal_sync(void *opaque, char *error, size_t capacity) {
    return metal_ops_sync_state((__bridge MynahMetalState *)opaque, error, capacity);
}

int mynah_metal_gelu_dev(void *opaque, float *data, size_t n, char *error, size_t capacity) {
    return metal_ops_dispatch_vector((__bridge MynahMetalState *)opaque, @"mynah_ops_gelu",
                                      data, NULL, n, error, capacity);
}

int mynah_metal_snake_dev(void *opaque, float *data, const float *alpha,
                          size_t channels, size_t length, size_t snake_channels,
                          char *error, size_t capacity) {
    return metal_ops_snake((__bridge MynahMetalState *)opaque, data, alpha,
                           channels, length, snake_channels, error, capacity);
}

int mynah_metal_residual_add_dev(void *opaque, float *out, const float *in, size_t n,
                                 char *error, size_t capacity) {
    return metal_ops_dispatch_vector((__bridge MynahMetalState *)opaque, @"mynah_ops_residual",
                                      out, in, n, error, capacity);
}

int mynah_metal_layer_norm_dev(void *opaque, const float *in, float *out,
                               const float *gain, const float *bias, size_t rows,
                               size_t width, char *error, size_t capacity) {
    return metal_ops_layer_norm((__bridge MynahMetalState *)opaque, in, out, gain,
                                bias, rows, width, error, capacity);
}

int mynah_metal_matmul_dev(void *opaque, const float *in, float *out, size_t rows,
                           size_t iw, size_t ow, const float *weight, const float *bias,
                           char *error, size_t capacity) {
    return metal_ops_matmul((__bridge MynahMetalState *)opaque, in, out, rows, iw, ow,
                             weight, bias, error, capacity);
}

int mynah_metal_dev_alloc(void *opaque, size_t n, float **out, char *error, size_t capacity) {
    return metal_ops_alloc((__bridge MynahMetalState *)opaque, n, out, error, capacity);
}

void mynah_metal_dev_free(void *opaque, float *pointer) {
    MynahMetalState *state = (__bridge MynahMetalState *)opaque;
    for (MynahMetalCachedBuffer *cached in [state.weight_cache copy]) {
        if (cached.host_pointer == pointer) {
            [state.weight_cache removeObject:cached];
            break;
        }
    }
}

int mynah_metal_h2d(void *opaque, const float *host, float *device, size_t n,
                    char *error, size_t capacity) {
    (void)opaque;
    (void)error;
    (void)capacity;
    memcpy(device, host, n * sizeof(float));
    return 0;
}

int mynah_metal_d2h(void *opaque, const float *device, float *host, size_t n,
                    char *error, size_t capacity) {
    return mynah_metal_download(opaque, device, host, n, error, capacity);
}

int mynah_metal_gelu_inplace(void *opaque, float *data, size_t n, char *error, size_t capacity) {
    return mynah_metal_gelu_dev(opaque, data, n, error, capacity);
}

int mynah_metal_residual_inplace(void *opaque, float *out, const float *in, size_t n,
                                 char *error, size_t capacity) {
    return mynah_metal_residual_add_dev(opaque, out, in, n, error, capacity);
}

int mynah_metal_layer_norm_inplace(void *opaque, const float *in, float *out,
                                   const float *gain, size_t rows, size_t width,
                                   char *error, size_t capacity) {
    return mynah_metal_layer_norm_dev(opaque, in, out, gain, NULL, rows, width,
                                      error, capacity);
}

int mynah_metal_matmul_d2d(void *opaque, const float *in, float *out, size_t rows,
                           size_t iw, size_t ow, const float *weight, const float *bias,
                           char *error, size_t capacity) {
    return mynah_metal_matmul_dev(opaque, in, out, rows, iw, ow, weight, bias,
                                  error, capacity);
}

int mynah_metal_conv1d(void *opaque, const float *input, float *output,
                       int in_channels, int out_channels, int length, int kernel,
                       int dilation, const float *weight, const float *bias,
                       char *error, size_t capacity) {
    return metal_ops_conv1d((__bridge MynahMetalState *)opaque, input, output,
                            in_channels, out_channels, length, kernel, dilation,
                            weight, bias, error, capacity);
}

int mynah_metal_ops_self_test(void *opaque, char *error, size_t capacity) {
    float *din = NULL;
    float *dout = NULL;
    float *dtmp = NULL;
    float input[] = {1.0f, 2.0f};
    float weight[] = {1.0f, 0.0f, 0.0f, 1.0f};
    float bias[] = {0.5f, -0.5f};
    float zeros[] = {0.0f, 0.0f};
    float output[2] = {0.0f, 0.0f};
    float conv_input[] = {1.0f, 2.0f, 3.0f};
    float conv_weight[] = {1.0f, 2.0f};
    float conv_output[3] = {0.0f, 0.0f, 0.0f};
    float norm_gain[] = {1.0f, 1.0f};
    float alpha[] = {1.0f, 1.0f};
    MynahMetalState *state = (__bridge MynahMetalState *)opaque;
    if (metal_ops_alloc(state, 2u, &din, error, capacity) != 0 ||
        metal_ops_alloc(state, 2u, &dout, error, capacity) != 0 ||
        metal_ops_alloc(state, 2u, &dtmp, error, capacity) != 0) goto fail;
    if (mynah_metal_h2d(opaque, input, din, 2u, error, capacity) != 0 ||
        mynah_metal_h2d(opaque, zeros, dtmp, 2u, error, capacity) != 0 ||
        mynah_metal_matmul_d2d(opaque, din, dout, 1u, 2u, 2u, weight, bias,
                               error, capacity) != 0 ||
        mynah_metal_sync(opaque, error, capacity) != 0 ||
        mynah_metal_d2h(opaque, dout, output, 2u, error, capacity) != 0 ||
        fabsf(output[0] - 1.5f) > 1e-5f || fabsf(output[1] - 1.5f) > 1e-5f) goto fail;
    if (mynah_metal_gelu_inplace(opaque, dout, 2u, error, capacity) != 0 ||
        mynah_metal_residual_inplace(opaque, dtmp, dout, 2u, error, capacity) != 0 ||
        mynah_metal_sync(opaque, error, capacity) != 0) goto fail;
    if (mynah_metal_layer_norm_inplace(opaque, dout, dtmp, norm_gain, 1u, 2u,
                                       error, capacity) != 0 ||
        mynah_metal_snake_dev(opaque, dtmp, alpha, 2u, 1u, 2u, error, capacity) != 0 ||
        mynah_metal_sync(opaque, error, capacity) != 0) goto fail;
    if (mynah_metal_conv1d(opaque, conv_input, conv_output, 1, 1, 3, 2, 1,
                           conv_weight, NULL, error, capacity) != 0 ||
        fabsf(conv_output[0] - 2.0f) > 1e-5f ||
        fabsf(conv_output[1] - 5.0f) > 1e-5f ||
        fabsf(conv_output[2] - 8.0f) > 1e-5f) goto fail;
    mynah_metal_dev_free(opaque, din);
    mynah_metal_dev_free(opaque, dout);
    mynah_metal_dev_free(opaque, dtmp);
    return 0;
fail:
    mynah_metal_dev_free(opaque, din);
    mynah_metal_dev_free(opaque, dout);
    mynah_metal_dev_free(opaque, dtmp);
    if (error != NULL && error[0] == '\0')
        snprintf(error, capacity, "%s", "Metal resident ops self-test mismatch");
    return -1;
}
