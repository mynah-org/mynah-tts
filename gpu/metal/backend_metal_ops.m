#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
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
@property(nonatomic, strong) NSMutableArray *params_buffers;
@property(nonatomic, assign) NSUInteger params_cursor;
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLComputePipelineState> matmul_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> tiled_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> row_tiled_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> row_simd_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> self_attention_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> cross_attention_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> copy_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> argmax_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> conv_transpose_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> scale_pipeline;
@property(nonatomic, strong) id<MTLComputePipelineState> clip_pipeline;
@property(nonatomic, assign) BOOL use_mps;
@property(nonatomic, assign) BOOL use_simd_matvec;
@property(nonatomic, strong) NSMutableArray *weight_cache;
@property(nonatomic, strong) id<MTLBuffer> params_buffer;
@property(nonatomic, strong) id<MTLBuffer> scalar_buffer;
- (id<MTLBuffer>)ioBufferWithLength:(NSUInteger)length;
- (id<MTLBuffer>)convColumnsBufferWithLength:(NSUInteger)length;
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

typedef struct {
    uint32_t heads;
    uint32_t head_width;
    uint32_t valid;
    uint32_t cache_stride;
    uint32_t position;
    float scale;
} metal_ops_attention_params;

typedef struct {
    uint32_t vocab;
    uint32_t codebook_size;
    uint32_t eos_id;
    uint32_t allow_eos;
} metal_ops_argmax_params;

static const void *k_metal_ops_library = &k_metal_ops_library;
static const void *k_metal_ops_last_command = &k_metal_ops_last_command;
static const void *k_metal_ops_active_command = &k_metal_ops_active_command;
static const void *k_metal_ops_gelu_pipeline = &k_metal_ops_gelu_pipeline;
static const void *k_metal_ops_residual_pipeline = &k_metal_ops_residual_pipeline;
static const void *k_metal_ops_norm_pipeline = &k_metal_ops_norm_pipeline;
static const void *k_metal_ops_snake_pipeline = &k_metal_ops_snake_pipeline;
static const void *k_metal_ops_im2col_pipeline = &k_metal_ops_im2col_pipeline;
static const void *k_metal_ops_conv_pipeline = &k_metal_ops_conv_pipeline;
static const void *k_metal_ops_conv_bias_pipeline = &k_metal_ops_conv_bias_pipeline;
static const void *k_metal_ops_conv_mps_cache = &k_metal_ops_conv_mps_cache;
static const void *k_metal_ops_self_attention_pipeline = &k_metal_ops_self_attention_pipeline;
static const void *k_metal_ops_cross_attention_pipeline = &k_metal_ops_cross_attention_pipeline;
static const void *k_metal_ops_copy_pipeline = &k_metal_ops_copy_pipeline;
static const void *k_metal_ops_argmax_pipeline = &k_metal_ops_argmax_pipeline;
static const void *k_metal_ops_conv_transpose_pipeline = &k_metal_ops_conv_transpose_pipeline;
static const void *k_metal_ops_scale_pipeline = &k_metal_ops_scale_pipeline;
static const void *k_metal_ops_clip_pipeline = &k_metal_ops_clip_pipeline;

@interface MynahMetalConvMPS : NSObject
@property(nonatomic, assign) NSUInteger inner;
@property(nonatomic, assign) NSUInteger out_channels;
@property(nonatomic, assign) NSUInteger length;
@property(nonatomic, strong) MPSMatrixMultiplication *kernel;
@property(nonatomic, strong) MPSMatrixDescriptor *weight_descriptor;
@property(nonatomic, strong) MPSMatrixDescriptor *columns_descriptor;
@property(nonatomic, strong) MPSMatrixDescriptor *output_descriptor;
@end

@implementation MynahMetalConvMPS
@end

static MynahMetalConvMPS *metal_ops_conv_mps(MynahMetalState *state,
                                              NSUInteger inner,
                                              NSUInteger out_channels,
                                              NSUInteger length) {
    NSMutableArray *cache = objc_getAssociatedObject(state, k_metal_ops_conv_mps_cache);
    if (cache == nil) {
        cache = [NSMutableArray array];
        objc_setAssociatedObject(state, k_metal_ops_conv_mps_cache, cache,
                                 OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    }
    for (MynahMetalConvMPS *entry in cache) {
        if (entry.inner == inner && entry.out_channels == out_channels &&
            entry.length == length) return entry;
    }
    MPSMatrixDescriptor *weight_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:out_channels columns:inner
                                              rowBytes:inner * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *columns_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:inner columns:length
                                              rowBytes:length * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *output_descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:out_channels columns:length
                                              rowBytes:length * sizeof(float)
                                              dataType:MPSDataTypeFloat32];
    MPSMatrixMultiplication *kernel =
        [[MPSMatrixMultiplication alloc] initWithDevice:state.device
                                         transposeLeft:NO transposeRight:NO
                                            resultRows:out_channels
                                         resultColumns:length
                                       interiorColumns:inner
                                                  alpha:1.0 beta:0.0];
    if (weight_descriptor == nil || columns_descriptor == nil ||
        output_descriptor == nil || kernel == nil) return nil;
    MynahMetalConvMPS *entry = [MynahMetalConvMPS new];
    entry.inner = inner;
    entry.out_channels = out_channels;
    entry.length = length;
    entry.kernel = kernel;
    entry.weight_descriptor = weight_descriptor;
    entry.columns_descriptor = columns_descriptor;
    entry.output_descriptor = output_descriptor;
    [cache addObject:entry];
    return entry;
}

static void metal_ops_error(char *error, size_t capacity, NSString *message) {
    if (error == NULL || capacity == 0) return;
    const char *text = message.UTF8String;
    snprintf(error, capacity, "%s", text == NULL ? "Metal operation failed" : text);
}

static id<MTLBuffer> metal_ops_params_buffer(MynahMetalState *state,
                                               const void *value, NSUInteger bytes,
                                               char *error, size_t capacity) {
    if (objc_getAssociatedObject(state, k_metal_ops_active_command) == nil)
        state.params_cursor = 0;
    if (state.params_buffers == nil)
        state.params_buffers = [NSMutableArray array];
    const NSUInteger index = state.params_cursor++;
    id<MTLBuffer> buffer = index < state.params_buffers.count
        ? state.params_buffers[index] : nil;
    if (buffer == nil || buffer.length < bytes) {
        buffer = [state.device newBufferWithLength:bytes
                                             options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            metal_ops_error(error, capacity, @"Metal parameter buffer allocation failed");
            return nil;
        }
        if (index < state.params_buffers.count)
            state.params_buffers[index] = buffer;
        else
            [state.params_buffers addObject:buffer];
    }
    memcpy(buffer.contents, value, bytes);
    return buffer;
}

static id<MTLBuffer> metal_ops_buffer(MynahMetalState *state,
                                       const void *pointer, size_t bytes) {
    if (pointer == NULL || bytes > NSUIntegerMax) return nil;
    return [state bufferForHostPointer:pointer length:(NSUInteger)bytes];
}

static id<MTLBuffer> metal_ops_buffer_range(MynahMetalState *state,
                                             const void *pointer, size_t bytes,
                                             NSUInteger *offset_out) {
    if (pointer == NULL || bytes > NSUIntegerMax || offset_out == NULL) return nil;
    const uintptr_t address = (uintptr_t)pointer;
    for (MynahMetalCachedBuffer *cached in state.weight_cache) {
        const uintptr_t base = (uintptr_t)cached.host_pointer;
        if (address < base) continue;
        const uintptr_t offset = address - base;
        if (offset <= cached.length && bytes <= cached.length - offset) {
            *offset_out = (NSUInteger)offset;
            return cached.buffer;
        }
    }
    *offset_out = 0;
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
        "struct AttentionParams { uint heads; uint head_width; uint valid; uint cache_stride; uint position; float scale; };\n"
        "struct ArgmaxParams { uint vocab; uint codebook_size; uint eos_id; uint allow_eos; };\n"
        "struct ScaleParams { float scale; };\n"
        "kernel void mynah_ops_gelu(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) { float v=x[i]; float c=0.7978845608f*(v+0.044715f*v*v*v); x[i]=0.5f*v*(1.0f+tanh(c)); }\n"
        "kernel void mynah_ops_residual(device float *out [[buffer(0)]], device const float *in [[buffer(1)]], uint i [[thread_position_in_grid]]) { out[i] += in[i]; }\n"
        "kernel void mynah_ops_copy(device const float *in [[buffer(0)]], device float *out [[buffer(1)]], uint i [[thread_position_in_grid]]) { out[i] = in[i]; }\n"
        "kernel void mynah_ops_scale(device float *data [[buffer(0)]], constant ScaleParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) { data[i] *= p.scale; }\n"
        "kernel void mynah_ops_clip(device float *data [[buffer(0)]], uint i [[thread_position_in_grid]]) { float v=data[i]; data[i]=isfinite(v) ? clamp(v, -1.0f, 1.0f) : 0.0f; }\n"
        "kernel void mynah_ops_argmax(device const float *logits [[buffer(0)]], device uint *result [[buffer(1)]], constant ArgmaxParams &p [[buffer(2)]], uint tid [[thread_position_in_grid]]) { if(tid!=0u) return; float best=-3.402823466e+38f; uint index=0u; for(uint i=0u;i<p.vocab;i++){bool allowed=i<p.codebook_size || (p.allow_eos!=0u && i==p.eos_id); if(allowed && logits[i]>best){best=logits[i];index=i;}} result[0]=index; }\n"
        "struct SnakeParams { uint channels; uint length; uint snake_channels; };\n"
        "kernel void mynah_ops_snake(device float *x [[buffer(0)]], device const float *alpha [[buffer(1)]], constant SnakeParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) { uint ch=i/p.length; if(ch<p.snake_channels){float a=alpha[ch];float v=x[i];float s=sin(a*v);x[i]=v+s*s/(a+1.0e-9f);} else if(x[i]<0.0f) x[i]*=0.01f; }\n"
        "struct ConvParams { uint in_channels; uint out_channels; uint length; uint kernel_size; uint dilation; };\n"
        "kernel void mynah_ops_im2col(device const float *input [[buffer(0)]], device float *columns [[buffer(1)]], constant ConvParams &p [[buffer(2)]], uint index [[thread_position_in_grid]]) { uint t=index%p.length; uint ik=index/p.length; uint k=ik%p.kernel_size; uint ch=ik/p.kernel_size; int src=int(t)-int((p.kernel_size-1u-k)*p.dilation); columns[index]=(src>=0)?input[ch*p.length+uint(src)]:0.0f; }\n"
        "kernel void mynah_ops_conv(device const float *columns [[buffer(0)]], device const float *weight [[buffer(1)]], device const float *bias [[buffer(2)]], device float *output [[buffer(3)]], constant ConvParams &p [[buffer(4)]], uint index [[thread_position_in_grid]]) { uint t=index%p.length; uint o=index/p.length; float v=bias==nullptr?0.0f:bias[o]; for(uint ch=0;ch<p.in_channels;ch++) for(uint k=0;k<p.kernel_size;k++) v+=weight[o*(p.in_channels*p.kernel_size)+ch*p.kernel_size+k]*columns[(ch*p.kernel_size+k)*p.length+t]; output[index]=v; }\n"
        "kernel void mynah_ops_conv_bias(device float *output [[buffer(0)]], device const float *bias [[buffer(1)]], constant ConvParams &p [[buffer(2)]], uint index [[thread_position_in_grid]]) { output[index] += bias[index/p.length]; }\n"
        "struct ConvTransposeParams { uint in_channels; uint out_channels; uint length; uint output_length; uint kernel_size; uint stride; uint groups; };\n"
        "kernel void mynah_ops_conv_transpose(device const float *input [[buffer(0)]], device const float *weight [[buffer(1)]], device const float *bias [[buffer(2)]], device float *output [[buffer(3)]], constant ConvTransposeParams &p [[buffer(4)]], uint index [[thread_position_in_grid]]) { uint t=index%p.output_length; uint o=index/p.output_length; uint in_per_group=p.in_channels/p.groups; uint out_per_group=p.out_channels/p.groups; uint group=o/out_per_group; uint o_local=o%out_per_group; float v=bias==nullptr?0.0f:bias[o]; for(uint k=0u;k<p.kernel_size;k++){ if(t<k || ((t-k)%p.stride)!=0u) continue; uint it=(t-k)/p.stride; if(it>=p.length) continue; for(uint il=0u;il<in_per_group;il++){uint i=group*in_per_group+il; v+=input[i*p.length+it]*weight[(i*out_per_group+o_local)*p.kernel_size+k];}} output[index]=v; }\n"
        "kernel void mynah_ops_layer_norm(device const float *in [[buffer(0)]], device float *out [[buffer(1)]], device const float *gain [[buffer(2)]], device const float *bias [[buffer(3)]], constant NormParams &p [[buffer(4)]], uint row [[thread_position_in_grid]]) {\n"
        " if (row >= p.rows) return; device const float *x=in+row*p.width; device float *y=out+row*p.width; float mean=0.0f; for(uint i=0;i<p.width;i++) mean+=x[i]; mean/=float(p.width); float var=0.0f; for(uint i=0;i<p.width;i++){float d=x[i]-mean;var+=d*d;} float inv=1.0f/sqrt(var/float(p.width)+p.epsilon); for(uint i=0;i<p.width;i++) y[i]=(x[i]-mean)*inv*gain[i]+(bias==nullptr?0.0f:bias[i]); }\n"
        "kernel void mynah_ops_self_attention(device const float *qkv [[buffer(0)]], device float *kcache [[buffer(1)]], device float *vcache [[buffer(2)]], device float *out [[buffer(3)]], constant AttentionParams &p [[buffer(4)]], uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tc [[threads_per_threadgroup]]) {\n"
        " if(head>=p.heads) return; uint hw=p.head_width; bool active=tid<hw; uint width=p.heads*hw; ulong hbase=(ulong)head*hw; device const float *q=qkv+hbase; device const float *k=qkv+width+hbase; device const float *v=qkv+2u*width+hbase; ulong cache_base=(ulong)p.position*p.cache_stride+hbase; if(active){kcache[cache_base+tid]=k[tid]; vcache[cache_base+tid]=v[tid];} threadgroup_barrier(mem_flags::mem_device); threadgroup float score_shared; float maximum=-1.0e30f, denom=0.0f, acc=0.0f; for(uint s=0;s<p.valid;s++){ulong kvbase=(ulong)s*p.cache_stride+hbase; device const float *ks=(s==p.position)?k:(kcache+kvbase); device const float *vs=(s==p.position)?v:(vcache+kvbase); if(tid==0u){float dot=0.0f; for(uint d=0u;d<hw;d++) dot+=q[d]*ks[d]; score_shared=dot*p.scale;} threadgroup_barrier(mem_flags::mem_threadgroup); float score=score_shared; float next=max(maximum,score); float corr=exp(maximum-next); float prob=exp(score-next); if(active){denom=denom*corr+prob;acc=acc*corr+prob*vs[tid];} maximum=next; threadgroup_barrier(mem_flags::mem_threadgroup);} if(active)out[hbase+tid]=acc/denom; }\n"
        "kernel void mynah_ops_cross_attention(device const float *q [[buffer(0)]], device const float *kcache [[buffer(1)]], device const float *vcache [[buffer(2)]], device float *out [[buffer(3)]], constant AttentionParams &p [[buffer(4)]], uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]], uint tc [[threads_per_threadgroup]]) {\n"
        " if(head>=p.heads) return; uint hw=p.head_width; bool active=tid<hw; ulong hbase=(ulong)head*hw; float qv=active?q[hbase+tid]:0.0f; threadgroup float partial[256]; float maximum=-1.0e30f, denom=0.0f, acc=0.0f; for(uint s=0;s<p.valid;s++){ulong kvbase=(ulong)s*p.cache_stride+hbase; partial[tid]=active?qv*kcache[kvbase+tid]:0.0f; threadgroup_barrier(mem_flags::mem_threadgroup); for(uint st=tc/2u;st>0;st>>=1u){if(tid<st)partial[tid]+=partial[tid+st];threadgroup_barrier(mem_flags::mem_threadgroup);} float score=partial[0]*p.scale; float next=max(maximum,score); float corr=exp(maximum-next); float prob=exp(score-next); if(active){denom=denom*corr+prob;acc=acc*corr+prob*vcache[kvbase+tid];} maximum=next; threadgroup_barrier(mem_flags::mem_threadgroup);} if(active)out[hbase+tid]=acc/denom; }\n";
        source = [source stringByAppendingString:
            @"kernel void mynah_ops_cross_attention_exact(device const float *q [[buffer(0)]], device const float *kcache [[buffer(1)]], device const float *vcache [[buffer(2)]], device float *out [[buffer(3)]], constant AttentionParams &p [[buffer(4)]], uint head [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]]) {\n"
             " if(head>=p.heads) return; uint hw=p.head_width; bool active=tid<hw; ulong hbase=(ulong)head*hw; threadgroup float score_shared; float maximum=-1.0e30f, denom=0.0f, acc=0.0f; for(uint s=0u;s<p.valid;s++){ulong kvbase=(ulong)s*p.cache_stride+hbase; if(tid==0u){float dot=0.0f; for(uint d=0u;d<hw;d++) dot+=q[hbase+d]*kcache[kvbase+d]; score_shared=dot*p.scale;} threadgroup_barrier(mem_flags::mem_threadgroup); float score=score_shared; float next=max(maximum,score); float corr=exp(maximum-next); float prob=exp(score-next); if(active){denom=denom*corr+prob; acc=acc*corr+prob*vcache[kvbase+tid];} maximum=next; threadgroup_barrier(mem_flags::mem_threadgroup);} if(active) out[hbase+tid]=acc/denom; }\n"];
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
    else if ([name isEqualToString:@"mynah_ops_conv_bias"]) assoc_key = k_metal_ops_conv_bias_pipeline;
    else if ([name isEqualToString:@"mynah_ops_self_attention"]) assoc_key = k_metal_ops_self_attention_pipeline;
    else if ([name isEqualToString:@"mynah_ops_cross_attention"] ||
             [name isEqualToString:@"mynah_ops_cross_attention_exact"])
        assoc_key = k_metal_ops_cross_attention_pipeline;
    else if ([name isEqualToString:@"mynah_ops_copy"]) assoc_key = k_metal_ops_copy_pipeline;
    else if ([name isEqualToString:@"mynah_ops_argmax"]) assoc_key = k_metal_ops_argmax_pipeline;
    else if ([name isEqualToString:@"mynah_ops_conv_transpose"]) assoc_key = k_metal_ops_conv_transpose_pipeline;
    else if ([name isEqualToString:@"mynah_ops_scale"]) assoc_key = k_metal_ops_scale_pipeline;
    else if ([name isEqualToString:@"mynah_ops_clip"]) assoc_key = k_metal_ops_clip_pipeline;
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
    id<MTLCommandBuffer> active = objc_getAssociatedObject(state, k_metal_ops_active_command);
    if (active == command) return 0;
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

static id<MTLCommandBuffer> metal_ops_command(MynahMetalState *state) {
    id<MTLCommandBuffer> active = objc_getAssociatedObject(state, k_metal_ops_active_command);
    return active == nil ? [state.queue commandBuffer] : active;
}

static int metal_ops_batch_begin(MynahMetalState *state, char *error, size_t capacity) {
    if (objc_getAssociatedObject(state, k_metal_ops_active_command) != nil) return 0;
    state.params_cursor = 0;
    id<MTLCommandBuffer> command = [state.queue commandBuffer];
    if (command == nil) {
        metal_ops_error(error, capacity, @"Metal command batch allocation failed");
        return -1;
    }
    objc_setAssociatedObject(state, k_metal_ops_active_command, command,
                             OBJC_ASSOCIATION_RETAIN_NONATOMIC);
    return 0;
}

static int metal_ops_sync_state(MynahMetalState *state, char *error, size_t capacity) {
    id<MTLCommandBuffer> active = objc_getAssociatedObject(state, k_metal_ops_active_command);
    if (active != nil) {
        objc_setAssociatedObject(state, k_metal_ops_active_command, nil,
                                 OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        if (metal_ops_commit(state, active, error, capacity) != 0) return -1;
    }
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
    id<MTLBuffer> da = metal_ops_buffer(state, alpha, snake_channels * sizeof(float));
    id<MTLComputePipelineState> pipeline = metal_ops_pipeline(state, @"mynah_ops_snake", error, capacity);
    if (dx == nil || da == nil || pipeline == nil) return -1;
    uint32_t params[3] = {(uint32_t)channels, (uint32_t)length, (uint32_t)snake_channels};
    id<MTLBuffer> params_buffer = metal_ops_params_buffer(state, params, sizeof(params),
                                                           error, capacity);
    if (params_buffer == nil) return -1;
    id<MTLCommandBuffer> command = metal_ops_command(state);
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dx offset:0 atIndex:0];
    [encoder setBuffer:da offset:0 atIndex:1];
    [encoder setBuffer:params_buffer offset:0 atIndex:2];
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
    id<MTLBuffer> resident_input = metal_ops_buffer(state, input, input_count * sizeof(float));
    id<MTLBuffer> resident_output = metal_ops_buffer(state, output, output_count * sizeof(float));
    const BOOL resident = resident_input != nil && resident_output != nil &&
                          resident_input.contents == input && resident_output.contents == output;
    if (resident) {
        id<MTLBuffer> columns = [state convColumnsBufferWithLength:column_count * sizeof(float)];
        id<MTLBuffer> dw = metal_ops_buffer(state, weight,
                                            inner * (size_t)out_channels * sizeof(float));
        id<MTLBuffer> db = bias == NULL ? nil : metal_ops_buffer(
            state, bias, (size_t)out_channels * sizeof(float));
        if (columns == nil || dw == nil || (bias != NULL && db == nil)) return -1;
        uint32_t params_value[5] = {(uint32_t)in_channels, (uint32_t)out_channels,
                                    (uint32_t)length, (uint32_t)kernel, (uint32_t)dilation};
        id<MTLBuffer> params_buffer = metal_ops_params_buffer(
            state, params_value, sizeof(params_value), error, capacity);
        id<MTLComputePipelineState> im2col = metal_ops_pipeline(
            state, @"mynah_ops_im2col", error, capacity);
        id<MTLComputePipelineState> conv = metal_ops_pipeline(
            state, @"mynah_ops_conv", error, capacity);
        if (params_buffer == nil || im2col == nil || conv == nil) return -1;
        id<MTLCommandBuffer> command = metal_ops_command(state);
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:im2col];
        [encoder setBuffer:resident_input offset:0 atIndex:0];
        [encoder setBuffer:columns offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        NSUInteger threads = MIN((NSUInteger)256, im2col.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(column_count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        MynahMetalConvMPS *mps = state.use_mps
            ? metal_ops_conv_mps(state, inner, (NSUInteger)out_channels,
                                 (NSUInteger)length)
            : nil;
        if (mps != nil) {
            MPSMatrix *left = [[MPSMatrix alloc] initWithBuffer:dw
                                                    descriptor:mps.weight_descriptor];
            MPSMatrix *right = [[MPSMatrix alloc] initWithBuffer:columns
                                                     descriptor:mps.columns_descriptor];
            MPSMatrix *destination = [[MPSMatrix alloc] initWithBuffer:resident_output
                                                           descriptor:mps.output_descriptor];
            if (left == nil || right == nil || destination == nil) return -1;
            [mps.kernel encodeToCommandBuffer:command leftMatrix:left
                                  rightMatrix:right resultMatrix:destination];
            if (db != nil) {
                id<MTLComputePipelineState> add_bias = metal_ops_pipeline(
                    state, @"mynah_ops_conv_bias", error, capacity);
                if (add_bias == nil) return -1;
                encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:add_bias];
                [encoder setBuffer:resident_output offset:0 atIndex:0];
                [encoder setBuffer:db offset:0 atIndex:1];
                [encoder setBuffer:params_buffer offset:0 atIndex:2];
                threads = MIN((NSUInteger)256,
                              add_bias.maxTotalThreadsPerThreadgroup);
                [encoder dispatchThreads:MTLSizeMake(output_count, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
                [encoder endEncoding];
            }
            return metal_ops_commit(state, command, error, capacity);
        }
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:conv];
        [encoder setBuffer:columns offset:0 atIndex:0];
        [encoder setBuffer:dw offset:0 atIndex:1];
        [encoder setBuffer:db offset:0 atIndex:2];
        [encoder setBuffer:resident_output offset:0 atIndex:3];
        [encoder setBuffer:params_buffer offset:0 atIndex:4];
        threads = MIN((NSUInteger)256, conv.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(output_count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        return metal_ops_commit(state, command, error, capacity);
    }
    id<MTLBuffer> io = [state ioBufferWithLength:bytes];
    id<MTLBuffer> dw = metal_ops_buffer(state, weight, inner * (size_t)out_channels * sizeof(float));
    id<MTLBuffer> db = bias == NULL ? nil : metal_ops_buffer(state, bias, (size_t)out_channels * sizeof(float));
    if (io == nil || dw == nil || (bias != NULL && db == nil)) return -1;
    float *io_ptr = (float *)io.contents;
    memcpy(io_ptr, input, input_count * sizeof(float));
    uint32_t params[5] = {(uint32_t)in_channels, (uint32_t)out_channels,
                          (uint32_t)length, (uint32_t)kernel, (uint32_t)dilation};
    id<MTLBuffer> params_buffer = metal_ops_params_buffer(state, params, sizeof(params),
                                                           error, capacity);
    if (params_buffer == nil) return -1;
    id<MTLComputePipelineState> im2col = metal_ops_pipeline(state, @"mynah_ops_im2col", error, capacity);
    id<MTLComputePipelineState> conv = metal_ops_pipeline(state, @"mynah_ops_conv", error, capacity);
    if (im2col == nil || conv == nil) return -1;
    id<MTLCommandBuffer> command = metal_ops_command(state);
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:im2col];
    [encoder setBuffer:io offset:0 atIndex:0];
    [encoder setBuffer:io offset:input_count * sizeof(float) atIndex:1];
    [encoder setBuffer:params_buffer offset:0 atIndex:2];
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
    [encoder setBuffer:params_buffer offset:0 atIndex:4];
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
    id<MTLCommandBuffer> command = metal_ops_command(state);
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
    id<MTLBuffer> params_buffer = metal_ops_params_buffer(state, &params,
                                                           sizeof(params), error, capacity);
    if (params_buffer == nil) {
        metal_ops_error(error, capacity, @"Metal layer norm parameter buffer allocation failed");
        return -1;
    }
    id<MTLCommandBuffer> command = metal_ops_command(state);
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

static int metal_ops_attention(MynahMetalState *state, int self,
                               const float *query, float *kcache, float *vcache,
                               size_t position, size_t cache_stride, size_t valid,
                               size_t heads, size_t head_width, float scale,
                               float *output, char *error, size_t capacity) {
    if (heads == 0 || head_width == 0 || head_width > 256u || valid == 0 ||
        heads > UINT32_MAX || head_width > UINT32_MAX || valid > UINT32_MAX ||
        cache_stride > UINT32_MAX || position > UINT32_MAX ||
        heads > SIZE_MAX / head_width) return -1;
    const size_t width = heads * head_width;
    if (self && width > SIZE_MAX / 3u) return -1;
    const size_t query_count = self ? width * 3u : width;
    if (cache_stride > SIZE_MAX / valid) return -1;
    NSUInteger q_offset = 0, k_offset = 0, v_offset = 0, out_offset = 0;
    id<MTLBuffer> dq = metal_ops_buffer_range(state, query,
                                               query_count * sizeof(float), &q_offset);
    id<MTLBuffer> dk = metal_ops_buffer_range(state, kcache,
                                               valid * cache_stride * sizeof(float), &k_offset);
    id<MTLBuffer> dv = metal_ops_buffer_range(state, vcache,
                                               valid * cache_stride * sizeof(float), &v_offset);
    id<MTLBuffer> dout = metal_ops_buffer_range(state, output,
                                                width * sizeof(float), &out_offset);
    NSString *name = self ? @"mynah_ops_self_attention" : @"mynah_ops_cross_attention_exact";
    id<MTLComputePipelineState> pipeline = metal_ops_pipeline(state, name, error, capacity);
    if (dq == nil || dk == nil || dv == nil || dout == nil || pipeline == nil) {
        metal_ops_error(error, capacity, @"Metal attention buffer lookup failed");
        return -1;
    }
    metal_ops_attention_params params = {(uint32_t)heads, (uint32_t)head_width,
                                         (uint32_t)valid, (uint32_t)cache_stride,
                                         (uint32_t)position, scale};
    id<MTLBuffer> params_buffer = metal_ops_params_buffer(state, &params,
                                                           sizeof(params), error, capacity);
    if (params_buffer == nil) return -1;
    NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threads = 1;
    while (threads < head_width && threads < max_threads) threads <<= 1u;
    if (threads < head_width || threads > 256u) {
        metal_ops_error(error, capacity, @"Metal attention threadgroup is too small");
        return -1;
    }
    id<MTLCommandBuffer> command = metal_ops_command(state);
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:dq offset:q_offset atIndex:0];
    [encoder setBuffer:dk offset:k_offset atIndex:1];
    [encoder setBuffer:dv offset:v_offset atIndex:2];
    [encoder setBuffer:dout offset:out_offset atIndex:3];
    [encoder setBuffer:params_buffer offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(heads, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
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
    id<MTLBuffer> params_buffer = metal_ops_params_buffer(state, &params,
                                                           sizeof(params), error, capacity);
    if (params_buffer == nil) {
        metal_ops_error(error, capacity, @"Metal matmul parameter buffer allocation failed");
        return -1;
    }
    id<MTLComputePipelineState> pipeline = ow >= 64u ? state.tiled_pipeline : state.matmul_pipeline;
    if (rows == 1u && state.use_simd_matvec && state.row_simd_pipeline != nil &&
        state.row_simd_pipeline.maxTotalThreadsPerThreadgroup >= 32u) {
        pipeline = state.row_simd_pipeline;
    } else if (rows == 1u && ow >= 64u && state.row_tiled_pipeline != nil &&
        state.row_tiled_pipeline.maxTotalThreadsPerThreadgroup >= 64u)
        pipeline = state.row_tiled_pipeline;
    if (pipeline == nil) {
        metal_ops_error(error, capacity, @"Metal matmul pipeline unavailable");
        return -1;
    }
    id<MTLCommandBuffer> command = metal_ops_command(state);
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:in offset:0 atIndex:0];
    [encoder setBuffer:dw offset:0 atIndex:1];
    [encoder setBuffer:db offset:0 atIndex:2];
    [encoder setBuffer:out offset:0 atIndex:3];
 [encoder setBuffer:params_buffer offset:0 atIndex:4];
    if (pipeline == state.row_simd_pipeline) {
        NSUInteger nsg = MIN((NSUInteger)8, pipeline.maxTotalThreadsPerThreadgroup / 32u);
        [encoder dispatchThreadgroups:MTLSizeMake((ow + nsg - 1u) / nsg, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(nsg * 32u, 1, 1)];
    } else if (pipeline == state.row_tiled_pipeline) {
        [encoder dispatchThreadgroups:MTLSizeMake((ow + 63u) / 64u, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    } else if (pipeline == state.tiled_pipeline) {
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

int mynah_metal_batch_begin(void *opaque, char *error, size_t capacity) {
    return metal_ops_batch_begin((__bridge MynahMetalState *)opaque, error, capacity);
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

int mynah_metal_copy_dev(void *opaque, float *destination, const float *source,
                         size_t n, char *error, size_t capacity) {
    MynahMetalState *state = (__bridge MynahMetalState *)opaque;
    if (n == 0u || n > UINT32_MAX || n > SIZE_MAX / sizeof(float)) return -1;
    id<MTLBuffer> src = metal_ops_buffer(state, source, n * sizeof(float));
    id<MTLBuffer> dst = metal_ops_buffer(state, destination, n * sizeof(float));
    id<MTLComputePipelineState> pipeline = metal_ops_pipeline(
        state, @"mynah_ops_copy", error, capacity);
    if (src == nil || dst == nil || pipeline == nil) return -1;
    id<MTLCommandBuffer> command = metal_ops_command(state);
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:src offset:0 atIndex:0];
    [encoder setBuffer:dst offset:0 atIndex:1];
    NSUInteger threads = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    return metal_ops_commit(state, command, error, capacity);
}

int mynah_metal_scale_dev(void *opaque, float *data, size_t n, float scale,
                          char *error, size_t capacity) {
    MynahMetalState *state = (__bridge MynahMetalState *)opaque;
    if (data == NULL || n == 0u || n > UINT32_MAX) return -1;
    id<MTLBuffer> buffer = metal_ops_buffer(state, data, n * sizeof(float));
    id<MTLComputePipelineState> pipeline = metal_ops_pipeline(
        state, @"mynah_ops_scale", error, capacity);
    struct { float value; } params = {scale};
    id<MTLBuffer> params_buffer = metal_ops_params_buffer(state, &params,
                                                           sizeof(params), error, capacity);
    if (buffer == nil || pipeline == nil || params_buffer == nil) return -1;
    id<MTLCommandBuffer> command = metal_ops_command(state);
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBuffer:params_buffer offset:0 atIndex:1];
    NSUInteger threads = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    return metal_ops_commit(state, command, error, capacity);
}

int mynah_metal_clip_dev(void *opaque, float *data, size_t n,
                         char *error, size_t capacity) {
    return metal_ops_dispatch_vector((__bridge MynahMetalState *)opaque,
                                      @"mynah_ops_clip", data, NULL, n,
                                      error, capacity);
}

int mynah_metal_argmax_dev(void *opaque, const float *dev_logits, size_t vocab,
                           size_t codebook_size, size_t eos_id, int allow_eos,
                           unsigned *argmax, char *error, size_t capacity) {
    MynahMetalState *state = (__bridge MynahMetalState *)opaque;
    if (dev_logits == NULL || argmax == NULL || vocab == 0u ||
        vocab > UINT32_MAX || codebook_size > vocab || eos_id > UINT32_MAX)
        return -1;
    id<MTLBuffer> logits = metal_ops_buffer(state, dev_logits, vocab * sizeof(float));
    id<MTLComputePipelineState> pipeline = metal_ops_pipeline(
        state, @"mynah_ops_argmax", error, capacity);
    if (logits == nil || pipeline == nil) return -1;
    if (state.scalar_buffer == nil)
        state.scalar_buffer = [state.device newBufferWithLength:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
    if (state.scalar_buffer == nil) {
        metal_ops_error(error, capacity, @"Metal argmax result buffer allocation failed");
        return -1;
    }
    metal_ops_argmax_params params = {(uint32_t)vocab, (uint32_t)codebook_size,
                                      (uint32_t)eos_id, allow_eos ? 1u : 0u};
    id<MTLBuffer> params_buffer = metal_ops_params_buffer(state, &params,
                                                           sizeof(params), error, capacity);
    if (params_buffer == nil) return -1;
    id<MTLCommandBuffer> command = metal_ops_command(state);
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:logits offset:0 atIndex:0];
    [encoder setBuffer:state.scalar_buffer offset:0 atIndex:1];
    [encoder setBuffer:params_buffer offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    if (metal_ops_commit(state, command, error, capacity) != 0 ||
        metal_ops_sync_state(state, error, capacity) != 0) return -1;
    *argmax = *(const uint32_t *)state.scalar_buffer.contents;
    return 0;
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

int mynah_metal_self_attention_dev(void *opaque, const float *qkv,
                                   float *kcache, float *vcache,
                                   size_t position, size_t cache_stride,
                                   size_t valid, size_t heads, size_t head_width,
                                   float scale, float *output,
                                   char *error, size_t capacity) {
    return metal_ops_attention((__bridge MynahMetalState *)opaque, 1, qkv,
                               kcache, vcache, position, cache_stride, valid,
                               heads, head_width, scale, output, error, capacity);
}

int mynah_metal_cross_attention_dev(void *opaque, const float *q,
                                    const float *kcache, const float *vcache,
                                    size_t valid, size_t cache_stride,
                                    size_t heads, size_t head_width, float scale,
                                    float *output, char *error, size_t capacity) {
    return metal_ops_attention((__bridge MynahMetalState *)opaque, 0, q,
                               (float *)kcache, (float *)vcache, 0, cache_stride,
                               valid, heads, head_width, scale, output,
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

int mynah_metal_conv_transpose_dev(void *opaque, const float *input, float *output,
                                   int in_channels, int out_channels, int length,
                                   int output_length, int kernel, int stride, int groups,
                                   const float *weight, const float *bias,
                                   char *error, size_t capacity) {
    MynahMetalState *state = (__bridge MynahMetalState *)opaque;
    if (in_channels <= 0 || out_channels <= 0 || length <= 0 || output_length <= 0 ||
        kernel <= 0 || stride <= 0 || groups <= 0 || in_channels % groups != 0 ||
        out_channels % groups != 0 || (size_t)in_channels * (size_t)length > UINT32_MAX ||
        (size_t)out_channels * (size_t)output_length > UINT32_MAX ||
        (size_t)kernel > UINT32_MAX || (size_t)stride > UINT32_MAX) return -1;
    id<MTLBuffer> din = metal_ops_buffer(state, input,
                                         (size_t)in_channels * (size_t)length * sizeof(float));
    id<MTLBuffer> dout = metal_ops_buffer(state, output,
                                          (size_t)out_channels * (size_t)output_length * sizeof(float));
    id<MTLBuffer> dw = metal_ops_buffer(state, weight,
                                        (size_t)in_channels * (size_t)(out_channels / groups) *
                                        (size_t)kernel * sizeof(float));
    id<MTLBuffer> db = bias == NULL ? nil : metal_ops_buffer(
        state, bias, (size_t)out_channels * sizeof(float));
    id<MTLComputePipelineState> pipeline = metal_ops_pipeline(
        state, @"mynah_ops_conv_transpose", error, capacity);
    if (din == nil || dout == nil || dw == nil || (bias != NULL && db == nil) ||
        din.contents != input || dout.contents != output || pipeline == nil) return -1;
    uint32_t params_value[7] = {(uint32_t)in_channels, (uint32_t)out_channels,
                                (uint32_t)length, (uint32_t)output_length,
                                (uint32_t)kernel, (uint32_t)stride, (uint32_t)groups};
    id<MTLBuffer> params_buffer = metal_ops_params_buffer(
        state, params_value, sizeof(params_value), error, capacity);
    if (params_buffer == nil) return -1;
    id<MTLCommandBuffer> command = metal_ops_command(state);
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:din offset:0 atIndex:0];
    [encoder setBuffer:dw offset:0 atIndex:1];
    [encoder setBuffer:db offset:0 atIndex:2];
    [encoder setBuffer:dout offset:0 atIndex:3];
    [encoder setBuffer:params_buffer offset:0 atIndex:4];
    NSUInteger threads = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)out_channels * (NSUInteger)output_length, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
    return metal_ops_commit(state, command, error, capacity);
}

int mynah_metal_ops_self_test(void *opaque, char *error, size_t capacity) {
    float *din = NULL;
    float *dout = NULL;
    float *dtmp = NULL;
    float *dqkv = NULL;
    float *dsk = NULL;
    float *dsv = NULL;
    /* Static, not stack. metal_ops_buffer keeps one device copy per host
     * ADDRESS and never refreshes it, on the contract that the host memory is a
     * resident, immutable weight that outlives the cache — true for the mmapped
     * model tensors it exists for. Stack frames recycle addresses, so as locals
     * these arrays landed on the previous self-test's frame and the GPU read
     * that one's numbers: the matmul saw a weight row of {1, 2} and returned
     * 4.5 instead of 1.5. Static storage gives each array its own address for
     * the life of the process, so the cached copy always matches. */
    static float input[] = {1.0f, 2.0f};
    static float weight[] = {1.0f, 0.0f, 0.0f, 1.0f};
    static float bias[] = {0.5f, -0.5f};
    static float zeros[] = {0.0f, 0.0f};
    static float output[2] = {0.0f, 0.0f};
    static float attention_qkv[] = {1.0f, 0.0f, 1.0f, 0.0f, 2.0f, 3.0f};
    static float attention_qkv_2[] = {1.0f, 0.0f, 0.0f, 1.0f, 4.0f, 5.0f};
    static float attention_out[2] = {0.0f, 0.0f};
    static float conv_input[] = {1.0f, 2.0f, 3.0f};
    static float conv_weight[] = {1.0f, 2.0f};
    static float conv_output[3] = {0.0f, 0.0f, 0.0f};
    static float norm_gain[] = {1.0f, 1.0f};
    static float alpha[] = {1.0f, 1.0f};
    MynahMetalState *state = (__bridge MynahMetalState *)opaque;
    if (metal_ops_alloc(state, 2u, &din, error, capacity) != 0 ||
        metal_ops_alloc(state, 2u, &dout, error, capacity) != 0 ||
        metal_ops_alloc(state, 2u, &dtmp, error, capacity) != 0 ||
        metal_ops_alloc(state, 6u, &dqkv, error, capacity) != 0 ||
        metal_ops_alloc(state, 2u, &dsk, error, capacity) != 0 ||
        metal_ops_alloc(state, 2u, &dsv, error, capacity) != 0) goto fail;
    if (mynah_metal_h2d(opaque, input, din, 2u, error, capacity) != 0 ||
        mynah_metal_h2d(opaque, zeros, dtmp, 2u, error, capacity) != 0 ||
        mynah_metal_matmul_d2d(opaque, din, dout, 1u, 2u, 2u, weight, bias,
                               error, capacity) != 0 ||
        mynah_metal_sync(opaque, error, capacity) != 0 ||
        mynah_metal_d2h(opaque, dout, output, 2u, error, capacity) != 0) goto fail;
    if (fabsf(output[0] - 1.5f) > 1e-5f || fabsf(output[1] - 1.5f) > 1e-5f) {
        snprintf(error, capacity, "Metal matmul mismatch: %.8g %.8g (want 1.5 1.5)",
                 (double)output[0], (double)output[1]);
        goto fail;
    }
    if (mynah_metal_h2d(opaque, attention_qkv, dqkv, 6u, error, capacity) != 0 ||
        mynah_metal_self_attention_dev(opaque, dqkv, dsk, dsv, 0u, 2u, 1u,
                                       1u, 2u, 1.0f, dout, error, capacity) != 0 ||
        mynah_metal_sync(opaque, error, capacity) != 0 ||
        mynah_metal_d2h(opaque, dout, attention_out, 2u, error, capacity) != 0) goto fail;
    if (fabsf(attention_out[0] - 2.0f) > 1e-5f || fabsf(attention_out[1] - 3.0f) > 1e-5f) {
        snprintf(error, capacity, "Metal 1-token attention mismatch: %.8g %.8g (want 2 3)",
                 (double)attention_out[0], (double)attention_out[1]);
        goto fail;
    }
    if (mynah_metal_h2d(opaque, attention_qkv_2, dqkv, 6u, error, capacity) != 0 ||
        mynah_metal_self_attention_dev(opaque, dqkv, dsk, dsv, 1u, 2u, 2u,
                                       1u, 2u, 1.0f, dout, error, capacity) != 0 ||
        mynah_metal_sync(opaque, error, capacity) != 0 ||
        mynah_metal_d2h(opaque, dout, attention_out, 2u, error, capacity) != 0) goto fail;
    if (fabsf(attention_out[0] - (2.0f * expf(1.0f) + 4.0f) / (expf(1.0f) + 1.0f)) > 1e-5f ||
        fabsf(attention_out[1] - (3.0f * expf(1.0f) + 5.0f) / (expf(1.0f) + 1.0f)) > 1e-5f) {
        snprintf(error, capacity, "Metal 2-token attention mismatch: %.8g %.8g",
                 (double)attention_out[0], (double)attention_out[1]);
        goto fail;
    }
    if (mynah_metal_cross_attention_dev(opaque, din, dsk, dsv, 1u, 2u,
                                         1u, 2u, 1.0f, dout, error, capacity) != 0 ||
        mynah_metal_sync(opaque, error, capacity) != 0 ||
        mynah_metal_d2h(opaque, dout, attention_out, 2u, error, capacity) != 0) goto fail;
    if (fabsf(attention_out[0] - 2.0f) > 1e-5f ||
        fabsf(attention_out[1] - 3.0f) > 1e-5f) {
        snprintf(error, capacity, "Metal cross attention mismatch: %.8g %.8g",
                 (double)attention_out[0], (double)attention_out[1]);
        goto fail;
    }
    if (mynah_metal_gelu_inplace(opaque, dout, 2u, error, capacity) != 0 ||
        mynah_metal_residual_inplace(opaque, dtmp, dout, 2u, error, capacity) != 0 ||
        mynah_metal_sync(opaque, error, capacity) != 0) goto fail;
    if (mynah_metal_layer_norm_inplace(opaque, dout, dtmp, norm_gain, 1u, 2u,
                                       error, capacity) != 0 ||
        mynah_metal_snake_dev(opaque, dtmp, alpha, 2u, 1u, 2u, error, capacity) != 0 ||
        mynah_metal_sync(opaque, error, capacity) != 0) goto fail;
    {
        const float clip_values[2] = {-2.0f, NAN};
        if (mynah_metal_h2d(opaque, clip_values, dtmp, 2u, error, capacity) != 0 ||
            mynah_metal_clip_dev(opaque, dtmp, 2u, error, capacity) != 0 ||
            mynah_metal_sync(opaque, error, capacity) != 0 ||
            mynah_metal_d2h(opaque, dtmp, output, 2u, error, capacity) != 0) goto fail;
        if (output[0] != -1.0f || output[1] != 0.0f) {
            snprintf(error, capacity, "Metal clip mismatch: %.8g %.8g (want -1 0)",
                     (double)output[0], (double)output[1]);
            goto fail;
        }
    }
    if (mynah_metal_conv1d(opaque, conv_input, conv_output, 1, 1, 3, 2, 1,
                           conv_weight, NULL, error, capacity) != 0) goto fail;
    if (fabsf(conv_output[0] - 2.0f) > 1e-5f ||
        fabsf(conv_output[1] - 5.0f) > 1e-5f ||
        fabsf(conv_output[2] - 8.0f) > 1e-5f) {
        snprintf(error, capacity, "Metal conv1d mismatch: %.8g %.8g %.8g (want 2 5 8)",
                 (double)conv_output[0], (double)conv_output[1], (double)conv_output[2]);
        goto fail;
    }
    mynah_metal_dev_free(opaque, din);
    mynah_metal_dev_free(opaque, dout);
    mynah_metal_dev_free(opaque, dtmp);
    mynah_metal_dev_free(opaque, dqkv);
    mynah_metal_dev_free(opaque, dsk);
    mynah_metal_dev_free(opaque, dsv);
    return 0;
fail:
    mynah_metal_dev_free(opaque, din);
    mynah_metal_dev_free(opaque, dout);
    mynah_metal_dev_free(opaque, dtmp);
    mynah_metal_dev_free(opaque, dqkv);
    mynah_metal_dev_free(opaque, dsk);
    mynah_metal_dev_free(opaque, dsv);
    if (error != NULL && error[0] == '\0')
        snprintf(error, capacity, "%s", "Metal resident ops self-test mismatch");
    return -1;
}
