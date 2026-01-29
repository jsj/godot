/**************************************************************************/
/*  metal_neural_mlp.mm                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Metal 4 Neural MLP Implementation                                      */
/**************************************************************************/

#include "metal_neural_mlp.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Implementation struct with Metal objects
struct MetalNeuralMLPImpl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> command_queue = nil;
    id<MTLComputePipelineState> forward_pipeline = nil;
    id<MTLLibrary> shader_library = nil;
    
    // Cached flattened weights for inference
    id<MTLBuffer> all_weights_buf = nil;
    id<MTLBuffer> all_biases_buf = nil;
    id<MTLBuffer> layer_sizes_buf = nil;
    
    // Layer sizes stored as plain int array (not ObjC objects)
    int *layer_sizes_data = nullptr;
    int layer_sizes_count = 0;
    
    int L = 4;  // Positional encoding levels
    int input_dim = 6;
    int num_layers_val = 0;
    int output_dim = 3;
    
    ~MetalNeuralMLPImpl() {
        if (layer_sizes_data) {
            free(layer_sizes_data);
            layer_sizes_data = nullptr;
        }
        all_weights_buf = nil;
        all_biases_buf = nil;
        layer_sizes_buf = nil;
        forward_pipeline = nil;
        shader_library = nil;
        command_queue = nil;
        device = nil;
    }
};

// Metal compute shader for MLP forward pass
static NSString *mlp_shader_source = @R"(
#include <metal_stdlib>
using namespace metal;

// Positional encoding
void positional_encode(thread float *input, thread float *output, int input_dim, int L) {
    int idx = 0;
    // Copy original
    for (int i = 0; i < input_dim; i++) {
        output[idx++] = input[i];
    }
    // Add sin/cos features
    for (int l = 0; l < L; l++) {
        float freq = pow(2.0f, float(l));
        for (int i = 0; i < input_dim; i++) {
            output[idx++] = sin(freq * input[i]);
            output[idx++] = cos(freq * input[i]);
        }
    }
}

// ReLU activation
float relu(float x) {
    return max(0.0f, x);
}

// Sigmoid activation
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

kernel void mlp_forward(
    device const float *inputs [[buffer(0)]],
    device float *outputs [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device const float *biases [[buffer(3)]],
    device const int *layer_sizes [[buffer(4)]],
    constant int &batch_size [[buffer(5)]],
    constant int &num_layers [[buffer(6)]],
    constant int &input_dim [[buffer(7)]],
    constant int &L [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= uint(batch_size)) return;
    
    // Encoded input dimension
    int encoded_dim = input_dim * (1 + 2 * L);
    
    // Temporary storage for layer activations
    float encoded[128];  // Max encoded size
    float act1[128];     // Activations
    float act2[128];
    
    // Get input for this sample
    float raw_input[16];
    for (int i = 0; i < input_dim; i++) {
        raw_input[i] = inputs[tid * input_dim + i];
    }
    
    // Positional encoding
    positional_encode(raw_input, encoded, input_dim, L);
    
    // Copy to activation buffer
    for (int i = 0; i < encoded_dim; i++) {
        act1[i] = encoded[i];
    }
    
    // Forward through layers
    int weight_offset = 0;
    int bias_offset = 0;
    int current_dim = encoded_dim;
    
    for (int layer = 0; layer < num_layers; layer++) {
        int next_dim = layer_sizes[layer];
        
        // Matrix multiply + bias
        for (int j = 0; j < next_dim; j++) {
            float sum = biases[bias_offset + j];
            for (int i = 0; i < current_dim; i++) {
                sum += act1[i] * weights[weight_offset + j * current_dim + i];
            }
            
            // Activation
            if (layer < num_layers - 1) {
                act2[j] = relu(sum);
            } else {
                act2[j] = sigmoid(sum);  // Output layer
            }
        }
        
        // Swap buffers
        for (int i = 0; i < next_dim; i++) {
            act1[i] = act2[i];
        }
        
        weight_offset += current_dim * next_dim;
        bias_offset += next_dim;
        current_dim = next_dim;
    }
    
    // Write output (RGB)
    for (int i = 0; i < 3; i++) {
        outputs[tid * 3 + i] = act1[i];
    }
}
)";

MetalNeuralMLP::MetalNeuralMLP() {
    impl = nullptr;
}

MetalNeuralMLP::~MetalNeuralMLP() {
    if (impl) {
        delete impl;
        impl = nullptr;
    }
}

Error MetalNeuralMLP::initialize() {
    impl = new MetalNeuralMLPImpl();
    
    impl->device = MTLCreateSystemDefaultDevice();
    if (!impl->device) {
        ERR_PRINT("Failed to create Metal device");
        return ERR_CANT_CREATE;
    }
    
    impl->command_queue = [impl->device newCommandQueue];
    if (!impl->command_queue) {
        ERR_PRINT("Failed to create command queue");
        return ERR_CANT_CREATE;
    }
    
    // Compile shader
    NSError *error = nil;
    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    
    impl->shader_library = [impl->device newLibraryWithSource:mlp_shader_source options:options error:&error];
    if (error) {
        ERR_PRINT(String("Failed to compile MLP shader: ") + String::utf8([error.localizedDescription UTF8String]));
        return ERR_COMPILATION_FAILED;
    }
    
    // Create compute pipeline for MLP forward
    id<MTLFunction> forward_func = [impl->shader_library newFunctionWithName:@"mlp_forward"];
    if (!forward_func) {
        ERR_PRINT("Failed to find mlp_forward function");
        return ERR_CANT_CREATE;
    }
    
    impl->forward_pipeline = [impl->device newComputePipelineStateWithFunction:forward_func error:&error];
    if (error) {
        ERR_PRINT(String("Failed to create pipeline: ") + String::utf8([error.localizedDescription UTF8String]));
        return ERR_CANT_CREATE;
    }
    
    return OK;
}

Error MetalNeuralMLP::load_weights(const String &path) {
    if (!impl || !impl->device) {
        ERR_PRINT("Must call initialize() first");
        return ERR_UNCONFIGURED;
    }
    
    Ref<FileAccess> f = FileAccess::open(path, FileAccess::READ);
    if (f.is_null()) {
        ERR_PRINT("Failed to open weights file: " + path);
        return ERR_FILE_CANT_OPEN;
    }
    
    // Read magic
    uint32_t magic = f->get_32();
    if (magic != 0x4E455552) {  // "NEUR"
        ERR_PRINT("Invalid weight file format");
        return ERR_FILE_CORRUPT;
    }
    
    // Read number of weight/bias pairs
    uint32_t num_pairs = f->get_32();
    num_layers = num_pairs / 2;
    impl->num_layers_val = num_layers;
    
    // Allocate layer sizes array
    if (impl->layer_sizes_data) {
        free(impl->layer_sizes_data);
    }
    impl->layer_sizes_data = (int *)malloc(num_layers * sizeof(int));
    impl->layer_sizes_count = num_layers;
    
    // Temporary storage for flattening
    PackedFloat32Array all_weights;
    PackedFloat32Array all_biases;
    
    // Read all weights and biases
    for (uint32_t i = 0; i < (uint32_t)num_layers; i++) {
        // Weight matrix
        uint32_t rows = f->get_32();
        uint32_t cols = f->get_32();
        
        if (i == 0) {
            input_dim = 6;  // pos + dir
            hidden_dim = rows;
        }
        if (i == (uint32_t)(num_layers - 1)) {
            output_dim = rows;  // Should be 3 (RGB)
            impl->output_dim = output_dim;
        }
        
        impl->layer_sizes_data[i] = rows;
        
        // Read weight data
        for (uint32_t j = 0; j < rows * cols; j++) {
            all_weights.push_back(f->get_float());
        }
        
        // Bias
        uint32_t b_rows = f->get_32();
        uint32_t b_cols = f->get_32();
        (void)b_rows;  // Unused
        
        // Read bias data
        for (uint32_t j = 0; j < b_cols; j++) {
            all_biases.push_back(f->get_float());
        }
    }
    
    // Create Metal buffers for weights and biases
    impl->all_weights_buf = [impl->device newBufferWithBytes:all_weights.ptr() 
                                                      length:all_weights.size() * sizeof(float) 
                                                     options:MTLResourceStorageModeShared];
    impl->all_biases_buf = [impl->device newBufferWithBytes:all_biases.ptr() 
                                                     length:all_biases.size() * sizeof(float) 
                                                    options:MTLResourceStorageModeShared];
    impl->layer_sizes_buf = [impl->device newBufferWithBytes:impl->layer_sizes_data 
                                                      length:impl->layer_sizes_count * sizeof(int) 
                                                     options:MTLResourceStorageModeShared];
    
    print_line(vformat("Loaded MLP: %d layers, input=%d, hidden=%d, output=%d", num_layers, input_dim, hidden_dim, output_dim));
    
    return OK;
}

PackedFloat32Array MetalNeuralMLP::infer_batch(const PackedFloat32Array &inputs, int batch_size) {
    PackedFloat32Array result;
    
    if (!impl || !impl->forward_pipeline || !impl->all_weights_buf) {
        ERR_PRINT("MLP not initialized or weights not loaded");
        return result;
    }
    
    // Create input buffer
    size_t input_bytes = batch_size * input_dim * sizeof(float);
    id<MTLBuffer> in_buf = [impl->device newBufferWithBytes:inputs.ptr() length:input_bytes options:MTLResourceStorageModeShared];
    
    // Create output buffer
    size_t output_bytes = batch_size * output_dim * sizeof(float);
    id<MTLBuffer> out_buf = [impl->device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
    
    // Create command buffer
    id<MTLCommandBuffer> cmd = [impl->command_queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
    
    int L_val = impl->L;
    int input_dim_val = input_dim;
    int num_layers_val = num_layers;
    
    [encoder setComputePipelineState:impl->forward_pipeline];
    [encoder setBuffer:in_buf offset:0 atIndex:0];
    [encoder setBuffer:out_buf offset:0 atIndex:1];
    [encoder setBuffer:impl->all_weights_buf offset:0 atIndex:2];
    [encoder setBuffer:impl->all_biases_buf offset:0 atIndex:3];
    [encoder setBuffer:impl->layer_sizes_buf offset:0 atIndex:4];
    [encoder setBytes:&batch_size length:sizeof(int) atIndex:5];
    [encoder setBytes:&num_layers_val length:sizeof(int) atIndex:6];
    [encoder setBytes:&input_dim_val length:sizeof(int) atIndex:7];
    [encoder setBytes:&L_val length:sizeof(int) atIndex:8];
    
    MTLSize grid_size = MTLSizeMake(batch_size, 1, 1);
    NSUInteger threadgroup_size = MIN(256, (NSUInteger)batch_size);
    MTLSize thread_group_size = MTLSizeMake(threadgroup_size, 1, 1);
    
    [encoder dispatchThreads:grid_size threadsPerThreadgroup:thread_group_size];
    [encoder endEncoding];
    
    [cmd commit];
    [cmd waitUntilCompleted];
    
    // Read results
    result.resize(batch_size * output_dim);
    memcpy(result.ptrw(), out_buf.contents, output_bytes);
    
    return result;
}

PackedByteArray MetalNeuralMLP::render_image(
    const Vector3 &cam_pos,
    const Basis &cam_basis,
    int width,
    int height,
    float fov
) {
    PackedByteArray result;
    
    if (!impl) {
        return result;
    }
    
    int batch_size = width * height;
    
    // Generate ray data (pos + dir for each pixel)
    PackedFloat32Array ray_data;
    ray_data.resize(batch_size * 6);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 6;
            
            float u = float(x) / float(width);
            float v = 1.0f - float(y) / float(height);
            
            Vector3 local_dir = Vector3(
                (u - 0.5f) * fov,
                (v - 0.5f) * fov,
                -1.0f
            ).normalized();
            
            Vector3 world_dir = cam_basis.xform(local_dir);
            
            ray_data.write[idx + 0] = cam_pos.x;
            ray_data.write[idx + 1] = cam_pos.y;
            ray_data.write[idx + 2] = cam_pos.z;
            ray_data.write[idx + 3] = world_dir.x;
            ray_data.write[idx + 4] = world_dir.y;
            ray_data.write[idx + 5] = world_dir.z;
        }
    }
    
    // Run batch inference
    PackedFloat32Array rgb = infer_batch(ray_data, batch_size);
    
    if (rgb.size() != batch_size * 3) {
        return result;
    }
    
    // Convert to byte image
    result.resize(width * height * 3);
    for (int i = 0; i < batch_size; i++) {
        result.write[i * 3 + 0] = uint8_t(CLAMP(rgb[i * 3 + 0] * 255.0f, 0.0f, 255.0f));
        result.write[i * 3 + 1] = uint8_t(CLAMP(rgb[i * 3 + 1] * 255.0f, 0.0f, 255.0f));
        result.write[i * 3 + 2] = uint8_t(CLAMP(rgb[i * 3 + 2] * 255.0f, 0.0f, 255.0f));
    }
    
    return result;
}

void MetalNeuralMLP::_bind_methods() {
    ClassDB::bind_method(D_METHOD("initialize"), &MetalNeuralMLP::initialize);
    ClassDB::bind_method(D_METHOD("load_weights", "path"), &MetalNeuralMLP::load_weights);
    ClassDB::bind_method(D_METHOD("infer_batch", "inputs", "batch_size"), &MetalNeuralMLP::infer_batch);
    ClassDB::bind_method(D_METHOD("render_image", "cam_pos", "cam_basis", "width", "height", "fov"), &MetalNeuralMLP::render_image);
    ClassDB::bind_method(D_METHOD("get_input_dim"), &MetalNeuralMLP::get_input_dim);
    ClassDB::bind_method(D_METHOD("get_output_dim"), &MetalNeuralMLP::get_output_dim);
    ClassDB::bind_method(D_METHOD("get_num_layers"), &MetalNeuralMLP::get_num_layers);
}
