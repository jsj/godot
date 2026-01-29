/**************************************************************************/
/*  metal_neural.h                                                        */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Metal 4 Neural Shader Support                                          */
/* Adds MTLTensor and Shader ML capabilities for neural rendering         */
/**************************************************************************/

#pragma once

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Metal 4 requires macOS 26+ / iOS 19+
#if __has_include(<Metal/MTLTensor.h>)
#define GODOT_METAL4_NEURAL_SUPPORTED 1
#import <Metal/MTLTensor.h>
#else
#define GODOT_METAL4_NEURAL_SUPPORTED 0
#endif

#include "core/object/ref_counted.h"
#include "core/variant/variant.h"

// Forward declarations
@protocol MTLDevice;
@protocol MTLCommandBuffer;
@protocol MTLComputeCommandEncoder;

namespace metal_neural {

#if GODOT_METAL4_NEURAL_SUPPORTED

/// Tensor data types matching Metal 4
enum class TensorDataType {
    FLOAT32,
    FLOAT16,
    BFLOAT16,
    INT32,
    INT16,
    INT8,
    UINT8
};

/// Tensor usage flags
enum TensorUsage {
    TENSOR_USAGE_MACHINE_LEARNING = 1 << 0,
    TENSOR_USAGE_COMPUTE = 1 << 1,
    TENSOR_USAGE_RENDER = 1 << 2,
};

/// Wrapper for MTLTensor
class API_AVAILABLE(macos(26.0), ios(19.0)) MetalTensor : public RefCounted {
    GDCLASS(MetalTensor, RefCounted);

private:
    id<MTLTensor> tensor = nil;
    Vector<int64_t> dimensions;
    TensorDataType data_type;
    uint32_t usage_flags;

public:
    MetalTensor() = default;
    ~MetalTensor();

    /// Create tensor from device (optimal layout)
    static Ref<MetalTensor> create_from_device(
        id<MTLDevice> device,
        const Vector<int64_t> &p_dimensions,
        TensorDataType p_data_type,
        uint32_t p_usage
    );

    /// Create tensor from existing buffer
    static Ref<MetalTensor> create_from_buffer(
        id<MTLBuffer> buffer,
        const Vector<int64_t> &p_dimensions,
        const Vector<int64_t> &p_strides,
        TensorDataType p_data_type,
        uint32_t p_usage
    );

    /// Upload data to tensor
    Error upload_data(const PackedFloat32Array &p_data);
    Error upload_data_float16(const PackedByteArray &p_data);

    /// Download data from tensor
    PackedFloat32Array download_data() const;

    /// Getters
    id<MTLTensor> get_metal_tensor() const { return tensor; }
    Vector<int64_t> get_dimensions() const { return dimensions; }
    TensorDataType get_data_type() const { return data_type; }
    int64_t get_element_count() const;

protected:
    static void _bind_methods();
};

/// Neural network pipeline for ML inference
class API_AVAILABLE(macos(26.0), ios(19.0)) MetalNeuralPipeline : public RefCounted {
    GDCLASS(MetalNeuralPipeline, RefCounted);

private:
    id<MTLDevice> device = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLLibrary> library = nil;
    
    // Network weights
    Vector<Ref<MetalTensor>> weight_tensors;
    
    // Intermediate heap for ML workloads
    id<MTLHeap> intermediates_heap = nil;

public:
    MetalNeuralPipeline() = default;
    ~MetalNeuralPipeline();

    /// Load neural network from .mtlpackage
    Error load_from_package(id<MTLDevice> p_device, const String &p_path);

    /// Load weights from file
    Error load_weights(const String &p_path);

    /// Bind to compute encoder for dispatch
    void bind(id<MTLComputeCommandEncoder> encoder);

    /// Dispatch inference
    void dispatch(
        id<MTLComputeCommandEncoder> encoder,
        const Ref<MetalTensor> &input,
        const Ref<MetalTensor> &output
    );

protected:
    static void _bind_methods();
};

/// Shader ML integration - for embedding tiny MLPs in shaders
class API_AVAILABLE(macos(26.0), ios(19.0)) ShaderMLContext : public RefCounted {
    GDCLASS(ShaderMLContext, RefCounted);

private:
    id<MTLDevice> device = nil;
    
    // Pre-compiled shader functions with ML ops
    HashMap<String, id<MTLFunction>> ml_functions;

public:
    ShaderMLContext() = default;
    ~ShaderMLContext();

    /// Initialize with Metal device
    Error initialize(id<MTLDevice> p_device);

    /// Compile shader with ML includes
    /// This enables #include <metal_tensor> and MPP operations
    Error compile_shader_with_ml(
        const String &p_source,
        const String &p_function_name,
        id<MTLFunction> *r_function
    );

    /// Create argument buffer for tensor bindings
    id<MTLBuffer> create_tensor_argument_buffer(
        const Vector<Ref<MetalTensor>> &p_tensors
    );

protected:
    static void _bind_methods();
};

#else // !GODOT_METAL4_NEURAL_SUPPORTED

// Stub classes for older systems
class MetalTensor : public RefCounted {
    GDCLASS(MetalTensor, RefCounted);
public:
    static bool is_supported() { return false; }
protected:
    static void _bind_methods() {}
};

class MetalNeuralPipeline : public RefCounted {
    GDCLASS(MetalNeuralPipeline, RefCounted);
public:
    static bool is_supported() { return false; }
protected:
    static void _bind_methods() {}
};

class ShaderMLContext : public RefCounted {
    GDCLASS(ShaderMLContext, RefCounted);
public:
    static bool is_supported() { return false; }
protected:
    static void _bind_methods() {}
};

#endif // GODOT_METAL4_NEURAL_SUPPORTED

/// Check if Metal 4 neural features are available at runtime
bool is_neural_supported(id<MTLDevice> device);

/// Get Metal 4 neural capability info
Dictionary get_neural_capabilities(id<MTLDevice> device);

} // namespace metal_neural
