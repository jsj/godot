/**************************************************************************/
/*  metal_neural.mm                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Metal 4 Neural Shader Support Implementation                           */
/**************************************************************************/

#import "metal_neural.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "core/io/file_access.h"

namespace metal_neural {

bool is_neural_supported(id<MTLDevice> device) {
#if GODOT_METAL4_NEURAL_SUPPORTED
    if (@available(macOS 26.0, iOS 26.0, *)) {
        // Check for Apple Silicon (required for Neural Engine)
        if (![device supportsFamily:MTLGPUFamilyApple7]) {
            return false;
        }
        // Metal 4 tensor support - check by trying to create descriptor
        return true; // If we compiled with SDK 26, tensors are available
    }
#endif
    return false;
}

Dictionary get_neural_capabilities(id<MTLDevice> device) {
    Dictionary caps;
    caps["supported"] = false;
    caps["tensor_ops"] = false;
    caps["shader_ml"] = false;
    caps["max_tensor_dimensions"] = 0;
    caps["neural_engine_available"] = false;
    
#if GODOT_METAL4_NEURAL_SUPPORTED
    if (@available(macOS 26.0, iOS 26.0, *)) {
        if ([device supportsFamily:MTLGPUFamilyApple7]) {
            caps["supported"] = true;
            caps["tensor_ops"] = true;
            caps["shader_ml"] = true;
            caps["max_tensor_dimensions"] = MTL_TENSOR_MAX_RANK;
            caps["neural_engine_available"] = true;
            caps["device_name"] = String::utf8([device.name UTF8String]);
            caps["registry_id"] = (uint64_t)device.registryID;
        }
    }
#endif
    
    return caps;
}

#if GODOT_METAL4_NEURAL_SUPPORTED

// ============================================================================
// MetalTensor Implementation
// ============================================================================

MetalTensor::~MetalTensor() {
    if (tensor != nil) {
        tensor = nil;
    }
}

Ref<MetalTensor> MetalTensor::create_from_device(
    id<MTLDevice> device,
    const Vector<int64_t> &p_dimensions,
    TensorDataType p_data_type,
    uint32_t p_usage
) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
        Ref<MetalTensor> ref;
        ref.instantiate();
        
        MTLTensorDescriptor *desc = [[MTLTensorDescriptor alloc] init];
        
        // Create MTLTensorExtents for dimensions
        NSInteger dims[MTL_TENSOR_MAX_RANK];
        NSUInteger rank = MIN((NSUInteger)p_dimensions.size(), (NSUInteger)MTL_TENSOR_MAX_RANK);
        for (NSUInteger i = 0; i < rank; i++) {
            dims[i] = (NSInteger)p_dimensions[i];
        }
        desc.dimensions = [[MTLTensorExtents alloc] initWithRank:rank values:dims];
        
        // Set data type
        switch (p_data_type) {
            case TensorDataType::FLOAT32:
                desc.dataType = MTLTensorDataTypeFloat32;
                break;
            case TensorDataType::FLOAT16:
                desc.dataType = MTLTensorDataTypeFloat16;
                break;
            case TensorDataType::BFLOAT16:
                desc.dataType = MTLTensorDataTypeBFloat16;
                break;
            case TensorDataType::INT32:
                desc.dataType = MTLTensorDataTypeInt32;
                break;
            case TensorDataType::INT16:
                desc.dataType = MTLTensorDataTypeInt16;
                break;
            case TensorDataType::INT8:
                desc.dataType = MTLTensorDataTypeInt8;
                break;
            case TensorDataType::UINT8:
                desc.dataType = MTLTensorDataTypeUInt8;
                break;
            default:
                desc.dataType = MTLTensorDataTypeFloat32;
        }
        
        // Set usage
        MTLTensorUsage usage = 0;
        if (p_usage & TENSOR_USAGE_MACHINE_LEARNING) {
            usage |= MTLTensorUsageMachineLearning;
        }
        if (p_usage & TENSOR_USAGE_COMPUTE) {
            usage |= MTLTensorUsageCompute;
        }
        if (p_usage & TENSOR_USAGE_RENDER) {
            usage |= MTLTensorUsageRender;
        }
        desc.usage = usage;
        
        // Create tensor
        NSError *error = nil;
        ref->tensor = [device newTensorWithDescriptor:desc error:&error];
        
        if (error != nil) {
            ERR_PRINT(String("Failed to create MTLTensor: ") + 
                     String::utf8([error.localizedDescription UTF8String]));
            return Ref<MetalTensor>();
        }
        
        ref->dimensions = p_dimensions;
        ref->data_type = p_data_type;
        ref->usage_flags = p_usage;
        
        return ref;
    }
    
    return Ref<MetalTensor>();
}

Ref<MetalTensor> MetalTensor::create_from_buffer(
    id<MTLBuffer> buffer,
    const Vector<int64_t> &p_dimensions,
    const Vector<int64_t> &p_strides,
    TensorDataType p_data_type,
    uint32_t p_usage
) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
        Ref<MetalTensor> ref;
        ref.instantiate();
        
        MTLTensorDescriptor *desc = [[MTLTensorDescriptor alloc] init];
        
        // Create MTLTensorExtents for dimensions
        NSInteger dims[MTL_TENSOR_MAX_RANK];
        NSUInteger rank = MIN((NSUInteger)p_dimensions.size(), (NSUInteger)MTL_TENSOR_MAX_RANK);
        for (NSUInteger i = 0; i < rank; i++) {
            dims[i] = (NSInteger)p_dimensions[i];
        }
        desc.dimensions = [[MTLTensorExtents alloc] initWithRank:rank values:dims];
        
        // Create MTLTensorExtents for strides
        if (p_strides.size() > 0) {
            NSInteger strides[MTL_TENSOR_MAX_RANK];
            for (NSUInteger i = 0; i < rank && i < (NSUInteger)p_strides.size(); i++) {
                strides[i] = (NSInteger)p_strides[i];
            }
            desc.strides = [[MTLTensorExtents alloc] initWithRank:rank values:strides];
        }
        
        // Set data type
        switch (p_data_type) {
            case TensorDataType::FLOAT32:
                desc.dataType = MTLTensorDataTypeFloat32;
                break;
            case TensorDataType::FLOAT16:
                desc.dataType = MTLTensorDataTypeFloat16;
                break;
            default:
                desc.dataType = MTLTensorDataTypeFloat32;
        }
        
        // Create tensor from buffer
        NSError *error = nil;
        ref->tensor = [buffer newTensorWithDescriptor:desc offset:0 error:&error];
        
        if (error != nil) {
            ERR_PRINT(String("Failed to create MTLTensor from buffer: ") + 
                     String::utf8([error.localizedDescription UTF8String]));
            return Ref<MetalTensor>();
        }
        
        ref->dimensions = p_dimensions;
        ref->data_type = p_data_type;
        ref->usage_flags = p_usage;
        
        return ref;
    }
    
    return Ref<MetalTensor>();
}

Error MetalTensor::upload_data(const PackedFloat32Array &p_data) {
    if (tensor == nil) {
        return ERR_INVALID_DATA;
    }
    
    if (@available(macOS 26.0, iOS 26.0, *)) {
        NSUInteger rank = dimensions.size();
        
        // Origin: all zeros
        NSInteger originVals[MTL_TENSOR_MAX_RANK] = {0};
        MTLTensorExtents *origin = [[MTLTensorExtents alloc] initWithRank:rank values:originVals];
        
        // Dimensions
        NSInteger dims[MTL_TENSOR_MAX_RANK];
        for (NSUInteger i = 0; i < rank && i < MTL_TENSOR_MAX_RANK; i++) {
            dims[i] = (NSInteger)dimensions[i];
        }
        MTLTensorExtents *sliceDims = [[MTLTensorExtents alloc] initWithRank:rank values:dims];
        
        // Calculate strides (row-major: first dimension is innermost)
        NSInteger strides[MTL_TENSOR_MAX_RANK];
        strides[0] = 1;
        for (NSUInteger i = 1; i < rank && i < MTL_TENSOR_MAX_RANK; i++) {
            strides[i] = strides[i-1] * dims[i-1];
        }
        MTLTensorExtents *strideExtents = [[MTLTensorExtents alloc] initWithRank:rank values:strides];
        
        [tensor replaceSliceOrigin:origin
                   sliceDimensions:sliceDims
                         withBytes:p_data.ptr()
                           strides:strideExtents];
        
        return OK;
    }
    
    return ERR_UNAVAILABLE;
}

PackedFloat32Array MetalTensor::download_data() const {
    PackedFloat32Array result;
    
    if (tensor == nil) {
        return result;
    }
    
    if (@available(macOS 26.0, iOS 26.0, *)) {
        int64_t count = get_element_count();
        result.resize(count);
        
        NSUInteger rank = dimensions.size();
        
        // Origin: all zeros
        NSInteger originVals[MTL_TENSOR_MAX_RANK] = {0};
        MTLTensorExtents *origin = [[MTLTensorExtents alloc] initWithRank:rank values:originVals];
        
        // Dimensions
        NSInteger dims[MTL_TENSOR_MAX_RANK];
        for (NSUInteger i = 0; i < rank && i < MTL_TENSOR_MAX_RANK; i++) {
            dims[i] = (NSInteger)dimensions[i];
        }
        MTLTensorExtents *sliceDims = [[MTLTensorExtents alloc] initWithRank:rank values:dims];
        
        // Strides
        NSInteger strides[MTL_TENSOR_MAX_RANK];
        strides[0] = 1;
        for (NSUInteger i = 1; i < rank && i < MTL_TENSOR_MAX_RANK; i++) {
            strides[i] = strides[i-1] * dims[i-1];
        }
        MTLTensorExtents *strideExtents = [[MTLTensorExtents alloc] initWithRank:rank values:strides];
        
        [tensor getBytes:result.ptrw()
                 strides:strideExtents
         fromSliceOrigin:origin
         sliceDimensions:sliceDims];
    }
    
    return result;
}

int64_t MetalTensor::get_element_count() const {
    int64_t count = 1;
    for (int i = 0; i < dimensions.size(); i++) {
        count *= dimensions[i];
    }
    return count;
}

void MetalTensor::_bind_methods() {
    // Note: We can't directly expose id<MTLDevice> to GDScript
    // These would need wrapper methods that get device from RenderingDevice
    ClassDB::bind_method(D_METHOD("get_dimensions"), &MetalTensor::get_dimensions);
    ClassDB::bind_method(D_METHOD("get_element_count"), &MetalTensor::get_element_count);
    ClassDB::bind_method(D_METHOD("upload_data", "data"), &MetalTensor::upload_data);
    ClassDB::bind_method(D_METHOD("download_data"), &MetalTensor::download_data);
}

// ============================================================================
// MetalNeuralPipeline Implementation
// ============================================================================

MetalNeuralPipeline::~MetalNeuralPipeline() {
    weight_tensors.clear();
    pipeline = nil;
    library = nil;
    intermediates_heap = nil;
}

Error MetalNeuralPipeline::load_from_package(id<MTLDevice> p_device, const String &p_path) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
        device = p_device;
        
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:p_path.utf8().get_data()]];
        NSError *error = nil;
        
        library = [device newLibraryWithURL:url error:&error];
        if (error != nil) {
            ERR_PRINT(String("Failed to load MTLPackage: ") + 
                     String::utf8([error.localizedDescription UTF8String]));
            return ERR_FILE_CANT_OPEN;
        }
        
        return OK;
    }
    
    return ERR_UNAVAILABLE;
}

Error MetalNeuralPipeline::load_weights(const String &p_path) {
    // Load neural network weights from file
    Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
    if (f.is_null()) {
        return ERR_FILE_CANT_OPEN;
    }
    
    // Read header
    uint32_t magic = f->get_32();
    if (magic != 0x4E455552) { // "NEUR"
        return ERR_FILE_CORRUPT;
    }
    
    uint32_t num_layers = f->get_32();
    
    for (uint32_t i = 0; i < num_layers; i++) {
        uint32_t rows = f->get_32();
        uint32_t cols = f->get_32();
        
        Vector<int64_t> dims;
        dims.push_back(rows);
        dims.push_back(cols);
        
        Ref<MetalTensor> tensor = MetalTensor::create_from_device(
            device, dims, TensorDataType::FLOAT16,
            TENSOR_USAGE_COMPUTE | TENSOR_USAGE_RENDER
        );
        
        if (tensor.is_null()) {
            return ERR_CANT_CREATE;
        }
        
        // Read weights
        PackedFloat32Array weights;
        weights.resize(rows * cols);
        for (uint32_t j = 0; j < rows * cols; j++) {
            weights.write[j] = f->get_float();
        }
        tensor->upload_data(weights);
        
        weight_tensors.push_back(tensor);
    }
    
    return OK;
}

void MetalNeuralPipeline::_bind_methods() {
    ClassDB::bind_method(D_METHOD("load_weights", "path"), &MetalNeuralPipeline::load_weights);
}

// ============================================================================
// ShaderMLContext Implementation
// ============================================================================

ShaderMLContext::~ShaderMLContext() {
    ml_functions.clear();
}

Error ShaderMLContext::initialize(id<MTLDevice> p_device) {
    device = p_device;
    return OK;
}

Error ShaderMLContext::compile_shader_with_ml(
    const String &p_source,
    const String &p_function_name,
    id<MTLFunction> *r_function
) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
        NSString *source = [NSString stringWithUTF8String:p_source.utf8().get_data()];
        
        MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
        options.languageVersion = MTLLanguageVersion3_2; // Use latest stable for now
        options.mathMode = MTLMathModeFast;
        
        NSError *error = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:source options:options error:&error];
        
        if (error != nil) {
            ERR_PRINT(String("Shader ML compilation failed: ") + 
                     String::utf8([error.localizedDescription UTF8String]));
            return ERR_COMPILATION_FAILED;
        }
        
        NSString *funcName = [NSString stringWithUTF8String:p_function_name.utf8().get_data()];
        *r_function = [lib newFunctionWithName:funcName];
        
        if (*r_function == nil) {
            return ERR_INVALID_PARAMETER;
        }
        
        return OK;
    }
    
    return ERR_UNAVAILABLE;
}

void ShaderMLContext::_bind_methods() {
    // Note: initialize needs device from RenderingDevice
}

#endif // GODOT_METAL4_NEURAL_SUPPORTED

} // namespace metal_neural
