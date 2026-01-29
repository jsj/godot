/**************************************************************************/
/*  metal_neural_singleton.mm                                             */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Metal 4 Neural Singleton Implementation                                */
/**************************************************************************/

#include "metal_neural_singleton.h"
#include "metal_neural.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

MetalNeuralSingleton *MetalNeuralSingleton::singleton = nullptr;

MetalNeuralSingleton::MetalNeuralSingleton() {
    singleton = this;
    
    // Check if we're on a supported system
#if GODOT_METAL4_NEURAL_SUPPORTED
    if (@available(macOS 26.0, iOS 26.0, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device && [device supportsFamily:MTLGPUFamilyApple7]) {
            _supported = true;
        }
    }
#endif
    _initialized = true;
}

MetalNeuralSingleton::~MetalNeuralSingleton() {
    singleton = nullptr;
}

bool MetalNeuralSingleton::is_supported() const {
    return _supported;
}

Dictionary MetalNeuralSingleton::get_capabilities() const {
    Dictionary caps;
    caps["supported"] = _supported;
    caps["initialized"] = _initialized;
    
#if GODOT_METAL4_NEURAL_SUPPORTED
    if (@available(macOS 26.0, iOS 26.0, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            caps["device_name"] = String::utf8([device.name UTF8String]);
            caps["apple7_family"] = (bool)[device supportsFamily:MTLGPUFamilyApple7];
            caps["apple8_family"] = (bool)[device supportsFamily:MTLGPUFamilyApple8];
            caps["apple9_family"] = (bool)[device supportsFamily:MTLGPUFamilyApple9];
            caps["max_tensor_rank"] = MTL_TENSOR_MAX_RANK;
            Array dtypes;
            dtypes.push_back("float32");
            dtypes.push_back("float16");
            dtypes.push_back("bfloat16");
            dtypes.push_back("int32");
            dtypes.push_back("int16");
            dtypes.push_back("int8");
            dtypes.push_back("uint8");
            caps["tensor_data_types"] = dtypes;
        }
    }
#else
    caps["build_support"] = false;
    caps["reason"] = "Godot not built with Metal 4 Neural support (needs SDK 26+)";
#endif
    
    return caps;
}

Ref<RefCounted> MetalNeuralSingleton::create_tensor(const PackedInt64Array &dimensions, int data_type, int usage) {
#if GODOT_METAL4_NEURAL_SUPPORTED
    if (!_supported) {
        ERR_PRINT("Metal 4 Neural not supported on this device");
        return Ref<RefCounted>();
    }
    
    if (@available(macOS 26.0, iOS 26.0, *)) {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return Ref<RefCounted>();
        }
        
        Vector<int64_t> dims;
        for (int i = 0; i < dimensions.size(); i++) {
            dims.push_back(dimensions[i]);
        }
        
        return metal_neural::MetalTensor::create_from_device(
            device, dims,
            static_cast<metal_neural::TensorDataType>(data_type),
            static_cast<uint32_t>(usage)
        );
    }
#endif
    return Ref<RefCounted>();
}

String MetalNeuralSingleton::get_device_name() const {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        return String::utf8([device.name UTF8String]);
    }
    return "Unknown";
}

String MetalNeuralSingleton::get_os_version() const {
    NSOperatingSystemVersion version = [[NSProcessInfo processInfo] operatingSystemVersion];
    return vformat("%d.%d.%d", (int)version.majorVersion, (int)version.minorVersion, (int)version.patchVersion);
}

void MetalNeuralSingleton::_bind_methods() {
    ClassDB::bind_method(D_METHOD("is_supported"), &MetalNeuralSingleton::is_supported);
    ClassDB::bind_method(D_METHOD("get_capabilities"), &MetalNeuralSingleton::get_capabilities);
    ClassDB::bind_method(D_METHOD("create_tensor", "dimensions", "data_type", "usage"), &MetalNeuralSingleton::create_tensor);
    ClassDB::bind_method(D_METHOD("get_device_name"), &MetalNeuralSingleton::get_device_name);
    ClassDB::bind_method(D_METHOD("get_os_version"), &MetalNeuralSingleton::get_os_version);
    
    // Tensor data type constants
    BIND_CONSTANT(0); // FLOAT32
    BIND_CONSTANT(1); // FLOAT16
    BIND_CONSTANT(2); // BFLOAT16
    BIND_CONSTANT(3); // INT32
    BIND_CONSTANT(4); // INT16
    BIND_CONSTANT(5); // INT8
    BIND_CONSTANT(6); // UINT8
    
    // Tensor usage constants
    BIND_CONSTANT(1); // USAGE_MACHINE_LEARNING
    BIND_CONSTANT(2); // USAGE_COMPUTE
    BIND_CONSTANT(4); // USAGE_RENDER
}
