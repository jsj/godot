/**************************************************************************/
/*  metal_neural_singleton.h                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Metal 4 Neural Singleton - GDScript accessible API                     */
/**************************************************************************/

#pragma once

#include "core/object/object.h"
#include "core/variant/variant.h"
#include "servers/rendering/rendering_device.h"

class MetalNeuralSingleton : public Object {
    GDCLASS(MetalNeuralSingleton, Object);

private:
    static MetalNeuralSingleton *singleton;
    bool _initialized = false;
    bool _supported = false;

protected:
    static void _bind_methods();

public:
    static MetalNeuralSingleton *get_singleton() { return singleton; }

    MetalNeuralSingleton();
    ~MetalNeuralSingleton();

    // Check if Metal 4 Neural is supported
    bool is_supported() const;

    // Get detailed capability info
    Dictionary get_capabilities() const;

    // Create a tensor (returns null if not supported)
    Ref<RefCounted> create_tensor(const PackedInt64Array &dimensions, int data_type, int usage);

    // Get the Metal device name
    String get_device_name() const;

    // Check macOS/iOS version
    String get_os_version() const;
};
