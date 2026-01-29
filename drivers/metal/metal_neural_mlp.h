/**************************************************************************/
/*  metal_neural_mlp.h                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Metal 4 Neural MLP - Batch inference on Neural Engine                  */
/**************************************************************************/

#pragma once

#include "core/object/ref_counted.h"
#include "core/io/file_access.h"
#include "core/math/vector3.h"
#include "core/math/basis.h"
#include "core/variant/typed_array.h"

// Forward declare opaque pointer for Metal objects
// Actual implementation in .mm file
struct MetalNeuralMLPImpl;

/// A simple MLP that runs on Metal compute shaders
/// Supports batch inference for rendering (all pixels at once)
class MetalNeuralMLP : public RefCounted {
    GDCLASS(MetalNeuralMLP, RefCounted);

private:
    MetalNeuralMLPImpl *impl = nullptr;
    
    // Network architecture (cached for GDScript access)
    int num_layers = 0;
    int input_dim = 0;
    int hidden_dim = 0;
    int output_dim = 0;

protected:
    static void _bind_methods();

public:
    MetalNeuralMLP();
    ~MetalNeuralMLP();
    
    /// Initialize with Metal device
    Error initialize();
    
    /// Load weights from .neur file
    Error load_weights(const String &path);
    
    /// Run batch inference
    /// Input: PackedFloat32Array of size [batch_size * input_dim]
    /// Output: PackedFloat32Array of size [batch_size * output_dim]
    PackedFloat32Array infer_batch(const PackedFloat32Array &inputs, int batch_size);
    
    /// Run inference for a full image (optimized for rendering)
    /// Returns RGB image data as PackedByteArray
    PackedByteArray render_image(
        const Vector3 &cam_pos,
        const Basis &cam_basis,
        int width,
        int height,
        float fov
    );
    
    int get_input_dim() const { return input_dim; }
    int get_output_dim() const { return output_dim; }
    int get_num_layers() const { return num_layers; }
};
