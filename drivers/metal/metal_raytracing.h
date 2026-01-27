/**************************************************************************/
/*  metal_raytracing.h                                                    */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#import <Metal/Metal.h>
#import <simd/simd.h>

#include "core/math/transform_3d.h"
#include "servers/rendering/rendering_device_driver.h"

// Metal Ray Tracing support - requires macOS 11.0+ / iOS 14.0+
// Hardware acceleration on Apple Silicon (M1+) and AMD GPUs

class API_AVAILABLE(macos(11.0), ios(14.0)) MDAccelerationStructure {
public:
	enum Type {
		TYPE_BOTTOM_LEVEL,  // BLAS - geometry
		TYPE_TOP_LEVEL,     // TLAS - instances
	};

	Type type = TYPE_BOTTOM_LEVEL;
	id<MTLAccelerationStructure> accel = nil;
	id<MTLBuffer> scratch_buffer = nil;
	
	// For BLAS
	id<MTLBuffer> vertex_buffer = nil;
	id<MTLBuffer> index_buffer = nil;
	uint32_t triangle_count = 0;
	
	// For TLAS
	id<MTLBuffer> instance_buffer = nil;
	uint32_t instance_count = 0;
	
	// Build state
	bool needs_build = true;
	bool needs_refit = false;
	
	MDAccelerationStructure() = default;
	~MDAccelerationStructure();
};

// Descriptor for creating a BLAS
struct MDAccelerationStructureBLASDescriptor {
	RenderingDeviceDriver::BufferID vertex_buffer;
	uint64_t vertex_offset = 0;
	uint32_t vertex_count = 0;
	uint32_t vertex_stride = 12; // sizeof(float) * 3
	
	RenderingDeviceDriver::BufferID index_buffer;
	uint64_t index_offset = 0;
	uint32_t index_count = 0;
	bool index_32bit = true;
	
	// Optional: per-primitive data
	RenderingDeviceDriver::BufferID primitive_data_buffer;
	uint32_t primitive_data_stride = 0;
};

// Descriptor for creating a TLAS
struct MDAccelerationStructureTLASDescriptor {
	struct Instance {
		RenderingDeviceDriver::ID blas_id;
		Transform3D transform;
		uint32_t mask = 0xFF;
		uint32_t instance_id = 0;
		uint32_t hit_group_index = 0;
	};
	
	Vector<Instance> instances;
};

// Ray tracing pipeline descriptor
struct MDRayTracingPipelineDescriptor {
	RenderingDeviceDriver::ShaderID raygen_shader;
	Vector<RenderingDeviceDriver::ShaderID> miss_shaders;
	Vector<RenderingDeviceDriver::ShaderID> closest_hit_shaders;
	Vector<RenderingDeviceDriver::ShaderID> any_hit_shaders;
	Vector<RenderingDeviceDriver::ShaderID> intersection_shaders;
	
	uint32_t max_recursion_depth = 1;
	uint32_t max_payload_size = 32;
	uint32_t max_attribute_size = 8;
};

// Ray tracing dispatch parameters
struct MDRayTracingDispatchParams {
	uint32_t width = 1;
	uint32_t height = 1;
	uint32_t depth = 1;
};

// Utility to convert Godot Transform3D to Metal's packed float4x3
API_AVAILABLE(macos(11.0), ios(14.0))
inline MTLPackedFloat4x3 transform_to_metal(const Transform3D &p_transform) {
	MTLPackedFloat4x3 result;
	
	// Metal expects column-major 3x4 matrix (transposed 4x3)
	// Row 0: basis.x + origin.x
	result.columns[0] = MTLPackedFloat3Make(
		p_transform.basis[0][0],
		p_transform.basis[1][0],
		p_transform.basis[2][0]
	);
	
	// Row 1: basis.y + origin.y
	result.columns[1] = MTLPackedFloat3Make(
		p_transform.basis[0][1],
		p_transform.basis[1][1],
		p_transform.basis[2][1]
	);
	
	// Row 2: basis.z + origin.z
	result.columns[2] = MTLPackedFloat3Make(
		p_transform.basis[0][2],
		p_transform.basis[1][2],
		p_transform.basis[2][2]
	);
	
	// Row 3: origin
	result.columns[3] = MTLPackedFloat3Make(
		p_transform.origin.x,
		p_transform.origin.y,
		p_transform.origin.z
	);
	
	return result;
}

// Check if device supports ray tracing
API_AVAILABLE(macos(11.0), ios(14.0))
inline bool metal_supports_raytracing(id<MTLDevice> device) {
	if (@available(macOS 11.0, iOS 14.0, *)) {
		return [device supportsRaytracing];
	}
	return false;
}
