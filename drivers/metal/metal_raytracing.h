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

#include "core/math/transform_3d.h"
#include <cstdint>

// Opaque handle for Metal acceleration structures (bridges C++ <-> Obj-C).
// The actual Metal objects are stored internally in the .mm implementation.
struct MDAccelerationStructureHandle {
	uint64_t id = 0;
	bool is_valid() const { return id != 0; }
};

// C++ interface to Metal ray tracing. Implementations in metal_raytracing.mm.
namespace MetalRT {

bool is_supported(void *mtl_device_cpp);

// BLAS creation from vertex/index buffers (Metal-cpp pointers).
MDAccelerationStructureHandle blas_create(
		void *mtl_device_cpp,
		void *vertex_buffer_cpp, uint64_t vertex_offset, uint32_t vertex_count, uint32_t vertex_stride,
		void *index_buffer_cpp, uint64_t index_offset, uint32_t index_count, bool index_32bit);

// TLAS creation from instances.
struct TLASInstance {
	MDAccelerationStructureHandle blas;
	Transform3D transform;
	uint32_t mask = 0xFF;
	uint32_t instance_id = 0;
};

MDAccelerationStructureHandle tlas_create(
		void *mtl_device_cpp,
		const TLASInstance *p_instances, uint32_t p_count);

void acceleration_structure_free(MDAccelerationStructureHandle p_handle);

uint32_t acceleration_structure_get_scratch_size(MDAccelerationStructureHandle p_handle);

// Build/refit an acceleration structure using a command buffer.
void command_build(void *cmd_buffer_cpp, MDAccelerationStructureHandle p_handle, void *scratch_buffer_cpp);

} // namespace MetalRT
