/**************************************************************************/
/*  metal_raytracing.mm                                                   */
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

#import "metal_raytracing.h"
#import <Metal/Metal.h>

#include "core/templates/hash_map.h"

// Internal storage for acceleration structures.
struct MDAccelInternal {
	enum Type { BLAS, TLAS };
	Type type = BLAS;
	id<MTLAccelerationStructure> accel = nil;
	id<MTLBuffer> scratch_buffer = nil;
	uint32_t scratch_size = 0;

	// For TLAS: keep refs to child BLASes alive.
	NSMutableArray<id<MTLAccelerationStructure>> *blas_refs = nil;
	id<MTLBuffer> instance_buffer = nil;

	~MDAccelInternal() {
		accel = nil;
		scratch_buffer = nil;
		instance_buffer = nil;
		blas_refs = nil;
	}
};

static HashMap<uint64_t, MDAccelInternal *> s_accels;
static uint64_t s_next_id = 1;

static MTLPackedFloat4x3 transform_to_metal(const Transform3D &t) {
	MTLPackedFloat4x3 r;
	r.columns[0] = MTLPackedFloat3Make(t.basis[0][0], t.basis[1][0], t.basis[2][0]);
	r.columns[1] = MTLPackedFloat3Make(t.basis[0][1], t.basis[1][1], t.basis[2][1]);
	r.columns[2] = MTLPackedFloat3Make(t.basis[0][2], t.basis[1][2], t.basis[2][2]);
	r.columns[3] = MTLPackedFloat3Make(t.origin.x, t.origin.y, t.origin.z);
	return r;
}

namespace MetalRT {

bool is_supported(void *mtl_device_cpp) {
	if (@available(macOS 11.0, iOS 14.0, *)) {
		id<MTLDevice> dev = (__bridge id<MTLDevice>)mtl_device_cpp;
		return [dev supportsRaytracing];
	}
	return false;
}

MDAccelerationStructureHandle blas_create(
		void *mtl_device_cpp,
		void *vertex_buffer_cpp, uint64_t vertex_offset, uint32_t vertex_count, uint32_t vertex_stride,
		void *index_buffer_cpp, uint64_t index_offset, uint32_t index_count, bool index_32bit) {
	MDAccelerationStructureHandle handle;

	if (@available(macOS 11.0, iOS 14.0, *)) {
		id<MTLDevice> dev = (__bridge id<MTLDevice>)mtl_device_cpp;
		id<MTLBuffer> vb = (__bridge id<MTLBuffer>)vertex_buffer_cpp;
		id<MTLBuffer> ib = index_buffer_cpp ? (__bridge id<MTLBuffer>)index_buffer_cpp : nil;

		MTLAccelerationStructureTriangleGeometryDescriptor *geom =
			[[MTLAccelerationStructureTriangleGeometryDescriptor alloc] init];
		geom.vertexBuffer = vb;
		geom.vertexBufferOffset = vertex_offset;
		geom.vertexStride = vertex_stride;

		if (@available(macOS 13.0, iOS 16.0, *)) {
			geom.vertexFormat = MTLAttributeFormatFloat3;
		}

		if (ib) {
			geom.indexBuffer = ib;
			geom.indexBufferOffset = index_offset;
			geom.indexType = index_32bit ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
			geom.triangleCount = index_count / 3;
		} else {
			geom.triangleCount = vertex_count / 3;
		}

		MTLPrimitiveAccelerationStructureDescriptor *desc =
			[[MTLPrimitiveAccelerationStructureDescriptor alloc] init];
		desc.geometryDescriptors = @[geom];

		MTLAccelerationStructureSizes sizes = [dev accelerationStructureSizesWithDescriptor:desc];

		id<MTLAccelerationStructure> accel = [dev newAccelerationStructureWithSize:sizes.accelerationStructureSize];
		id<MTLBuffer> scratch = [dev newBufferWithLength:sizes.buildScratchBufferSize options:MTLResourceStorageModePrivate];

		// Immediate build.
		id<MTLCommandQueue> queue = [dev newCommandQueue];
		id<MTLCommandBuffer> cmd = [queue commandBuffer];
		id<MTLAccelerationStructureCommandEncoder> enc = [cmd accelerationStructureCommandEncoder];
		[enc buildAccelerationStructure:accel descriptor:desc scratchBuffer:scratch scratchBufferOffset:0];
		[enc endEncoding];
		[cmd commit];
		[cmd waitUntilCompleted];

		MDAccelInternal *internal = new MDAccelInternal();
		internal->type = MDAccelInternal::BLAS;
		internal->accel = accel;
		internal->scratch_buffer = scratch;
		internal->scratch_size = (uint32_t)sizes.buildScratchBufferSize;

		uint64_t id = s_next_id++;
		s_accels[id] = internal;
		handle.id = id;
	}

	return handle;
}

MDAccelerationStructureHandle tlas_create(
		void *mtl_device_cpp,
		const TLASInstance *p_instances, uint32_t p_count) {
	MDAccelerationStructureHandle handle;

	if (@available(macOS 11.0, iOS 14.0, *)) {
		if (p_count == 0) {
			return handle;
		}

		id<MTLDevice> dev = (__bridge id<MTLDevice>)mtl_device_cpp;

		// Gather BLAS references.
		NSMutableArray<id<MTLAccelerationStructure>> *blas_arr = [NSMutableArray arrayWithCapacity:p_count];
		for (uint32_t i = 0; i < p_count; i++) {
			MDAccelInternal *blas_int = s_accels.has(p_instances[i].blas.id) ? s_accels[p_instances[i].blas.id] : nullptr;
			if (!blas_int || !blas_int->accel) {
				return handle;
			}
			[blas_arr addObject:blas_int->accel];
		}

		size_t inst_stride = sizeof(MTLAccelerationStructureInstanceDescriptor);
		id<MTLBuffer> inst_buf = [dev newBufferWithLength:p_count * inst_stride options:MTLResourceStorageModeShared];
		MTLAccelerationStructureInstanceDescriptor *descs = (MTLAccelerationStructureInstanceDescriptor *)[inst_buf contents];

		for (uint32_t i = 0; i < p_count; i++) {
			descs[i].transformationMatrix = transform_to_metal(p_instances[i].transform);
			descs[i].mask = p_instances[i].mask;
			descs[i].accelerationStructureIndex = i;
			descs[i].intersectionFunctionTableOffset = 0;
			descs[i].options = MTLAccelerationStructureInstanceOptionOpaque;
		}

		MTLInstanceAccelerationStructureDescriptor *desc =
			[[MTLInstanceAccelerationStructureDescriptor alloc] init];
		desc.instancedAccelerationStructures = blas_arr;
		desc.instanceCount = p_count;
		desc.instanceDescriptorBuffer = inst_buf;

		MTLAccelerationStructureSizes sizes = [dev accelerationStructureSizesWithDescriptor:desc];
		id<MTLAccelerationStructure> accel = [dev newAccelerationStructureWithSize:sizes.accelerationStructureSize];
		id<MTLBuffer> scratch = [dev newBufferWithLength:sizes.buildScratchBufferSize options:MTLResourceStorageModePrivate];

		id<MTLCommandQueue> queue = [dev newCommandQueue];
		id<MTLCommandBuffer> cmd = [queue commandBuffer];
		id<MTLAccelerationStructureCommandEncoder> enc = [cmd accelerationStructureCommandEncoder];
		[enc buildAccelerationStructure:accel descriptor:desc scratchBuffer:scratch scratchBufferOffset:0];
		[enc endEncoding];
		[cmd commit];
		[cmd waitUntilCompleted];

		MDAccelInternal *internal = new MDAccelInternal();
		internal->type = MDAccelInternal::TLAS;
		internal->accel = accel;
		internal->scratch_buffer = scratch;
		internal->scratch_size = (uint32_t)sizes.buildScratchBufferSize;
		internal->blas_refs = blas_arr;
		internal->instance_buffer = inst_buf;

		uint64_t id = s_next_id++;
		s_accels[id] = internal;
		handle.id = id;
	}

	return handle;
}

void acceleration_structure_free(MDAccelerationStructureHandle p_handle) {
	if (s_accels.has(p_handle.id)) {
		delete s_accels[p_handle.id];
		s_accels.erase(p_handle.id);
	}
}

uint32_t acceleration_structure_get_scratch_size(MDAccelerationStructureHandle p_handle) {
	if (s_accels.has(p_handle.id)) {
		return s_accels[p_handle.id]->scratch_size;
	}
	return 0;
}

void command_build(void *cmd_buffer_cpp, MDAccelerationStructureHandle p_handle, void *scratch_buffer_cpp) {
	// For now this is a no-op since we build immediately in create.
	// A proper implementation would defer building to the command buffer.
	(void)cmd_buffer_cpp;
	(void)p_handle;
	(void)scratch_buffer_cpp;
}

} // namespace MetalRT
