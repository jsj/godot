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
#import "rendering_device_driver_metal.h"

#import <Metal/Metal.h>

MDAccelerationStructure::~MDAccelerationStructure() {
	accel = nil;
	scratch_buffer = nil;
	vertex_buffer = nil;
	index_buffer = nil;
	instance_buffer = nil;
}

#pragma mark - Acceleration Structure Creation

API_AVAILABLE(macos(11.0), ios(14.0))
id<MTLAccelerationStructure> create_blas(
	id<MTLDevice> device,
	id<MTLCommandQueue> queue,
	const MDAccelerationStructureBLASDescriptor &desc,
	id<MTLBuffer> vertex_buffer,
	id<MTLBuffer> index_buffer,
	id<MTLBuffer> *out_scratch_buffer
) {
	// Create geometry descriptor for triangles
	MTLAccelerationStructureTriangleGeometryDescriptor *geom_desc =
		[[MTLAccelerationStructureTriangleGeometryDescriptor alloc] init];
	
	geom_desc.vertexBuffer = vertex_buffer;
	geom_desc.vertexBufferOffset = desc.vertex_offset;
	geom_desc.vertexStride = desc.vertex_stride;
	
	if (@available(macOS 13.0, iOS 16.0, *)) {
		geom_desc.vertexFormat = MTLAttributeFormatFloat3;
	}
	
	if (index_buffer) {
		geom_desc.indexBuffer = index_buffer;
		geom_desc.indexBufferOffset = desc.index_offset;
		geom_desc.indexType = desc.index_32bit ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;
		geom_desc.triangleCount = desc.index_count / 3;
	} else {
		geom_desc.triangleCount = desc.vertex_count / 3;
	}
	
	// Create primitive acceleration structure descriptor (BLAS)
	MTLPrimitiveAccelerationStructureDescriptor *accel_desc =
		[[MTLPrimitiveAccelerationStructureDescriptor alloc] init];
	accel_desc.geometryDescriptors = @[geom_desc];
	
	// Get sizes
	MTLAccelerationStructureSizes sizes =
		[device accelerationStructureSizesWithDescriptor:accel_desc];
	
	// Allocate acceleration structure
	id<MTLAccelerationStructure> accel =
		[device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
	
	// Allocate scratch buffer
	id<MTLBuffer> scratch =
		[device newBufferWithLength:sizes.buildScratchBufferSize
						   options:MTLResourceStorageModePrivate];
	
	// Build
	id<MTLCommandBuffer> cmd = [queue commandBuffer];
	id<MTLAccelerationStructureCommandEncoder> encoder =
		[cmd accelerationStructureCommandEncoder];
	
	[encoder buildAccelerationStructure:accel
							 descriptor:accel_desc
						  scratchBuffer:scratch
					scratchBufferOffset:0];
	
	[encoder endEncoding];
	[cmd commit];
	[cmd waitUntilCompleted];
	
	if (out_scratch_buffer) {
		*out_scratch_buffer = scratch;
	}
	
	return accel;
}

API_AVAILABLE(macos(12.0), ios(15.0))
id<MTLAccelerationStructure> create_tlas(
	id<MTLDevice> device,
	id<MTLCommandQueue> queue,
	const MDAccelerationStructureTLASDescriptor &desc,
	NSArray<id<MTLAccelerationStructure>> *blas_array,
	id<MTLBuffer> *out_instance_buffer,
	id<MTLBuffer> *out_scratch_buffer
) {
	uint32_t instance_count = desc.instances.size();
	if (instance_count == 0) {
		return nil;
	}
	
	// Create instance buffer
	size_t instance_stride = sizeof(MTLAccelerationStructureInstanceDescriptor);
	id<MTLBuffer> instance_buffer =
		[device newBufferWithLength:instance_count * instance_stride
						   options:MTLResourceStorageModeShared];
	
	MTLAccelerationStructureInstanceDescriptor *instances =
		(MTLAccelerationStructureInstanceDescriptor *)[instance_buffer contents];
	
	// Fill instance data
	for (uint32_t i = 0; i < instance_count; i++) {
		const auto &inst = desc.instances[i];
		
		instances[i].transformationMatrix = transform_to_metal(inst.transform);
		instances[i].mask = inst.mask;
		instances[i].intersectionFunctionTableOffset = inst.hit_group_index;
		instances[i].accelerationStructureIndex = inst.instance_id;
		
		// Options: opaque, non-opaque, etc.
		instances[i].options = MTLAccelerationStructureInstanceOptionOpaque;
	}
	
	// Create instance acceleration structure descriptor (TLAS)
	MTLInstanceAccelerationStructureDescriptor *accel_desc =
		[[MTLInstanceAccelerationStructureDescriptor alloc] init];
	accel_desc.instancedAccelerationStructures = blas_array;
	accel_desc.instanceCount = instance_count;
	accel_desc.instanceDescriptorBuffer = instance_buffer;
	accel_desc.instanceDescriptorBufferOffset = 0;
	accel_desc.instanceDescriptorStride = instance_stride;
	
	if (@available(macOS 12.0, iOS 15.0, *)) {
		accel_desc.instanceDescriptorType = MTLAccelerationStructureInstanceDescriptorTypeDefault;
	}
	
	// Get sizes
	MTLAccelerationStructureSizes sizes =
		[device accelerationStructureSizesWithDescriptor:accel_desc];
	
	// Allocate
	id<MTLAccelerationStructure> accel =
		[device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
	
	id<MTLBuffer> scratch =
		[device newBufferWithLength:sizes.buildScratchBufferSize
						   options:MTLResourceStorageModePrivate];
	
	// Build
	id<MTLCommandBuffer> cmd = [queue commandBuffer];
	id<MTLAccelerationStructureCommandEncoder> encoder =
		[cmd accelerationStructureCommandEncoder];
	
	[encoder buildAccelerationStructure:accel
							 descriptor:accel_desc
						  scratchBuffer:scratch
					scratchBufferOffset:0];
	
	[encoder endEncoding];
	[cmd commit];
	[cmd waitUntilCompleted];
	
	if (out_instance_buffer) {
		*out_instance_buffer = instance_buffer;
	}
	if (out_scratch_buffer) {
		*out_scratch_buffer = scratch;
	}
	
	return accel;
}

#pragma mark - Ray Tracing Commands

API_AVAILABLE(macos(11.0), ios(14.0))
void encode_ray_tracing_dispatch(
	id<MTLComputeCommandEncoder> encoder,
	id<MTLComputePipelineState> pipeline,
	id<MTLAccelerationStructure> tlas,
	id<MTLTexture> output_texture,
	uint32_t width,
	uint32_t height
) {
	[encoder setComputePipelineState:pipeline];
	[encoder setAccelerationStructure:tlas atBufferIndex:0];
	[encoder setTexture:output_texture atIndex:0];
	
	// Calculate thread groups
	MTLSize threads_per_group = MTLSizeMake(8, 8, 1);
	MTLSize thread_groups = MTLSizeMake(
		(width + threads_per_group.width - 1) / threads_per_group.width,
		(height + threads_per_group.height - 1) / threads_per_group.height,
		1
	);
	
	[encoder dispatchThreadgroups:thread_groups
			threadsPerThreadgroup:threads_per_group];
}

#pragma mark - Refit (for dynamic geometry)

API_AVAILABLE(macos(11.0), ios(14.0))
void refit_acceleration_structure(
	id<MTLCommandBuffer> cmd,
	id<MTLAccelerationStructure> accel,
	MTLAccelerationStructureDescriptor *desc,
	id<MTLBuffer> scratch_buffer
) {
	id<MTLAccelerationStructureCommandEncoder> encoder =
		[cmd accelerationStructureCommandEncoder];
	
	[encoder refitAccelerationStructure:accel
							 descriptor:desc
							destination:accel
						  scratchBuffer:scratch_buffer
					scratchBufferOffset:0];
	
	[encoder endEncoding];
}
