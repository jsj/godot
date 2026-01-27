# Metal Ray Tracing Implementation Plan for Godot

## Overview
Add hardware-accelerated ray tracing to Godot's Metal driver using Apple's Metal Ray Tracing APIs.

## Required Metal APIs
- `MTLAccelerationStructure` - BVH for geometry
- `MTLPrimitiveAccelerationStructureDescriptor` - Triangle/AABB geometry  
- `MTLInstanceAccelerationStructureDescriptor` - Instance transforms
- `MTLIntersectionFunctionTable` - Custom intersection shaders
- `MTLComputePipelineState` with ray tracing functions

## Files to Modify/Create

### 1. New Files
```
drivers/metal/
├── metal_raytracing.h          # RT struct definitions
├── metal_raytracing.mm         # RT implementation
└── metal_acceleration_structure.h/mm
```

### 2. Modify Existing Files

#### `servers/rendering/rendering_device_commons.h`
Add new Features enum:
```cpp
enum Features {
    // ... existing ...
    SUPPORTS_RAY_TRACING,
    SUPPORTS_RAY_TRACING_PIPELINE,
    SUPPORTS_RAY_QUERY,
};
```

#### `servers/rendering/rendering_device_driver.h`
Add new ID types and methods:
```cpp
DEFINE_ID(AccelerationStructure);

// Acceleration Structure
virtual AccelerationStructureID acceleration_structure_create_blas(
    BufferID p_vertex_buffer,
    BufferID p_index_buffer,
    uint32_t p_vertex_count,
    uint32_t p_index_count,
    DataFormat p_vertex_format
) = 0;

virtual AccelerationStructureID acceleration_structure_create_tlas(
    VectorView<AccelerationStructureID> p_instances,
    VectorView<Transform3D> p_transforms
) = 0;

virtual void acceleration_structure_free(AccelerationStructureID p_accel) = 0;

// Ray Tracing Pipeline
virtual PipelineID ray_tracing_pipeline_create(
    ShaderID p_raygen_shader,
    VectorView<ShaderID> p_miss_shaders,
    VectorView<ShaderID> p_hit_shaders
) = 0;

// Commands
virtual void command_build_acceleration_structure(
    CommandBufferID p_cmd_buffer,
    AccelerationStructureID p_accel
) = 0;

virtual void command_trace_rays(
    CommandBufferID p_cmd_buffer,
    PipelineID p_pipeline,
    uint32_t p_width,
    uint32_t p_height,
    uint32_t p_depth
) = 0;
```

#### `drivers/metal/rendering_device_driver_metal.h`
Add implementations for the new methods.

#### `drivers/metal/rendering_device_driver_metal.mm`
Implement Metal RT APIs:

```objc
// BLAS Creation
AccelerationStructureID RenderingDeviceDriverMetal::acceleration_structure_create_blas(...) {
    MTLPrimitiveAccelerationStructureDescriptor *desc = 
        [[MTLPrimitiveAccelerationStructureDescriptor alloc] init];
    
    MTLAccelerationStructureTriangleGeometryDescriptor *geom = 
        [[MTLAccelerationStructureTriangleGeometryDescriptor alloc] init];
    geom.vertexBuffer = vertex_buffer;
    geom.vertexStride = vertex_stride;
    geom.indexBuffer = index_buffer;
    geom.indexType = MTLIndexTypeUInt32;
    geom.triangleCount = index_count / 3;
    
    desc.geometryDescriptors = @[geom];
    
    MTLAccelerationStructureSizes sizes = 
        [device accelerationStructureSizesWithDescriptor:desc];
    
    id<MTLAccelerationStructure> accel = 
        [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    
    // Build command
    id<MTLCommandBuffer> cmd = [device_queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> encoder = 
        [cmd accelerationStructureCommandEncoder];
    
    id<MTLBuffer> scratch = [device newBufferWithLength:sizes.buildScratchBufferSize 
                                               options:MTLResourceStorageModePrivate];
    
    [encoder buildAccelerationStructure:accel
                             descriptor:desc
                          scratchBuffer:scratch
                    scratchBufferOffset:0];
    [encoder endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    
    return AccelerationStructureID(accel);
}
```

### 3. Shader Support

#### New MSL shader functions needed:
```metal
#include <metal_raytracing>

using namespace metal::raytracing;

struct Ray {
    float3 origin;
    float3 direction;
    float min_distance;
    float max_distance;
};

kernel void raytrace_kernel(
    uint2 tid [[thread_position_in_grid]],
    instance_acceleration_structure accel [[buffer(0)]],
    texture2d<float, access::write> output [[texture(0)]]
) {
    intersector<triangle_data> i;
    i.assume_geometry_type(geometry_type::triangle);
    
    Ray ray = generate_camera_ray(tid);
    
    intersection_result<triangle_data> result = i.intersect(
        ray, accel, intersection_params()
    );
    
    if (result.type != intersection_type::none) {
        // Hit - shade the intersection
        float3 color = shade_hit(result);
        output.write(float4(color, 1.0), tid);
    } else {
        // Miss - sky color
        output.write(float4(0.5, 0.7, 1.0, 1.0), tid);
    }
}
```

## Implementation Phases

### Phase 1: Basic Infrastructure (Week 1)
- [ ] Add Feature flag SUPPORTS_RAY_TRACING
- [ ] Create AccelerationStructureID type
- [ ] Implement BLAS creation for triangle meshes
- [ ] Implement TLAS creation for instances

### Phase 2: Ray Tracing Pipeline (Week 2)
- [ ] Add ray tracing pipeline creation
- [ ] Implement command_trace_rays
- [ ] Basic compute shader RT test

### Phase 3: Integration (Week 3)
- [ ] Add to Forward+ renderer for shadows
- [ ] Add to Forward+ renderer for reflections
- [ ] GI probe ray tracing

### Phase 4: Optimization (Week 4)
- [ ] Denoising integration (MetalFX?)
- [ ] Temporal accumulation
- [ ] LOD for acceleration structures

## Hardware Requirements
- Apple Silicon (M1+) or AMD GPU with macOS 11.0+
- Metal 3 for best performance

## Testing
- Cornell box scene
- Sponza with RT shadows
- Reflective surfaces
- Performance benchmarks vs software RT

## References
- [Metal Ray Tracing WWDC 2021](https://developer.apple.com/videos/play/wwdc2021/10149/)
- [Metal Ray Tracing Sample Code](https://developer.apple.com/metal/sample-code/)
- [Godot RenderingDevice Architecture](https://docs.godotengine.org/en/stable/contributing/development/core_and_modules/internal_rendering_architecture.html)
