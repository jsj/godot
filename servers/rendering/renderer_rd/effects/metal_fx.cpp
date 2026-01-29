/**************************************************************************/
/*  metal_fx.mm                                                           */
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

#import "metal_fx.h"

#import "../storage_rd/render_scene_buffers_rd.h"
#import "drivers/metal/pixel_formats.h"
#import "drivers/metal/rendering_device_driver_metal.h"

#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

using namespace RendererRD;

#pragma mark - Spatial Scaler

MFXSpatialContext::~MFXSpatialContext() {
}

MFXSpatialEffect::MFXSpatialEffect() {
}

MFXSpatialEffect::~MFXSpatialEffect() {
}

void MFXSpatialEffect::callback(RDD *p_driver, RDD::CommandBufferID p_command_buffer, CallbackArgs *p_userdata) {
	GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")

	MDCommandBuffer *obj = (MDCommandBuffer *)(p_command_buffer.id);
	obj->end();

	id<MTLTexture> src_texture = rid::get(p_userdata->src);
	id<MTLTexture> dst_texture = rid::get(p_userdata->dst);

	__block id<MTLFXSpatialScaler> scaler = p_userdata->ctx.scaler;
	scaler.colorTexture = src_texture;
	scaler.outputTexture = dst_texture;
	[scaler encodeToCommandBuffer:obj->get_command_buffer()];
	// TODO(sgc): add API to retain objects until the command buffer completes
	[obj->get_command_buffer() addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
		// This block retains a reference to the scaler until the command buffer.
		// completes.
		scaler = nil;
	}];

	CallbackArgs::free(&p_userdata);

	GODOT_CLANG_WARNING_POP
}

void MFXSpatialEffect::ensure_context(Ref<RenderSceneBuffersRD> p_render_buffers) {
	p_render_buffers->ensure_mfx(this);
}

void MFXSpatialEffect::process(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_src, RID p_dst) {
	MFXSpatialContext *ctx = p_render_buffers->get_mfx_spatial_context();
	DEV_ASSERT(ctx); // this should have been done by the caller via ensure_context

	CallbackArgs *userdata = args_allocator.alloc(
			this,
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_src)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_dst)),
			*ctx);
	RD::CallbackResource res[2] = {
		{ .rid = p_src, .usage = RD::CALLBACK_RESOURCE_USAGE_TEXTURE_SAMPLE },
		{ .rid = p_dst, .usage = RD::CALLBACK_RESOURCE_USAGE_STORAGE_IMAGE_READ_WRITE }
	};
	RD::get_singleton()->driver_callback_add((RDD::DriverCallback)MFXSpatialEffect::callback, userdata, VectorView<RD::CallbackResource>(res, 2));
}

MFXSpatialContext *MFXSpatialEffect::create_context(CreateParams p_params) const {
	DEV_ASSERT(RD::get_singleton()->has_feature(RD::SUPPORTS_METALFX_SPATIAL));

	GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")

	RenderingDeviceDriverMetal *rdd = (RenderingDeviceDriverMetal *)RD::get_singleton()->get_device_driver();
	PixelFormats &pf = rdd->get_pixel_formats();
	id<MTLDevice> dev = rdd->get_device();

	MTLFXSpatialScalerDescriptor *desc = [MTLFXSpatialScalerDescriptor new];
	desc.inputWidth = (NSUInteger)p_params.input_size.width;
	desc.inputHeight = (NSUInteger)p_params.input_size.height;

	desc.outputWidth = (NSUInteger)p_params.output_size.width;
	desc.outputHeight = (NSUInteger)p_params.output_size.height;

	desc.colorTextureFormat = pf.getMTLPixelFormat(p_params.input_format);
	desc.outputTextureFormat = pf.getMTLPixelFormat(p_params.output_format);
	desc.colorProcessingMode = MTLFXSpatialScalerColorProcessingModeLinear;
	id<MTLFXSpatialScaler> scaler = [desc newSpatialScalerWithDevice:dev];
	MFXSpatialContext *context = memnew(MFXSpatialContext);
	context->scaler = scaler;

	GODOT_CLANG_WARNING_POP

	return context;
}

#ifdef METAL_MFXTEMPORAL_ENABLED

#pragma mark - Temporal Scaler

MFXTemporalContext::~MFXTemporalContext() {}

MFXTemporalEffect::MFXTemporalEffect() {}
MFXTemporalEffect::~MFXTemporalEffect() {}

MFXTemporalContext *MFXTemporalEffect::create_context(CreateParams p_params) const {
	DEV_ASSERT(RD::get_singleton()->has_feature(RD::SUPPORTS_METALFX_TEMPORAL));

	GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")

	RenderingDeviceDriverMetal *rdd = (RenderingDeviceDriverMetal *)RD::get_singleton()->get_device_driver();
	PixelFormats &pf = rdd->get_pixel_formats();
	id<MTLDevice> dev = rdd->get_device();

	MTLFXTemporalScalerDescriptor *desc = [MTLFXTemporalScalerDescriptor new];
	desc.inputWidth = (NSUInteger)p_params.input_size.width;
	desc.inputHeight = (NSUInteger)p_params.input_size.height;

	desc.outputWidth = (NSUInteger)p_params.output_size.width;
	desc.outputHeight = (NSUInteger)p_params.output_size.height;

	desc.colorTextureFormat = pf.getMTLPixelFormat(p_params.input_format);
	desc.depthTextureFormat = pf.getMTLPixelFormat(p_params.depth_format);
	desc.motionTextureFormat = pf.getMTLPixelFormat(p_params.motion_format);
	desc.autoExposureEnabled = NO;

	desc.outputTextureFormat = pf.getMTLPixelFormat(p_params.output_format);

	id<MTLFXTemporalScaler> scaler = [desc newTemporalScalerWithDevice:dev];
	MFXTemporalContext *context = memnew(MFXTemporalContext);
	context->scaler = scaler;

	scaler.motionVectorScaleX = p_params.motion_vector_scale.x;
	scaler.motionVectorScaleY = p_params.motion_vector_scale.y;
	scaler.depthReversed = true; // Godot uses reverse Z per https://github.com/godotengine/godot/pull/88328

	GODOT_CLANG_WARNING_POP

	return context;
}

void MFXTemporalEffect::process(RendererRD::MFXTemporalContext *p_ctx, RendererRD::MFXTemporalEffect::Params p_params) {
	CallbackArgs *userdata = args_allocator.alloc(
			this,
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.src)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.depth)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.motion)),
			p_params.exposure.is_valid() ? RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.exposure)) : RDD::TextureID(),
			p_params.jitter_offset,
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.dst)),
			*p_ctx,
			p_params.reset);
	RD::CallbackResource res[3] = {
		{ .rid = p_params.src, .usage = RD::CALLBACK_RESOURCE_USAGE_TEXTURE_SAMPLE },
		{ .rid = p_params.depth, .usage = RD::CALLBACK_RESOURCE_USAGE_TEXTURE_SAMPLE },
		{ .rid = p_params.dst, .usage = RD::CALLBACK_RESOURCE_USAGE_STORAGE_IMAGE_READ_WRITE },
	};
	RD::get_singleton()->driver_callback_add((RDD::DriverCallback)MFXTemporalEffect::callback, userdata, VectorView<RD::CallbackResource>(res, 3));
}

void MFXTemporalEffect::callback(RDD *p_driver, RDD::CommandBufferID p_command_buffer, CallbackArgs *p_userdata) {
	GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")

	MDCommandBuffer *obj = (MDCommandBuffer *)(p_command_buffer.id);
	obj->end();

	id<MTLTexture> src_texture = rid::get(p_userdata->src);
	id<MTLTexture> depth = rid::get(p_userdata->depth);
	id<MTLTexture> motion = rid::get(p_userdata->motion);
	id<MTLTexture> exposure = rid::get(p_userdata->exposure);

	id<MTLTexture> dst_texture = rid::get(p_userdata->dst);

	__block id<MTLFXTemporalScaler> scaler = p_userdata->ctx.scaler;
	scaler.reset = p_userdata->reset;
	scaler.colorTexture = src_texture;
	scaler.depthTexture = depth;
	scaler.motionTexture = motion;
	scaler.exposureTexture = exposure;
	scaler.jitterOffsetX = p_userdata->jitter_offset.x;
	scaler.jitterOffsetY = p_userdata->jitter_offset.y;
	scaler.outputTexture = dst_texture;
	[scaler encodeToCommandBuffer:obj->get_command_buffer()];
	// TODO(sgc): add API to retain objects until the command buffer completes
	[obj->get_command_buffer() addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
		// This block retains a reference to the scaler until the command buffer.
		// completes.
		scaler = nil;
	}];

	CallbackArgs::free(&p_userdata);

	GODOT_CLANG_WARNING_POP
}

#endif

#ifdef METAL4_MFXDENOISER_ENABLED

#pragma mark - Temporal Denoised Scaler (Metal 4)

MFXDenoisedContext::~MFXDenoisedContext() {}

MFXDenoisedEffect::MFXDenoisedEffect() {}
MFXDenoisedEffect::~MFXDenoisedEffect() {}

bool MFXDenoisedEffect::is_supported() {
	if (@available(macOS 26.0, iOS 26.0, *)) {
		RenderingDeviceDriverMetal *rdd = (RenderingDeviceDriverMetal *)RD::get_singleton()->get_device_driver();
		id<MTLDevice> dev = rdd->get_device();
		return [MTLFXTemporalDenoisedScalerDescriptor supportsDevice:dev];
	}
	return false;
}

MFXDenoisedContext *MFXDenoisedEffect::create_context(CreateParams p_params) const {
	if (@available(macOS 26.0, iOS 26.0, *)) {
		GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")

		RenderingDeviceDriverMetal *rdd = (RenderingDeviceDriverMetal *)RD::get_singleton()->get_device_driver();
		PixelFormats &pf = rdd->get_pixel_formats();
		id<MTLDevice> dev = rdd->get_device();

		MTLFXTemporalDenoisedScalerDescriptor *desc = [MTLFXTemporalDenoisedScalerDescriptor new];
		desc.inputWidth = (NSUInteger)p_params.input_size.width;
		desc.inputHeight = (NSUInteger)p_params.input_size.height;
		desc.outputWidth = (NSUInteger)p_params.output_size.width;
		desc.outputHeight = (NSUInteger)p_params.output_size.height;

		desc.colorTextureFormat = pf.getMTLPixelFormat(p_params.color_format);
		desc.depthTextureFormat = pf.getMTLPixelFormat(p_params.depth_format);
		desc.motionTextureFormat = pf.getMTLPixelFormat(p_params.motion_format);
		desc.normalTextureFormat = pf.getMTLPixelFormat(p_params.normal_format);
		desc.diffuseAlbedoTextureFormat = pf.getMTLPixelFormat(p_params.albedo_format);
		desc.roughnessTextureFormat = pf.getMTLPixelFormat(p_params.roughness_format);
		desc.outputTextureFormat = pf.getMTLPixelFormat(p_params.output_format);

		[desc setIsAutoExposureEnabled:YES];

		id<MTLFXTemporalDenoisedScaler> scaler = [desc makeTemporalDenoisedScalerWithDevice:dev];
		if (!scaler) {
			ERR_PRINT("Failed to create MTLFXTemporalDenoisedScaler");
			return nullptr;
		}

		scaler.motionVectorScaleX = p_params.motion_vector_scale.x;
		scaler.motionVectorScaleY = p_params.motion_vector_scale.y;
		scaler.isDepthReversed = true; // Godot uses reverse Z

		MFXDenoisedContext *context = memnew(MFXDenoisedContext);
		context->scaler = scaler;

		GODOT_CLANG_WARNING_POP

		return context;
	}
	return nullptr;
}

void MFXDenoisedEffect::process(MFXDenoisedContext *p_ctx, Params p_params) {
	CallbackArgs *userdata = args_allocator.alloc(
			this,
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.color)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.depth)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.motion)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.normal)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.diffuse_albedo)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.roughness)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.dst)),
			*p_ctx,
			p_params.jitter_offset,
			p_params.world_to_view,
			p_params.view_to_clip,
			p_params.reset);

	RD::CallbackResource res[4] = {
		{ .rid = p_params.color, .usage = RD::CALLBACK_RESOURCE_USAGE_TEXTURE_SAMPLE },
		{ .rid = p_params.depth, .usage = RD::CALLBACK_RESOURCE_USAGE_TEXTURE_SAMPLE },
		{ .rid = p_params.motion, .usage = RD::CALLBACK_RESOURCE_USAGE_TEXTURE_SAMPLE },
		{ .rid = p_params.dst, .usage = RD::CALLBACK_RESOURCE_USAGE_STORAGE_IMAGE_READ_WRITE },
	};
	RD::get_singleton()->driver_callback_add((RDD::DriverCallback)MFXDenoisedEffect::callback, userdata, VectorView<RD::CallbackResource>(res, 4));
}

void MFXDenoisedEffect::callback(RDD *p_driver, RDD::CommandBufferID p_command_buffer, CallbackArgs *p_userdata) {
	if (@available(macOS 26.0, iOS 26.0, *)) {
		GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")

		MDCommandBuffer *obj = (MDCommandBuffer *)(p_command_buffer.id);
		obj->end();

		id<MTLTexture> color = rid::get(p_userdata->color);
		id<MTLTexture> depth = rid::get(p_userdata->depth);
		id<MTLTexture> motion = rid::get(p_userdata->motion);
		id<MTLTexture> normal = rid::get(p_userdata->normal);
		id<MTLTexture> albedo = rid::get(p_userdata->albedo);
		id<MTLTexture> roughness = rid::get(p_userdata->roughness);
		id<MTLTexture> dst = rid::get(p_userdata->dst);

		__block id<MTLFXTemporalDenoisedScaler> scaler = p_userdata->ctx.scaler;
		scaler.shouldResetHistory = p_userdata->reset;
		scaler.colorTexture = color;
		scaler.depthTexture = depth;
		scaler.motionTexture = motion;
		scaler.normalTexture = normal;
		scaler.diffuseAlbedoTexture = albedo;
		scaler.roughnessTexture = roughness;
		scaler.outputTexture = dst;

		scaler.jitterOffsetX = p_userdata->jitter_offset.x;
		scaler.jitterOffsetY = p_userdata->jitter_offset.y;

		// Set matrices for denoising
		Transform3D w2v = p_userdata->world_to_view;
		simd_float4x4 world_to_view = {
			simd_make_float4(w2v.basis[0][0], w2v.basis[1][0], w2v.basis[2][0], 0),
			simd_make_float4(w2v.basis[0][1], w2v.basis[1][1], w2v.basis[2][1], 0),
			simd_make_float4(w2v.basis[0][2], w2v.basis[1][2], w2v.basis[2][2], 0),
			simd_make_float4(w2v.origin.x, w2v.origin.y, w2v.origin.z, 1)
		};
		scaler.worldToViewMatrix = world_to_view;

		Projection v2c = p_userdata->view_to_clip;
		simd_float4x4 view_to_clip = {
			simd_make_float4(v2c.columns[0][0], v2c.columns[0][1], v2c.columns[0][2], v2c.columns[0][3]),
			simd_make_float4(v2c.columns[1][0], v2c.columns[1][1], v2c.columns[1][2], v2c.columns[1][3]),
			simd_make_float4(v2c.columns[2][0], v2c.columns[2][1], v2c.columns[2][2], v2c.columns[2][3]),
			simd_make_float4(v2c.columns[3][0], v2c.columns[3][1], v2c.columns[3][2], v2c.columns[3][3])
		};
		scaler.viewToClipMatrix = view_to_clip;

		[scaler encodeToCommandBuffer:obj->get_command_buffer()];

		[obj->get_command_buffer() addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
			scaler = nil;
		}];

		CallbackArgs::free(&p_userdata);

		GODOT_CLANG_WARNING_POP
	}
}

#endif // METAL4_MFXDENOISER_ENABLED

#ifdef METAL4_MFXFRAMEINTERP_ENABLED

#pragma mark - Frame Interpolation (Metal 4)

MFXFrameInterpContext::~MFXFrameInterpContext() {}

MFXFrameInterpEffect::MFXFrameInterpEffect() {}
MFXFrameInterpEffect::~MFXFrameInterpEffect() {}

bool MFXFrameInterpEffect::is_supported() {
	if (@available(macOS 26.0, iOS 26.0, *)) {
		RenderingDeviceDriverMetal *rdd = (RenderingDeviceDriverMetal *)RD::get_singleton()->get_device_driver();
		id<MTLDevice> dev = rdd->get_device();
		return [MTLFXFrameInterpolatorDescriptor supportsDevice:dev];
	}
	return false;
}

MFXFrameInterpContext *MFXFrameInterpEffect::create_context(CreateParams p_params) const {
	if (@available(macOS 26.0, iOS 26.0, *)) {
		GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")

		RenderingDeviceDriverMetal *rdd = (RenderingDeviceDriverMetal *)RD::get_singleton()->get_device_driver();
		PixelFormats &pf = rdd->get_pixel_formats();
		id<MTLDevice> dev = rdd->get_device();

		MTLFXFrameInterpolatorDescriptor *desc = [MTLFXFrameInterpolatorDescriptor new];
		desc.inputWidth = (NSUInteger)p_params.input_size.width;
		desc.inputHeight = (NSUInteger)p_params.input_size.height;
		desc.outputWidth = (NSUInteger)p_params.output_size.width;
		desc.outputHeight = (NSUInteger)p_params.output_size.height;

		desc.colorTextureFormat = pf.getMTLPixelFormat(p_params.color_format);
		desc.depthTextureFormat = pf.getMTLPixelFormat(p_params.depth_format);
		desc.motionTextureFormat = pf.getMTLPixelFormat(p_params.motion_format);
		desc.outputTextureFormat = pf.getMTLPixelFormat(p_params.output_format);

		id<MTLFXFrameInterpolator> interpolator = [desc makeFrameInterpolatorWithDevice:dev];
		if (!interpolator) {
			ERR_PRINT("Failed to create MTLFXFrameInterpolator");
			return nullptr;
		}

		MFXFrameInterpContext *context = memnew(MFXFrameInterpContext);
		context->interpolator = interpolator;

		GODOT_CLANG_WARNING_POP

		return context;
	}
	return nullptr;
}

void MFXFrameInterpEffect::process(MFXFrameInterpContext *p_ctx, Params p_params) {
	CallbackArgs *userdata = args_allocator.alloc(
			this,
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.color_previous)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.color_current)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.depth)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.motion)),
			RDD::TextureID(RD::get_singleton()->get_driver_resource(RDC::DRIVER_RESOURCE_TEXTURE, p_params.dst)),
			*p_ctx);

	RD::CallbackResource res[4] = {
		{ .rid = p_params.color_previous, .usage = RD::CALLBACK_RESOURCE_USAGE_TEXTURE_SAMPLE },
		{ .rid = p_params.color_current, .usage = RD::CALLBACK_RESOURCE_USAGE_TEXTURE_SAMPLE },
		{ .rid = p_params.depth, .usage = RD::CALLBACK_RESOURCE_USAGE_TEXTURE_SAMPLE },
		{ .rid = p_params.dst, .usage = RD::CALLBACK_RESOURCE_USAGE_STORAGE_IMAGE_READ_WRITE },
	};
	RD::get_singleton()->driver_callback_add((RDD::DriverCallback)MFXFrameInterpEffect::callback, userdata, VectorView<RD::CallbackResource>(res, 4));
}

void MFXFrameInterpEffect::callback(RDD *p_driver, RDD::CommandBufferID p_command_buffer, CallbackArgs *p_userdata) {
	if (@available(macOS 26.0, iOS 26.0, *)) {
		GODOT_CLANG_WARNING_PUSH_AND_IGNORE("-Wunguarded-availability")

		MDCommandBuffer *obj = (MDCommandBuffer *)(p_command_buffer.id);
		obj->end();

		// Note: Frame interpolator requires specific setup based on Apple's docs
		// The scaler property needs to be configured with previous/current frames
		__block id<MTLFXFrameInterpolator> interpolator = p_userdata->ctx.interpolator;

		[interpolator encodeToCommandBuffer:obj->get_command_buffer()];

		[obj->get_command_buffer() addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull) {
			interpolator = nil;
		}];

		CallbackArgs::free(&p_userdata);

		GODOT_CLANG_WARNING_POP
	}
}

#endif // METAL4_MFXFRAMEINTERP_ENABLED
