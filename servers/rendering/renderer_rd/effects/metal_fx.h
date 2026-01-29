/**************************************************************************/
/*  metal_fx.h                                                            */
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

#if defined(METAL_ENABLED) && !defined(VISIONOS_ENABLED)
#define METAL_MFXTEMPORAL_ENABLED
// Metal 4 features require macOS 26+ / iOS 26+ SDK
// TODO: Enable these when building with Xcode 26+ / macOS 26 SDK
// These use MTLFXTemporalDenoisedScaler and MTLFXFrameInterpolator APIs
// #define METAL4_MFXDENOISER_ENABLED
// #define METAL4_MFXFRAMEINTERP_ENABLED
#endif

#ifdef METAL_ENABLED

#include "spatial_upscaler.h"

#include "core/templates/paged_allocator.h"
#include "servers/rendering/renderer_scene_render.h"

#ifdef __OBJC__
@protocol MTLFXSpatialScaler;
@protocol MTLFXTemporalScaler;
@protocol MTLFXTemporalDenoisedScaler;
@protocol MTLFXFrameInterpolator;
#endif

namespace RendererRD {

struct MFXSpatialContext {
#ifdef __OBJC__
	id<MTLFXSpatialScaler> scaler = nullptr;
#else
	void *scaler = nullptr;
#endif
	MFXSpatialContext() = default;
	~MFXSpatialContext();
};

class MFXSpatialEffect : public SpatialUpscaler {
	struct CallbackArgs {
		MFXSpatialEffect *owner;
		RDD::TextureID src;
		RDD::TextureID dst;
		MFXSpatialContext ctx;

		CallbackArgs(MFXSpatialEffect *p_owner, RDD::TextureID p_src, RDD::TextureID p_dst, MFXSpatialContext p_ctx) :
				owner(p_owner), src(p_src), dst(p_dst), ctx(p_ctx) {}

		static void free(CallbackArgs **p_args) {
			(*p_args)->owner->args_allocator.free(*p_args);
			*p_args = nullptr;
		}
	};

	PagedAllocator<CallbackArgs, true, 16> args_allocator;
	static void callback(RDD *p_driver, RDD::CommandBufferID p_command_buffer, CallbackArgs *p_userdata);

public:
	virtual const Span<char> get_label() const final { return "MetalFX Spatial Upscale"; }
	virtual void ensure_context(Ref<RenderSceneBuffersRD> p_render_buffers) final;
	virtual void process(Ref<RenderSceneBuffersRD> p_render_buffers, RID p_src, RID p_dst) final;

	struct CreateParams {
		Vector2i input_size;
		Vector2i output_size;
		RD::DataFormat input_format;
		RD::DataFormat output_format;
	};

	MFXSpatialContext *create_context(CreateParams p_params) const;

	MFXSpatialEffect();
	~MFXSpatialEffect();
};

#ifdef METAL_MFXTEMPORAL_ENABLED

struct MFXTemporalContext {
#ifdef __OBJC__
	id<MTLFXTemporalScaler> scaler = nullptr;
#else
	void *scaler = nullptr;
#endif
	MFXTemporalContext() = default;
	~MFXTemporalContext();
};

class MFXTemporalEffect {
	struct CallbackArgs {
		MFXTemporalEffect *owner;
		RDD::TextureID src;
		RDD::TextureID depth;
		RDD::TextureID motion;
		RDD::TextureID exposure;
		Vector2 jitter_offset;
		RDD::TextureID dst;
		MFXTemporalContext ctx;
		bool reset = false;

		CallbackArgs(
				MFXTemporalEffect *p_owner,
				RDD::TextureID p_src,
				RDD::TextureID p_depth,
				RDD::TextureID p_motion,
				RDD::TextureID p_exposure,
				Vector2 p_jitter_offset,
				RDD::TextureID p_dst,
				MFXTemporalContext p_ctx,
				bool p_reset) :
				owner(p_owner),
				src(p_src),
				depth(p_depth),
				motion(p_motion),
				exposure(p_exposure),
				jitter_offset(p_jitter_offset),
				dst(p_dst),
				ctx(p_ctx),
				reset(p_reset) {}

		static void free(CallbackArgs **p_args) {
			(*p_args)->owner->args_allocator.free(*p_args);
			*p_args = nullptr;
		}
	};

	PagedAllocator<CallbackArgs, true, 16> args_allocator;

	static void callback(RDD *p_driver, RDD::CommandBufferID p_command_buffer, CallbackArgs *p_userdata);

public:
	MFXTemporalEffect();
	~MFXTemporalEffect();

	struct CreateParams {
		Vector2i input_size;
		Vector2i output_size;
		RD::DataFormat input_format;
		RD::DataFormat depth_format;
		RD::DataFormat motion_format;
		RD::DataFormat reactive_format;
		RD::DataFormat output_format;
		Vector2 motion_vector_scale;
	};

	MFXTemporalContext *create_context(CreateParams p_params) const;

	struct Params {
		RID src;
		RID depth;
		RID motion;
		RID exposure;
		RID dst;
		Vector2 jitter_offset;
		bool reset = false;
	};

	void process(MFXTemporalContext *p_ctx, Params p_params);
};

#endif

#ifdef METAL4_MFXDENOISER_ENABLED

/// MetalFX Temporal Denoised Scaler (Metal 4 / macOS 26+)
/// Combines denoising with temporal upscaling for ray-traced content
struct MFXDenoisedContext {
#ifdef __OBJC__
	id<MTLFXTemporalDenoisedScaler> scaler = nullptr;
#else
	void *scaler = nullptr;
#endif
	MFXDenoisedContext() = default;
	~MFXDenoisedContext();
};

class MFXDenoisedEffect {
	struct CallbackArgs {
		MFXDenoisedEffect *owner;
		RDD::TextureID color;
		RDD::TextureID depth;
		RDD::TextureID motion;
		RDD::TextureID normal;
		RDD::TextureID albedo;
		RDD::TextureID roughness;
		RDD::TextureID dst;
		MFXDenoisedContext ctx;
		Vector2 jitter_offset;
		Transform3D world_to_view;
		Projection view_to_clip;
		bool reset = false;

		CallbackArgs(
				MFXDenoisedEffect *p_owner,
				RDD::TextureID p_color,
				RDD::TextureID p_depth,
				RDD::TextureID p_motion,
				RDD::TextureID p_normal,
				RDD::TextureID p_albedo,
				RDD::TextureID p_roughness,
				RDD::TextureID p_dst,
				MFXDenoisedContext p_ctx,
				Vector2 p_jitter,
				const Transform3D &p_world_to_view,
				const Projection &p_view_to_clip,
				bool p_reset) :
				owner(p_owner),
				color(p_color),
				depth(p_depth),
				motion(p_motion),
				normal(p_normal),
				albedo(p_albedo),
				roughness(p_roughness),
				dst(p_dst),
				ctx(p_ctx),
				jitter_offset(p_jitter),
				world_to_view(p_world_to_view),
				view_to_clip(p_view_to_clip),
				reset(p_reset) {}

		static void free(CallbackArgs **p_args) {
			(*p_args)->owner->args_allocator.free(*p_args);
			*p_args = nullptr;
		}
	};

	PagedAllocator<CallbackArgs, true, 16> args_allocator;

	static void callback(RDD *p_driver, RDD::CommandBufferID p_command_buffer, CallbackArgs *p_userdata);

public:
	MFXDenoisedEffect();
	~MFXDenoisedEffect();

	static bool is_supported();

	struct CreateParams {
		Vector2i input_size;
		Vector2i output_size;
		RD::DataFormat color_format;
		RD::DataFormat depth_format;
		RD::DataFormat motion_format;
		RD::DataFormat normal_format;
		RD::DataFormat albedo_format;
		RD::DataFormat roughness_format;
		RD::DataFormat output_format;
		Vector2 motion_vector_scale;
	};

	MFXDenoisedContext *create_context(CreateParams p_params) const;

	struct Params {
		RID color;           // Noisy ray traced color
		RID depth;           // Depth buffer
		RID motion;          // Motion vectors
		RID normal;          // World-space normals
		RID diffuse_albedo;  // Diffuse albedo (for denoising)
		RID roughness;       // Roughness
		RID dst;             // Output
		Vector2 jitter_offset;
		Transform3D world_to_view;
		Projection view_to_clip;
		bool reset = false;
	};

	void process(MFXDenoisedContext *p_ctx, Params p_params);
};

#endif // METAL4_MFXDENOISER_ENABLED

#ifdef METAL4_MFXFRAMEINTERP_ENABLED

/// MetalFX Frame Interpolation (Metal 4 / macOS 26+)
/// Generates intermediate frames for higher perceived frame rates
struct MFXFrameInterpContext {
#ifdef __OBJC__
	id<MTLFXFrameInterpolator> interpolator = nullptr;
#else
	void *interpolator = nullptr;
#endif
	MFXFrameInterpContext() = default;
	~MFXFrameInterpContext();
};

class MFXFrameInterpEffect {
	struct CallbackArgs {
		MFXFrameInterpEffect *owner;
		RDD::TextureID color_prev;
		RDD::TextureID color_curr;
		RDD::TextureID depth;
		RDD::TextureID motion;
		RDD::TextureID dst;
		MFXFrameInterpContext ctx;

		CallbackArgs(
				MFXFrameInterpEffect *p_owner,
				RDD::TextureID p_prev,
				RDD::TextureID p_curr,
				RDD::TextureID p_depth,
				RDD::TextureID p_motion,
				RDD::TextureID p_dst,
				MFXFrameInterpContext p_ctx) :
				owner(p_owner),
				color_prev(p_prev),
				color_curr(p_curr),
				depth(p_depth),
				motion(p_motion),
				dst(p_dst),
				ctx(p_ctx) {}

		static void free(CallbackArgs **p_args) {
			(*p_args)->owner->args_allocator.free(*p_args);
			*p_args = nullptr;
		}
	};

	PagedAllocator<CallbackArgs, true, 16> args_allocator;

	static void callback(RDD *p_driver, RDD::CommandBufferID p_command_buffer, CallbackArgs *p_userdata);

public:
	MFXFrameInterpEffect();
	~MFXFrameInterpEffect();

	static bool is_supported();

	struct CreateParams {
		Vector2i input_size;
		Vector2i output_size;
		RD::DataFormat color_format;
		RD::DataFormat depth_format;
		RD::DataFormat motion_format;
		RD::DataFormat output_format;
	};

	MFXFrameInterpContext *create_context(CreateParams p_params) const;

	struct Params {
		RID color_previous;  // Previous frame
		RID color_current;   // Current frame
		RID depth;           // Current depth
		RID motion;          // Motion vectors
		RID dst;             // Interpolated output
	};

	void process(MFXFrameInterpContext *p_ctx, Params p_params);
};

#endif // METAL4_MFXFRAMEINTERP_ENABLED

} //namespace RendererRD

#endif // METAL_ENABLED
