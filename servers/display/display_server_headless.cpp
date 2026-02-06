/**************************************************************************/
/*  display_server_headless.cpp                                           */
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

#include "display_server_headless.h"

#include "core/input/input.h"
#include "core/input/input_event.h"
#include "servers/display/native_menu.h"
#include "servers/rendering/dummy/rasterizer_dummy.h"

#if defined(RD_ENABLED)
#include "servers/rendering/renderer_rd/renderer_compositor_rd.h"
#include "servers/rendering/rendering_device.h"

#if defined(METAL_ENABLED)
#include "drivers/metal/rendering_context_driver_metal.h"
#endif
#if defined(VULKAN_ENABLED)
#if defined(MACOS_ENABLED)
#include "platform/macos/rendering_context_driver_vulkan_macos.h"
#elif defined(LINUXBSD_ENABLED)
#include "platform/linuxbsd/rendering_context_driver_vulkan_linuxbsd.h"
#elif defined(WINDOWS_ENABLED)
#include "platform/windows/rendering_context_driver_vulkan_windows.h"
#endif
#endif
#endif

DisplayServer *DisplayServerHeadless::create_func(const String &p_rendering_driver, DisplayServerEnums::WindowMode p_mode, DisplayServerEnums::VSyncMode p_vsync_mode, uint32_t p_flags, const Vector2i *p_position, const Vector2i &p_resolution, int p_screen, DisplayServerEnums::Context p_context, int64_t p_parent_window, Error &r_error) {
	r_error = OK;
	DisplayServerHeadless *display_server = memnew(DisplayServerHeadless());
	display_server->window_size = p_resolution;

#if defined(RD_ENABLED)
	if (p_rendering_driver != "dummy") {
#if defined(METAL_ENABLED)
		if (p_rendering_driver == "metal") {
			display_server->rendering_context = memnew(RenderingContextDriverMetal);
		}
#endif
#if defined(VULKAN_ENABLED)
		if (p_rendering_driver == "vulkan") {
#if defined(MACOS_ENABLED)
			display_server->rendering_context = memnew(RenderingContextDriverVulkanMacOS);
#elif defined(LINUXBSD_ENABLED)
			display_server->rendering_context = memnew(RenderingContextDriverVulkanLinuxBSD);
#elif defined(WINDOWS_ENABLED)
			display_server->rendering_context = memnew(RenderingContextDriverVulkanWindows);
#endif
		}
#endif

		if (display_server->rendering_context && display_server->rendering_context->initialize() == OK) {
			display_server->rendering_device = memnew(RenderingDevice);
			if (display_server->rendering_device->initialize(display_server->rendering_context) == OK) {
				RendererCompositorRD::make_current();
				return display_server;
			}
			memdelete(display_server->rendering_device);
			display_server->rendering_device = nullptr;
		}
		if (display_server->rendering_context) {
			memdelete(display_server->rendering_context);
			display_server->rendering_context = nullptr;
		}
		WARN_PRINT("Headless rendering device initialization failed. Godot will use the dummy renderer.");
	}
#endif

	RasterizerDummy::make_current();
	return display_server;
}

void DisplayServerHeadless::_dispatch_input_events(const Ref<InputEvent> &p_event) {
	static_cast<DisplayServerHeadless *>(get_singleton())->_dispatch_input_event(p_event);
}

void DisplayServerHeadless::_dispatch_input_event(const Ref<InputEvent> &p_event) {
	if (input_event_callback.is_valid()) {
		input_event_callback.call(p_event);
	}
}

void DisplayServerHeadless::process_events() {
	Input::get_singleton()->flush_buffered_events();
}

DisplayServerHeadless::DisplayServerHeadless() {
	native_menu = memnew(NativeMenu);
	Input::get_singleton()->set_event_dispatch_function(_dispatch_input_events);
}

DisplayServerHeadless::~DisplayServerHeadless() {
#if defined(RD_ENABLED)
	if (rendering_device) {
		memdelete(rendering_device);
	}
	if (rendering_context) {
		memdelete(rendering_context);
	}
#endif
	if (native_menu) {
		memdelete(native_menu);
		native_menu = nullptr;
	}
}
