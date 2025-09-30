#include "BgBlur.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/types.hpp>
#include <util\platform.h>
#include <wchar.h>

/*static*/
bool BgBlurGraphics::getRGBAFromStageSurface(gs_texrender_t* texrender, gs_stagesurf_t* stagesurface, obs_source_t* source, uint32_t& width, uint32_t& height, cv::Mat& outputBGRA)
{
	// Captures a live video frame from a source, renders it to a texture, transfers it onto
	//	a staging surface, maps it into CPU-accessible memory, then it wraps the pixel buffer into an OpenCV cv::Mat (BGRA format)

	if (!obs_source_enabled(source))
		return false;

	obs_source_t *target = obs_filter_get_target(source);

	if (!target)
		return false;

	width = obs_source_get_base_width(target);
	height = obs_source_get_base_height(target);

	if (width == 0 || height == 0)
		return false;
	
	gs_texrender_reset(texrender);

	if (!gs_texrender_begin(texrender, width, height))
		return false;

	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(texrender);

	if (stagesurface)
	{
		uint32_t stagesurf_width = gs_stagesurface_get_width(stagesurface);
		uint32_t stagesurf_height = gs_stagesurface_get_height(stagesurface);

		if (stagesurf_width != width || stagesurf_height != height)
		{
			gs_stagesurface_destroy(stagesurface);
			stagesurface = nullptr;
		}
	}

	if (!stagesurface)
		stagesurface = gs_stagesurface_create(width, height, GS_BGRA);

	gs_stage_texture(stagesurface, gs_texrender_get_texture(texrender));

	uint8_t* video_data;
	uint32_t linesize;

	if (!gs_stagesurface_map(stagesurface, &video_data, &linesize))
		return false;

	outputBGRA = cv::Mat(height, width, CV_8UC4, video_data, linesize);

	gs_stagesurface_unmap(stagesurface);
	return true;
}

/*static*/
gs_texture_t* BgBlurGraphics::blurBackground(gs_texrender_t* texrender, gs_effect_t* kawaseBlurEffect, uint32_t amount, uint32_t width, uint32_t height, gs_texture_t *alphaTexture)
{
	if (amount == 0 || !kawaseBlurEffect)
		return nullptr;

	gs_texture_t* blurredTexture = gs_texture_create(width, height, GS_BGRA, 1, nullptr, 0);
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(texrender));

	gs_eparam_t* image = gs_effect_get_param_by_name(kawaseBlurEffect, "image");
	gs_eparam_t* focalmask = gs_effect_get_param_by_name(kawaseBlurEffect, "focalmask");
	gs_eparam_t* xOffset = gs_effect_get_param_by_name(kawaseBlurEffect, "xOffset");
	gs_eparam_t* yOffset = gs_effect_get_param_by_name(kawaseBlurEffect, "yOffset");
	gs_eparam_t* blurIter = gs_effect_get_param_by_name(kawaseBlurEffect, "blurIter");
	gs_eparam_t* blurTotal = gs_effect_get_param_by_name(kawaseBlurEffect, "blurTotal");
	gs_eparam_t* blurFocusPointParam = gs_effect_get_param_by_name(kawaseBlurEffect, "blurFocusPoint");
	gs_eparam_t* blurFocusDepthParam = gs_effect_get_param_by_name(kawaseBlurEffect, "blurFocusDepth");

	for (uint32_t i = 0; i < amount; i++)
	{
		gs_texrender_reset(texrender);

		if (!gs_texrender_begin(texrender, width, height))
		{
			blog(LOG_INFO, "BgBlurGraphics::blurBackground - Could not open background blur texrender!");
			return blurredTexture;
		}

		gs_effect_set_texture(image, blurredTexture);
		gs_effect_set_texture(focalmask, alphaTexture);
		gs_effect_set_float(xOffset, ((float)i + 0.5f) / (float)width);
		gs_effect_set_float(yOffset, ((float)i + 0.5f) / (float)height);
		gs_effect_set_int(blurIter, (int)i);
		gs_effect_set_int(blurTotal, (int)amount);

		// Edit as needed
		static float blurFocusPoint = 0.1f;
		static float blurFocusDepth = 0.0f;
		gs_effect_set_float(blurFocusPointParam, blurFocusPoint);
		gs_effect_set_float(blurFocusDepthParam, blurFocusDepth);

		struct vec4 background;
		vec4_zero(&background);
		gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
		gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

		const char* blur_type = "Draw";

		while (gs_effect_loop(kawaseBlurEffect, blur_type))
			gs_draw_sprite(blurredTexture, 0, width, height);
		
		gs_blend_state_pop();
		gs_texrender_end(texrender);
		gs_copy_texture(blurredTexture, gs_texrender_get_texture(texrender));
	}

	return blurredTexture;
}
