#include "BgBlur.h"
#include "OnnxInstance.h"

#include <util\platform.h>
#include <wchar.h>

/*static*/
bool BgBlurGraphics::getRGBAFromStageSurface(uint32_t &width, uint32_t &height, cv::Mat& outputBGRA)
{
	// Captures a live video frame from a source, renders it to a texture, transfers it onto
	//	a staging surface, maps it into CPU-accessible memory, then it wraps the pixel buffer into an OpenCV cv::Mat (BGRA format)

	if (!obs_source_enabled(OnnxInstance::instance().m_source))
		return false;

	obs_source_t *target = obs_filter_get_target(OnnxInstance::instance().m_source);

	if (!target)
		return false;

	width = obs_source_get_base_width(target);
	height = obs_source_get_base_height(target);

	if (width == 0 || height == 0)
		return false;

	gs_texrender_reset(OnnxInstance::instance().m_texrender);

	if (!gs_texrender_begin(OnnxInstance::instance().m_texrender, width, height))
		return false;

	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(OnnxInstance::instance().m_texrender);

	if (OnnxInstance::instance().m_stagesurface)
	{
		uint32_t stagesurf_width = gs_stagesurface_get_width(OnnxInstance::instance().m_stagesurface);
		uint32_t stagesurf_height = gs_stagesurface_get_height(OnnxInstance::instance().m_stagesurface);

		if (stagesurf_width != width || stagesurf_height != height)
		{
			gs_stagesurface_destroy(OnnxInstance::instance().m_stagesurface);
			OnnxInstance::instance().m_stagesurface = nullptr;
		}
	}

	if (!OnnxInstance::instance().m_stagesurface)
		OnnxInstance::instance().m_stagesurface = gs_stagesurface_create(width, height, GS_BGRA);

	gs_stage_texture(OnnxInstance::instance().m_stagesurface, gs_texrender_get_texture(OnnxInstance::instance().m_texrender));

	uint8_t *video_data;
	uint32_t linesize;

	if (!gs_stagesurface_map(OnnxInstance::instance().m_stagesurface, &video_data, &linesize))
		return false;

	outputBGRA = cv::Mat(height, width, CV_8UC4, video_data, linesize);

	gs_stagesurface_unmap(OnnxInstance::instance().m_stagesurface);
	return true;
}

/*static*/
gs_texture_t* BgBlurGraphics::blurBackground(uint32_t width, uint32_t height, gs_texture_t *alphaTexture)
{
	if (OnnxInstance::instance().m_blurBackground == 0 || !OnnxInstance::instance().m_kawaseBlurEffect)
		return nullptr;

	gs_texture_t *blurredTexture = gs_texture_create(width, height, GS_BGRA, 1, nullptr, 0);
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(OnnxInstance::instance().m_texrender));
	gs_eparam_t *image = gs_effect_get_param_by_name(OnnxInstance::instance().m_kawaseBlurEffect, "image");
	gs_eparam_t *focalmask = gs_effect_get_param_by_name(OnnxInstance::instance().m_kawaseBlurEffect, "focalmask");
	gs_eparam_t *xOffset = gs_effect_get_param_by_name(OnnxInstance::instance().m_kawaseBlurEffect, "xOffset");
	gs_eparam_t *yOffset = gs_effect_get_param_by_name(OnnxInstance::instance().m_kawaseBlurEffect, "yOffset");
	gs_eparam_t *blurIter = gs_effect_get_param_by_name(OnnxInstance::instance().m_kawaseBlurEffect, "blurIter");
	gs_eparam_t *blurTotal = gs_effect_get_param_by_name(OnnxInstance::instance().m_kawaseBlurEffect, "blurTotal");
	gs_eparam_t *blurFocusPointParam = gs_effect_get_param_by_name(OnnxInstance::instance().m_kawaseBlurEffect, "blurFocusPoint");
	gs_eparam_t *blurFocusDepthParam = gs_effect_get_param_by_name(OnnxInstance::instance().m_kawaseBlurEffect, "blurFocusDepth");

	for (int i = 0; i < (int)OnnxInstance::instance().m_blurBackground; i++)
	{
		gs_texrender_reset(OnnxInstance::instance().m_texrender);

		if (!gs_texrender_begin(OnnxInstance::instance().m_texrender, width, height))
		{
			blog(LOG_INFO, "BgBlurGraphics::blurBackground - Could not open background blur texrender!");
			return blurredTexture;
		}

		gs_effect_set_texture(image, blurredTexture);
		gs_effect_set_texture(focalmask, alphaTexture);
		gs_effect_set_float(xOffset, ((float)i + 0.5f) / (float)width);
		gs_effect_set_float(yOffset, ((float)i + 0.5f) / (float)height);
		gs_effect_set_int(blurIter, i);
		gs_effect_set_int(blurTotal, (int)OnnxInstance::instance().m_blurBackground);
		gs_effect_set_float(blurFocusPointParam, OnnxInstance::instance().m_blurFocusPoint);
		gs_effect_set_float(blurFocusDepthParam, OnnxInstance::instance().m_blurFocusDepth);

		struct vec4 background;
		vec4_zero(&background);
		gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
		gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

		const char *blur_type = "Draw";

		while (gs_effect_loop(OnnxInstance::instance().m_kawaseBlurEffect, blur_type))
			gs_draw_sprite(blurredTexture, 0, width, height);
		
		gs_blend_state_pop();
		gs_texrender_end(OnnxInstance::instance().m_texrender);
		gs_copy_texture(blurredTexture, gs_texrender_get_texture(OnnxInstance::instance().m_texrender));
	}

	return blurredTexture;
}
