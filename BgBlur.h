#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <obs.h>
#include <obs-module.h>
#include <opencv2/core/types.hpp>

/*static*/
class BgBlur
{
public:
	static BgBlur& instance()
	{
		static BgBlur a;
		return a;
	}

	static void obs_activate(void* data);
	static void obs_destroy(void* data);
	static void obs_defaults(obs_data_t* settings);
	static void obs_update_settings(void* data, obs_data_t* settings);
	static void obs_deactivate(void* data);
	static void obs_video_tick(void* data, float seconds);
	static void obs_video_render(void* data, gs_effect_t* _effect);

	static void* obs_create(obs_data_t* settings, obs_source_t* source);

	static const char* obs_getname(void* unused);

	static obs_properties_t* obs_properties(void* data);

private:
	BgBlur();
	~BgBlur();

	struct BgBlurData
	{
		obs_source_t* source = nullptr;
		gs_texrender_t* texrender = nullptr;
		gs_stagesurf_t* stagesurface = nullptr;
		int64_t blurBackground = 10;
	};
};

class BgBlurGraphics
{
public:	
	static bool getRGBAFromStageSurface(gs_texrender_t* texrender, gs_stagesurf_t* stagesurface, obs_source_t* source, uint32_t& width, uint32_t& height, cv::Mat& outputBGRA);
	static gs_texture_t *blurBackground(gs_texrender_t* texrender, gs_effect_t* kawaseBlurEffect, uint32_t amount, uint32_t width, uint32_t height, gs_texture_t *alphaTexture);
};
