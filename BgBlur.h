#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif


#include <obs.h>
#include <obs-module.h>

#include <opencv2/core/types.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <dml_provider_factory.h>

struct FilterData;

/*static*/
class BgBlur
{
public:
	static BgBlur& instance()
	{
		static BgBlur a;
		return a;
	}

	static void obs_activate(void *data);
	static void obs_destroy(void *data);
	static void obs_defaults(obs_data_t *settings);
	static void obs_update_settings(void *data, obs_data_t *settings);
	static void obs_deactivate(void *data);
	static void obs_video_tick(void *data, float seconds);
	static void obs_video_render(void *data, gs_effect_t *_effect);

	static void* obs_create(obs_data_t *settings, obs_source_t *source);

	static const char* obs_getname(void *unused);

	static obs_properties_t *obs_properties(void *data);

private:
	BgBlur();
	~BgBlur();

protected:
	#define OBS_BGREMOVAL_ORT_SESSION_ERROR_FILE_NOT_FOUND 1
	#define OBS_BGREMOVAL_ORT_SESSION_ERROR_INVALID_MODEL 2
	#define OBS_BGREMOVAL_ORT_SESSION_ERROR_INVALID_INPUT_OUTPUT 3
	#define OBS_BGREMOVAL_ORT_SESSION_ERROR_STARTUP 5
	#define OBS_BGREMOVAL_ORT_SESSION_SUCCESS 0
};

class BgBlurGraphics
{
public:
	static int createOrtSession(FilterData *tf);
	static bool runFilterModelInference(FilterData *tf, const cv::Mat &imageBGRA, cv::Mat &output);
	static bool getRGBAFromStageSurface(FilterData *tf, uint32_t &width, uint32_t &height);
	static gs_texture_t* blurBackground(FilterData *tf, uint32_t width, uint32_t height, gs_texture_t *alphaTexture);
};
