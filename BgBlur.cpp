#include "BgBlur.h"
#include "OnnxModel.h"
#include "FilterData.h"

#include <util\platform.h>

#include <filesystem>
#include <wchar.h>
#include <windows.h>
#include <algorithm>

BgBlur::BgBlur()
{

}

BgBlur::~BgBlur()
{

}

/*static*/
const char *BgBlur::obs_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Background Removal";
}

/*static*/
void *BgBlur::obs_create(obs_data_t *settings, obs_source_t *source)
{
	blog(LOG_INFO, "BgBlur::create");

	auto modelFilepath = (std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / L"SelfieMulticlass.onnx");

	FilterData *filterD = new FilterData;
	filterD->source = source;
	filterD->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	filterD->model = std::make_unique<OnnxModel>(modelFilepath.wstring().c_str());

	if (!filterD->model->isGood())
	{
		blog(LOG_ERROR, "OnnxModel failed to init properly.");
		delete filterD;
		return nullptr;
	}

	obs_update_settings(filterD, settings);
	return (void *)filterD;
}

/*static*/
void BgBlur::obs_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(data);
	UNUSED_PARAMETER(seconds);
}

/*static*/
void BgBlur::obs_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);
	FilterData *filterD = (FilterData *)data;

	if (filterD->isDisabled || !filterD->source || !obs_source_enabled(filterD->source))
		return;

	uint32_t width = 0, height = 0;

	if (!BgBlurGraphics::getRGBAFromStageSurface(filterD, width, height) || !filterD->maskEffect)
		return;

	/***
	* Build mask
	*/

	bool boolQueryOnnx = true;
	cv::Mat fullBGRA = filterD->inputBGRA.clone();

	// 10fps is fine because we use temporal smoothing
	if (boolQueryOnnx && ::clock() - filterD->lastModelRun < 50)
		boolQueryOnnx = false;
		
	cv::Mat backgroundMask;

	if (boolQueryOnnx)
	{
		filterD->model->runImage(fullBGRA, cv::COLOR_BGRA2RGB, filterD->lastOnnxOutput);
		filterD->lastModelRun = ::clock();
	}

	auto bgRef = filterD->lastOnnxOutput[OnnxModel::CATEGORY_BACKGROUND_INVERSE].clone();

	if (bgRef.empty())
		return;

	if (filterD->temporalSmoothFactor <= 0 || filterD->lastSmallBackgroundMask.empty() || filterD->lastSmallBackgroundMask.size() != bgRef.size() || filterD->lastSmallBackgroundMask.type() != bgRef.type())
	{
		filterD->lastSmallBackgroundMask = bgRef.clone();
	}
	else if (filterD->temporalSmoothFactor > 0)
	{
		const double f = std::clamp(filterD->temporalSmoothFactor, 0.0f, 1.0f);
		cv::addWeighted(filterD->lastSmallBackgroundMask, f, bgRef, 1.0 - f, 0.0, filterD->lastSmallBackgroundMask);

	}

	backgroundMask = filterD->lastSmallBackgroundMask.clone();

	if (filterD->smoothContour > 0.0)
	{
		int k = (int)(3 + 11 * filterD->smoothContour);
		if ((k & 1) == 0)
			++k;

		cv::stackBlur(backgroundMask, backgroundMask, cv::Size(k, k));
	}

	// Resize mask back to input image size
	cv::resize(backgroundMask, backgroundMask, fullBGRA.size());

	// If we smoothed, re-binarize
	if (filterD->smoothContour > 0.0)
		backgroundMask = backgroundMask > 128;

	filterD->lastFullBackgroundMask = backgroundMask;
	filterD->lastFullBGRA = fullBGRA.clone();

	/***
	* Rendering
	*/

	gs_texture_t *alphaTexture = gs_texture_create(backgroundMask.cols, backgroundMask.rows, GS_R8, 1, (const uint8_t **)&backgroundMask.data, 0);
	gs_texture_t *blurredTexture = BgBlurGraphics::blurBackground(filterD, width, height, alphaTexture);

	if (!obs_source_process_filter_begin(filterD->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING))
	{
		gs_texture_destroy(alphaTexture);
		gs_texture_destroy(blurredTexture);
		return;
	}

	gs_eparam_t *alphamask = gs_effect_get_param_by_name(filterD->maskEffect, "alphamask");
	gs_eparam_t *blurredBackground = gs_effect_get_param_by_name(filterD->maskEffect, "blurredBackground");
	gs_effect_set_texture(alphamask, alphaTexture);

	if (filterD->blurBackground > 0)
		gs_effect_set_texture(blurredBackground, blurredTexture);

	gs_blend_state_push();
	gs_reset_blend_state();

	const char *techName;
	if (filterD->blurBackground > 0)
		techName = "DrawWithBlur";
	else
		techName = "DrawWithoutBlur";

	obs_source_process_filter_tech_end(filterD->source, filterD->maskEffect, 0, 0, techName);

	gs_blend_state_pop();
	gs_texture_destroy(alphaTexture);
	gs_texture_destroy(blurredTexture);
}

/*static*/
void BgBlur::obs_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "blur_background", 10);
	obs_data_set_default_double(settings, "smooth_contour", 1.0);
	obs_data_set_default_double(settings, "temporal_smooth_factor", 0.35);
}

/*static*/
obs_properties_t *BgBlur::obs_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();

	obs_properties_add_int_slider(props, "blur_background", "Blur Amount", 0, 20, 1);
	obs_properties_add_float_slider(props, "smooth_contour", "Smooth", 0.0, 1.0, 0.01);
	obs_properties_add_float_slider(props, "temporal_smooth_factor", "Removal Smoothing", 0.0, 0.90, 0.01);

	return props;
}

/*static*/
void BgBlur::obs_update_settings(void *data, obs_data_t *settings)
{
	FilterData *filterD = (FilterData*)(data);

	filterD->isDisabled = true;

	filterD->blurBackground = obs_data_get_int(settings, "blur_background");
	filterD->smoothContour = (float)obs_data_get_double(settings, "smooth_contour");
	filterD->temporalSmoothFactor = (float)obs_data_get_double(settings, "temporal_smooth_factor");

	obs_enter_graphics();

	gs_effect_destroy(filterD->maskEffect);
	filterD->maskEffect = gs_effect_create_from_file((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / MASK_EFFECT_PATH).string().c_str(), NULL);

	gs_effect_destroy(filterD->kawaseBlurEffect);
	filterD->kawaseBlurEffect = gs_effect_create_from_file((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / KAWASE_BLUR_EFFECT_PATH).string().c_str(), NULL);

	obs_leave_graphics();

	// enable
	filterD->isDisabled = false;
}

/*static*/
void BgBlur::obs_activate(void *data)
{
	FilterData* filterD = (FilterData*)data;
	filterD->isDisabled = false;
}

/*static*/
void BgBlur::obs_destroy(void *data)
{
	blog(LOG_INFO, "BgBlur::destroy");

	if (FilterData* filterD = (FilterData *)data)
	{
		filterD->isDisabled = true;

		obs_enter_graphics();
		gs_texrender_destroy(filterD->texrender);

		if (filterD->stagesurface)
			gs_stagesurface_destroy(filterD->stagesurface);
		
		gs_effect_destroy(filterD->maskEffect);
		gs_effect_destroy(filterD->kawaseBlurEffect);
		obs_leave_graphics();

		delete filterD;
	}
}

/*static*/
void BgBlur::obs_deactivate(void *data)
{
	FilterData *filterD = (FilterData *)data;
	filterD->isDisabled = true;
}


