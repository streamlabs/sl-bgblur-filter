#include "BgBlur.h"
#include "OnnxModel.h"
#include "OnnxInstance.h"

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
const char* BgBlur::obs_getname(void* unused)
{
	UNUSED_PARAMETER(unused);
	return "Background Removal";
}

/*static*/
void* BgBlur::obs_create(obs_data_t* settings, obs_source_t* source)
{
	//debug
	AllocConsole();
	freopen("conin$", "r", stdin);
	freopen("conout$", "w", stdout);
	freopen("conout$", "w", stderr);

	blog(LOG_INFO, "BgBlur::create");

	BgBlurData* data = new BgBlurData;
	data->source = source;
	data->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	Onnx::instance().registerIncrementSource(source);

	auto onnxInstance = Onnx::instance().get(source);
	onnxInstance->init((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / L"SelfieMulticlass.onnx").wstring());

	BgBlur::obs_update_settings((void *)data, settings);
	return (void*)data;
}

/*static*/
void BgBlur::obs_video_tick(void* data, float seconds)
{
	UNUSED_PARAMETER(seconds);
	BgBlurData *blurData = (BgBlurData *)data;
	
	/***
	* Build mask
	*/

	auto onnxInstance = Onnx::instance().get(blurData->source);
	onnxInstance->update(blurData->source, blurData->texrender, blurData->stagesurface, OnnxModel::CATEGORY_BACKGROUND_INVERSE);
}

/*static*/
void BgBlur::obs_destroy(void *data)
{
	blog(LOG_INFO, "BgBlur::destroy");

	BgBlurData *blurData = (BgBlurData *)data;
	auto onnxInstance = Onnx::instance().get(blurData->source);

	obs_enter_graphics();
	gs_texrender_destroy(blurData->texrender);

	if (blurData->stagesurface)
		gs_stagesurface_destroy(blurData->stagesurface);

	gs_effect_destroy(onnxInstance->m_maskEffect);
	gs_effect_destroy(onnxInstance->m_kawaseBlurEffect);

	obs_leave_graphics();

	// Cleanup
	Onnx::instance().unregisterDeIncrementSource(blurData->source);
	delete blurData;
}

/*static*/
void BgBlur::obs_video_render(void* data, gs_effect_t* _effect)
{
	UNUSED_PARAMETER(_effect);
	BgBlurData* blurData = (BgBlurData*)data;	

	if (!obs_source_enabled(blurData->source))
		return;

	/***
	* Rendering
	*/

	auto onnxInstance = Onnx::instance().get(blurData->source);
	auto &backgroundMask = onnxInstance->m_lastFullMask[OnnxModel::CATEGORY_BACKGROUND_INVERSE];

	gs_texture_t* alphaTexture = gs_texture_create(backgroundMask.cols, backgroundMask.rows, GS_R8, 1, (const uint8_t**)&backgroundMask.data, 0);
	gs_texture_t *blurredTexture = BgBlurGraphics::blurBackground(blurData->texrender, onnxInstance->m_kawaseBlurEffect, blurData->blurBackground,
		onnxInstance->m_maskWidth, onnxInstance->m_maskHeight, alphaTexture);

	if (!obs_source_process_filter_begin(blurData->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING))
	{
		gs_texture_destroy(alphaTexture);
		gs_texture_destroy(blurredTexture);
		return;
	}

	gs_eparam_t *alphamask = gs_effect_get_param_by_name(onnxInstance->m_maskEffect, "alphamask");
	gs_eparam_t *blurredBackground = gs_effect_get_param_by_name(onnxInstance->m_maskEffect, "blurredBackground");
	gs_effect_set_texture(alphamask, alphaTexture);

	if (blurData->blurBackground > 0)
		gs_effect_set_texture(blurredBackground, blurredTexture);

	gs_blend_state_push();
	gs_reset_blend_state();

	obs_source_process_filter_tech_end(blurData->source, onnxInstance->m_maskEffect, 0, 0, blurData->blurBackground > 0 ? "DrawWithBlur" : "DrawWithoutBlur");

	gs_blend_state_pop();
	gs_texture_destroy(alphaTexture);
	gs_texture_destroy(blurredTexture);
}

/*static*/
void BgBlur::obs_defaults(obs_data_t* settings)
{
	obs_data_set_default_int(settings, "blur_background", 10);
	obs_data_set_default_double(settings, "smooth_contour", 1.0);
	obs_data_set_default_double(settings, "temporal_smooth_factor", 0.5);
}

/*static*/
obs_properties_t* BgBlur::obs_properties(void* data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t* props = obs_properties_create();

	obs_properties_add_int_slider(props, "blur_background", "Blur Amount", 0, 20, 1);
	obs_properties_add_float_slider(props, "temporal_smooth_factor", "Motion Smoothing", 0.0, 0.90, 0.01);

	return props;
}

/*static*/
void BgBlur::obs_update_settings(void* data, obs_data_t* settings)
{
	BgBlurData *blurData = (BgBlurData *)data;
	auto onnxInstance = Onnx::instance().get(blurData->source);

	blurData->blurBackground = (uint32_t)obs_data_get_int(settings, "blur_background");
	onnxInstance->m_temporalSmoothFactor = (float)obs_data_get_double(settings, "temporal_smooth_factor");

	obs_enter_graphics();

	gs_effect_destroy(onnxInstance->m_maskEffect);
	onnxInstance->m_maskEffect = gs_effect_create_from_file((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / MASK_EFFECT_PATH).string().c_str(), NULL);

	gs_effect_destroy(onnxInstance->m_kawaseBlurEffect);
	onnxInstance->m_kawaseBlurEffect = gs_effect_create_from_file((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / KAWASE_BLUR_EFFECT_PATH).string().c_str(), NULL);

	obs_leave_graphics();
}

/*static*/
void BgBlur::obs_activate(void* data)
{

}

/*static*/
void BgBlur::obs_deactivate(void* data)
{

}
