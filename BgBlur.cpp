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
void BgBlur::obs_destroy(void *data)
{
	blog(LOG_INFO, "BgBlur::destroy");

	BgBlurData *blurData = (BgBlurData *)data;
	auto onnxInstance = Onnx::instance().get(blurData->source);

	obs_enter_graphics();
	gs_texrender_destroy(blurData->texrender);

	if (blurData->stagesurface)
		gs_stagesurface_destroy(blurData->stagesurface);

	gs_effect_destroy(blurData->maskEffect);
	gs_effect_destroy(blurData->kawaseBlurEffect);

	obs_leave_graphics();

	// Cleanup
	Onnx::instance().unregisterDeIncrementSource(blurData->source);
	delete blurData;
}

/*static*/
void BgBlur::obs_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);
	BgBlurData* blurData = (BgBlurData*)data;
}

/*static*/
void BgBlur::obs_video_render(void* data, gs_effect_t* _effect)
{
	UNUSED_PARAMETER(_effect);
	BgBlurData* blurData = (BgBlurData*)data;	

	if (!obs_source_enabled(blurData->source))
		return;

	/***
	* Build mask
	*/

	auto onnxInstance = Onnx::instance().get(blurData->source);
	onnxInstance->update(blurData->source, blurData->texrender, blurData->stagesurface, { OnnxModel::CATEGORY_BACKGROUND_INVERSE });

	/***
	* Rendering
	*/

	cv::Mat& backgroundMask = onnxInstance->m_lastFullMask[OnnxModel::CATEGORY_BACKGROUND_INVERSE];

	if (backgroundMask.empty())
	{
		obs_source_skip_video_filter(blurData->source);
		return;
	}

	gs_texture_t* alphaTexture = gs_texture_create(backgroundMask.cols, backgroundMask.rows, GS_R8, 1, (const uint8_t**)&backgroundMask.data, 0);
	gs_texture_t* blurredTexture = blurBackground(blurData->texrender, blurData->kawaseBlurEffect, blurData->blurBackground,
		onnxInstance->m_maskWidth, onnxInstance->m_maskHeight, alphaTexture);

	if (!obs_source_process_filter_begin(blurData->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING))
	{
		gs_texture_destroy(alphaTexture);
		gs_texture_destroy(blurredTexture);
		return;
	}

	gs_eparam_t* alphamask = gs_effect_get_param_by_name(blurData->maskEffect, "alphamask");
	gs_eparam_t* blurredBackground = gs_effect_get_param_by_name(blurData->maskEffect, "blurredBackground");
	gs_effect_set_texture(alphamask, alphaTexture);

	if (blurData->blurBackground > 0)
		gs_effect_set_texture(blurredBackground, blurredTexture);

	gs_blend_state_push();
	gs_reset_blend_state();

	obs_source_process_filter_tech_end(blurData->source, blurData->maskEffect, 0, 0, blurData->blurBackground > 0 ? "DrawWithBlur" : "DrawWithoutBlur");

	gs_blend_state_pop();
	gs_texture_destroy(alphaTexture);
	gs_texture_destroy(blurredTexture);
}

/*static*/
void BgBlur::obs_defaults(obs_data_t* settings)
{
	obs_data_set_default_int(settings, "blur_background", 10);
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

	gs_effect_destroy(blurData->maskEffect);
	blurData->maskEffect = gs_effect_create_from_file((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / MASK_EFFECT_PATH).string().c_str(), NULL);

	gs_effect_destroy(blurData->kawaseBlurEffect);
	blurData->kawaseBlurEffect = gs_effect_create_from_file((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / KAWASE_BLUR_EFFECT_PATH).string().c_str(), NULL);

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

/*static*/
gs_texture_t* BgBlur::blurBackground(gs_texrender_t* texrender, gs_effect_t* kawaseBlurEffect, uint32_t amount, uint32_t width, uint32_t height, gs_texture_t* alphaTexture)
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
