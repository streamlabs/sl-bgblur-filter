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
const char *BgBlur::obs_getname(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Background Removal";
}

/*static*/
void *BgBlur::obs_create(obs_data_t *settings, obs_source_t *source)
{
	blog(LOG_INFO, "BgBlur::create");

	AllocConsole();
	freopen("conin$", "r", stdin);
	freopen("conout$", "w", stdout);
	freopen("conout$", "w", stderr);

	auto modelFilepath = (std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / L"SelfieMulticlass.onnx");
	OnnxInstance::instance().init(source, modelFilepath.wstring());

	obs_update_settings((void *)source, settings);
	return (void *)source;
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
	obs_source_t* source = (obs_source_t*)data;

	if (!obs_source_enabled(source))
		return;

	/***
	* Build mask
	*/

	if (!OnnxInstance::instance().update())
	{
		obs_source_skip_video_filter(source);
		return;
	}

	/***
	* Rendering
	*/

	auto &backgroundMask = OnnxInstance::instance().m_lastFullMask[OnnxModel::CATEGORY_BACKGROUND_INVERSE];

	gs_texture_t *alphaTexture = gs_texture_create(backgroundMask.cols, backgroundMask.rows, GS_R8, 1, (const uint8_t **)&backgroundMask.data, 0);
	gs_texture_t *blurredTexture = BgBlurGraphics::blurBackground(OnnxInstance::instance().m_maskWidth, OnnxInstance::instance().m_maskHeight, alphaTexture);

	if (!obs_source_process_filter_begin(OnnxInstance::instance().m_source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING))
	{
		gs_texture_destroy(alphaTexture);
		gs_texture_destroy(blurredTexture);
		return;
	}

	gs_eparam_t *alphamask = gs_effect_get_param_by_name(OnnxInstance::instance().m_maskEffect, "alphamask");
	gs_eparam_t *blurredBackground = gs_effect_get_param_by_name(OnnxInstance::instance().m_maskEffect, "blurredBackground");
	gs_effect_set_texture(alphamask, alphaTexture);

	if (OnnxInstance::instance().m_blurBackground > 0)
		gs_effect_set_texture(blurredBackground, blurredTexture);

	gs_blend_state_push();
	gs_reset_blend_state();

	const char *techName;
	if (OnnxInstance::instance().m_blurBackground > 0)
		techName = "DrawWithBlur";
	else
		techName = "DrawWithoutBlur";

	obs_source_process_filter_tech_end(source, OnnxInstance::instance().m_maskEffect, 0, 0, techName);

	gs_blend_state_pop();
	gs_texture_destroy(alphaTexture);
	gs_texture_destroy(blurredTexture);
}

/*static*/
void BgBlur::obs_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "blur_background", 10);
	obs_data_set_default_double(settings, "smooth_contour", 1.0);
	obs_data_set_default_double(settings, "temporal_smooth_factor", 0.5);
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
	OnnxInstance::instance().m_isDisabled = true;

	OnnxInstance::instance().m_blurBackground = obs_data_get_int(settings, "blur_background");
	OnnxInstance::instance().m_smoothContour = (float)obs_data_get_double(settings, "smooth_contour");
	OnnxInstance::instance().m_temporalSmoothFactor = (float)obs_data_get_double(settings, "temporal_smooth_factor");

	obs_enter_graphics();

	gs_effect_destroy(OnnxInstance::instance().m_maskEffect);
	OnnxInstance::instance().m_maskEffect = gs_effect_create_from_file((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / MASK_EFFECT_PATH).string().c_str(), NULL);

	gs_effect_destroy(OnnxInstance::instance().m_kawaseBlurEffect);
	OnnxInstance::instance().m_kawaseBlurEffect = gs_effect_create_from_file((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / KAWASE_BLUR_EFFECT_PATH).string().c_str(), NULL);

	obs_leave_graphics();

	// enable
	OnnxInstance::instance().m_isDisabled = false;
}

/*static*/
void BgBlur::obs_activate(void *data)
{
	OnnxInstance::instance().m_isDisabled = false;
}

/*static*/
void BgBlur::obs_destroy(void *data)
{
	blog(LOG_INFO, "BgBlur::destroy");

	OnnxInstance::instance().m_isDisabled = true;

	obs_enter_graphics();
	gs_texrender_destroy(OnnxInstance::instance().m_texrender);

	if (OnnxInstance::instance().m_stagesurface)
		gs_stagesurface_destroy(OnnxInstance::instance().m_stagesurface);

	gs_effect_destroy(OnnxInstance::instance().m_maskEffect);
	gs_effect_destroy(OnnxInstance::instance().m_kawaseBlurEffect);
	obs_leave_graphics();
}

/*static*/
void BgBlur::obs_deactivate(void *data)
{
	OnnxInstance::instance().m_isDisabled = true;
}
