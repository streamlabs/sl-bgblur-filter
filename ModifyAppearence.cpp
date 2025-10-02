#include "ModifyAppearence.h"
#include "OnnxModel.h"
#include "OnnxInstance.h"

#include <util\platform.h>

#include <filesystem>
#include <wchar.h>
#include <windows.h>
#include <algorithm>

ModifyAppearence::ModifyAppearence()
{

}

ModifyAppearence::~ModifyAppearence()
{

}

/*static*/
const char* ModifyAppearence::obs_getname(void* unused)
{
	UNUSED_PARAMETER(unused);
	return "Modify Appearence";
}

/*static*/
void* ModifyAppearence::obs_create(obs_data_t* settings, obs_source_t* source)
{
	//debug
	AllocConsole();
	freopen("conin$", "r", stdin);
	freopen("conout$", "w", stdout);
	freopen("conout$", "w", stderr);

	blog(LOG_INFO, "ModifyAppearence::create");

	ModData* data = new ModData;
	data->source = source;
	data->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	Onnx::instance().registerIncrementSource(source);

	auto onnxInstance = Onnx::instance().get(source);
	onnxInstance->init((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / L"SelfieMulticlass.onnx").wstring());

	ModifyAppearence::obs_update_settings((void* )data, settings);
	return (void*)data;
}

/*static*/
void ModifyAppearence::obs_destroy(void* data)
{
	blog(LOG_INFO, "ModifyAppearence::destroy");

	ModData* blurData = (ModData*)data;
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
void ModifyAppearence::obs_video_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(seconds);
	ModData *blurData = (ModData *)data;
}

/*static*/
void ModifyAppearence::obs_video_render(void* data, gs_effect_t* _effect)
{
	UNUSED_PARAMETER(_effect);
	ModData* blurData = (ModData*)data;	

	if (!obs_source_enabled(blurData->source))
		return;

	/***
	* Build mask
	*/

	auto onnxInstance = Onnx::instance().get(blurData->source);
	onnxInstance->update(blurData->source, blurData->texrender, blurData->stagesurface, OnnxModel::CATEGORY_BACKGROUND_INVERSE);

	/***
	* Rendering
	*/

	;
}

/*static*/
void ModifyAppearence::obs_defaults(obs_data_t* settings)
{
	obs_data_set_default_double(settings, "temporal_smooth_factor", 0.5);
}

/*static*/
obs_properties_t* ModifyAppearence::obs_properties(void* data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t* props = obs_properties_create();

	obs_properties_add_int_slider(props, "blur_background", "Blur Amount", 0, 20, 1);
	obs_properties_add_float_slider(props, "temporal_smooth_factor", "Motion Smoothing", 0.0, 0.90, 0.01);

	return props;
}

/*static*/
void ModifyAppearence::obs_update_settings(void* data, obs_data_t* settings)
{
	ModData* blurData = (ModData*)data;
	auto onnxInstance = Onnx::instance().get(blurData->source);

	onnxInstance->m_temporalSmoothFactor = (float)obs_data_get_double(settings, "temporal_smooth_factor");

}

/*static*/
void ModifyAppearence::obs_activate(void* data)
{

}

/*static*/
void ModifyAppearence::obs_deactivate(void* data)
{

}
