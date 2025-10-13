#include "ModifyAppearence.h"
#include "OnnxModel.h"
#include "OnnxInstance.h"

#include <util\platform.h>

#include <filesystem>
#include <wchar.h>
#include <windows.h>
#include <algorithm>

#include <obs-module.h>
#include <graphics/matrix4.h>

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

	ModData* modData = (ModData*)data;
	auto onnxInstance = Onnx::instance().get(modData->source);

	obs_enter_graphics();
	gs_texrender_destroy(modData->texrender);

	if (modData->stagesurface)
		gs_stagesurface_destroy(modData->stagesurface);

	for (auto &itr : modData->cat)
		gs_effect_destroy(itr.maskEffect);

	obs_leave_graphics();

	// Cleanup
	Onnx::instance().unregisterDeIncrementSource(modData->source);
	delete modData;
}

/*static*/
obs_properties_t* ModifyAppearence::obs_properties(void* data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t* props = obs_properties_create();

	// Category selector
	obs_property_t* segmentation_category = obs_properties_add_list(props, "segmentation_category", "Segmentation category", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(segmentation_category, "Hair", "hair");
	obs_property_list_add_string(segmentation_category, "Clothing", "clothing");
	obs_property_list_add_string(segmentation_category, "Body", "body-skin");
	obs_property_list_add_string(segmentation_category, "Face", "face-skin");

	// Per-category groups
	add_category_controls(props, "grp_hair", "Hair settings", "hair");
	add_category_controls(props, "grp_clothing", "Clothing settings", "clothing");
	add_category_controls(props, "grp_face", "Face settings", "face_skin");
	add_category_controls(props, "grp_body", "Body settings", "body_skin");

	// Show only the relevant group when the dropdown changes
	obs_property_set_modified_callback(segmentation_category, show_only_selected_group);

	// Initialize visibility for first open (assume hair default)
	obs_data_t* fake = obs_data_create();
	obs_data_set_string(fake, "segmentation_category", "hair");
	show_only_selected_group(props, segmentation_category, fake);
	obs_data_release(fake);

	return props;
}

/*static*/
void ModifyAppearence::obs_update_settings(void* data, obs_data_t* settings)
{
	ModData* modData = (ModData* )data;

	read_cat(settings, "hair", modData->cat[OnnxModel::CATEGORY_HAIR]);
	read_cat(settings, "body_skin", modData->cat[OnnxModel::CATEGORY_BODY_SKIN]);
	read_cat(settings, "face_skin", modData->cat[OnnxModel::CATEGORY_FACE_SKIN]);
	read_cat(settings, "clothing", modData->cat[OnnxModel::CATEGORY_CLOTHES]);

	// optional if you keep this setting
	modData->temporalSmooth = (float)obs_data_get_double(settings, "temporal_smooth_factor");
}

/*static*/
void ModifyAppearence::obs_activate(void* data)
{

}

/*static*/
void ModifyAppearence::obs_deactivate(void* data)
{

}

/*static*/
void ModifyAppearence::obs_video_tick(void* data, float seconds)
{
	UNUSED_PARAMETER(seconds);
	ModData* modData = (ModData* )data;
}

void ModifyAppearence::obs_video_render(void *data, gs_effect_t *_effect)
{
	UNUSED_PARAMETER(_effect);
	ModData *modData = (ModData *)data;

	if (!obs_source_enabled(modData->source))
		return;

	auto onnxInstance = Onnx::instance().get(modData->source);
	onnxInstance->update(modData->source, modData->texrender, modData->stagesurface, ModifyAppearence::instance().m_cats);

	obs_source_video_render(obs_filter_get_target(modData->source));

	const enum gs_color_space preferred_spaces[] = {
		GS_CS_SRGB,
		GS_CS_SRGB_16F,
		GS_CS_709_EXTENDED,
	};

	const enum gs_color_space source_space = obs_source_get_color_space(obs_filter_get_target(modData->source), OBS_COUNTOF(preferred_spaces), preferred_spaces);

	if (source_space == GS_CS_709_EXTENDED)
	{
		obs_source_skip_video_filter(modData->source);
		return;
	}

	const enum gs_color_format format = gs_get_format_from_space(source_space);
	
	// Begin the OBS filter render ONCE
	if (!obs_source_process_filter_begin_with_color_space(modData->source, format, source_space, OBS_ALLOW_DIRECT_RENDERING))
		return;

	for (auto& cat : ModifyAppearence::instance().m_cats)
	{
		gs_blend_state_push();
		gs_reset_blend_state();

		auto catData = modData->cat[cat];
		cv::Mat cvMask = onnxInstance->m_lastFullMask[cat].clone();
		gs_texture_t *alphaTexture = gs_texture_create(cvMask.cols, cvMask.rows, GS_R8, 1, (const uint8_t **)&cvMask.data, 0);

		auto effect = modData->cat[cat].maskEffect;
		gs_effect_set_texture(gs_effect_get_param_by_name(effect, "alphamask"), alphaTexture);
		gs_effect_set_float(gs_effect_get_param_by_name(effect, "gamma"), catData.gamma);
		gs_effect_set_float(gs_effect_get_param_by_name(effect, "contrast"), catData.contrast);
		gs_effect_set_float(gs_effect_get_param_by_name(effect, "saturation"), catData.saturation);
		gs_effect_set_float(gs_effect_get_param_by_name(effect, "smooth"), catData.smooth);
		
		obs_source_process_filter_tech_end(modData->source, effect, 0, 0, "DrawWithoutBlur");

		gs_texture_destroy(alphaTexture);

		gs_blend_state_pop();
		obs_source_process_filter_end(modData->source, effect, 0, 0);
	}
}

/*static*/
bool ModifyAppearence::show_only_selected_group(obs_properties_t* props, obs_property_t* list, obs_data_t* settings)
{
	const char* sel = obs_data_get_string(settings, "segmentation_category");
	bool hair = strcmp(sel, "hair") == 0;
	bool body = strcmp(sel, "body-skin") == 0;
	bool face = strcmp(sel, "face-skin") == 0;
	bool clothing = strcmp(sel, "clothing") == 0;

	obs_property_t* grp_hair = obs_properties_get(props, "grp_hair");
	obs_property_t *grp_body = obs_properties_get(props, "grp_body");
	obs_property_t *grp_face = obs_properties_get(props, "grp_face");
	obs_property_t *grp_clothing = obs_properties_get(props, "grp_clothing");

	if (grp_hair)
		obs_property_set_visible(grp_hair, hair);

	if (grp_body)
		obs_property_set_visible(grp_body, body);

	if (grp_face)
		obs_property_set_visible(grp_face, face);

	if (grp_clothing)
		obs_property_set_visible(grp_clothing, clothing);

	return true;
}

/*static*/
void ModifyAppearence::obs_defaults(obs_data_t *settings)
{
	auto applCatDefaults = [&](const std::string& suffix)
	{
		obs_data_set_default_double(settings, (std::string("gamma_") + suffix).c_str(), 0.0);
		obs_data_set_default_double(settings, (std::string("contrast_") + suffix).c_str(), 0.0);
		obs_data_set_default_double(settings, (std::string("saturation_") + suffix).c_str(), 1.0);
		obs_data_set_default_double(settings, (std::string("smooth_") + suffix).c_str(), 0.0);
	};

	applCatDefaults("hair");
	applCatDefaults("body_skin");
	applCatDefaults("face_skin");
	applCatDefaults("clothing");
}

/*static*/
void ModifyAppearence::add_category_controls(obs_properties_t* props, const char* group_name, const char* group_label, const char* suffix)
{
	obs_properties_t* grp = obs_properties_create();
	obs_properties_add_group(props, group_name, group_label, OBS_GROUP_NORMAL, grp);

	// Key format: <setting>_<suffix> 
	obs_properties_add_float_slider(grp, (std::string("smooth_") + suffix).c_str(), "Smoothing", 0.0, 100.0, 1.0);  
	obs_properties_add_float_slider(grp, (std::string("gamma_") + suffix).c_str(), "Gamma", 0.0, 0.5, 0.01);                  
	obs_properties_add_float_slider(grp, (std::string("contrast_") + suffix).c_str(), "Contrast", 0.0, 1.0, 0.01);            
	obs_properties_add_float_slider(grp, (std::string("saturation_") + suffix).c_str(), "Saturation", 0.00, 3.0, 0.01);       
}

/*static*/
void ModifyAppearence::read_cat(obs_data_t* s, const char* suf, CategorySettings& out)
{
	obs_enter_graphics();
	gs_effect_destroy(out.maskEffect);
	out.maskEffect = gs_effect_create_from_file((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / "mask_color_correction.effect").string().c_str(), NULL);
	printf("modData->maskEffect = %d\n", (int)(uint64_t)out.maskEffect);
	obs_leave_graphics();

	float gamma = (float)obs_data_get_double(s, (std::string("gamma_") + suf).c_str());
	float contrast = (float)obs_data_get_double(s, (std::string("contrast_") + suf).c_str());
	out.saturation = (float)obs_data_get_double(s, (std::string("saturation_") + suf).c_str());
	out.smooth = (float)obs_data_get_double(s, (std::string("smooth_") + suf).c_str());

	out.gamma = (gamma < 0.0) ? (-gamma + 1.0f) : (1.0f / (gamma + 1.0f));
	out.contrast = (contrast < 0.0f) ? (1.0f / (-contrast + 1.0f)) : (contrast + 1.0f);
}

/*static*/
gs_color_space ModifyAppearence::obs_video_get_color_space(void *data, size_t count, const enum gs_color_space *preferred_spaces)
{
	UNUSED_PARAMETER(count);
	UNUSED_PARAMETER(preferred_spaces);

	const enum gs_color_space potential_spaces[] = {
		GS_CS_SRGB,
		GS_CS_SRGB_16F,
		GS_CS_709_EXTENDED,
	};

	ModData *modData = (ModData *)data;
	const enum gs_color_space source_space = obs_source_get_color_space(obs_filter_get_target(modData->source), OBS_COUNTOF(potential_spaces), potential_spaces);
	return source_space;
}
