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

	ModData* modData = (ModData*)data;
	auto onnxInstance = Onnx::instance().get(modData->source);

	obs_enter_graphics();
	gs_texrender_destroy(modData->texrender);

	if (modData->stagesurface)
		gs_stagesurface_destroy(modData->stagesurface);

	gs_effect_destroy(modData->maskEffect);

	obs_leave_graphics();

	// Cleanup
	Onnx::instance().unregisterDeIncrementSource(modData->source);
	delete modData;
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

	// Category selector
	obs_property_t* segmentation_category = obs_properties_add_list(props, "segmentation_category", "Segmentation category", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(segmentation_category, "Hair", "hair");
	obs_property_list_add_string(segmentation_category, "Body Skin", "body-skin");
	obs_property_list_add_string(segmentation_category, "Face Skin", "face-skin");

	// Per-category groups
	add_category_controls(props, "grp_hair", "Hair settings", "hair");
	add_category_controls(props, "grp_body", "Body Skin settings", "body_skin");
	add_category_controls(props, "grp_face", "Face Skin settings", "face_skin");

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

	obs_enter_graphics();

	gs_effect_destroy(modData->maskEffect);
	modData->maskEffect = gs_effect_create_from_file((std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / MASK_EFFECT_PATH).string().c_str(), NULL);

	obs_leave_graphics();

	read_cat(settings, "hair", modData->cat[OnnxModel::CATEGORY_HAIR]);
	read_cat(settings, "body_skin", modData->cat[OnnxModel::CATEGORY_BODY_SKIN]);
	read_cat(settings, "face_skin", modData->cat[OnnxModel::CATEGORY_FACE_SKIN]);

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

/*static*/
void ModifyAppearence::obs_video_render(void* data, gs_effect_t* _effect)
{
	UNUSED_PARAMETER(_effect);
	ModData* modData = (ModData* )data;

	if (!obs_source_enabled(modData->source))
		return;

	if (!obs_source_process_filter_begin(modData->source, GS_RGBA, OBS_ALLOW_DIRECT_RENDERING))
		return;

	gs_blend_state_push();
	gs_reset_blend_state();

	auto onnxInstance = Onnx::instance().get(modData->source);
	onnxInstance->update(modData->source, modData->texrender, modData->stagesurface, ModifyAppearence::instance().m_cats);

	for (auto &cat : ModifyAppearence::instance().m_cats)
	{
		cv::Mat &cvMask = onnxInstance->m_lastFullMask[cat];
		if (cvMask.empty())
			continue;

		gs_texture_t* alphaTexture = gs_texture_create(cvMask.cols, cvMask.rows, GS_R8, 1, (const uint8_t **)&cvMask.data, 0);

		gs_eparam_t* alphamask = gs_effect_get_param_by_name(modData->maskEffect, "alphamask");
		gs_effect_set_texture(alphamask, alphaTexture);

		obs_source_process_filter_tech_end(modData->source, modData->maskEffect, 0, 0, "DrawWithoutBlur");

		gs_texture_destroy(alphaTexture);
	}

	gs_blend_state_pop();
	obs_source_process_filter_end(modData->source, modData->maskEffect, 0, 0);
}

/*static*/
bool ModifyAppearence::show_only_selected_group(obs_properties_t* props, obs_property_t* list, obs_data_t* settings)
{
	const char* sel = obs_data_get_string(settings, "segmentation_category");
	bool hair = strcmp(sel, "hair") == 0;
	bool body = strcmp(sel, "body-skin") == 0;
	bool face = strcmp(sel, "face-skin") == 0;

	obs_property_t* grp_hair = obs_properties_get(props, "grp_hair");
	obs_property_t* grp_body = obs_properties_get(props, "grp_body");
	obs_property_t* grp_face = obs_properties_get(props, "grp_face");

	if (grp_hair)
		obs_property_set_visible(grp_hair, hair);

	if (grp_body)
		obs_property_set_visible(grp_body, body);

	if (grp_face)
		obs_property_set_visible(grp_face, face);

	return true;
}

/*static*/
void ModifyAppearence::add_category_controls(obs_properties_t* props, const char* group_name, const char* group_label, const char* suffix)
{
	obs_properties_t* grp = obs_properties_create();
	obs_properties_add_group(props, group_name, group_label, OBS_GROUP_NORMAL, grp);

	// Key format: <setting>_<suffix>
	obs_properties_add_float_slider(grp, (std::string("gamma_") + suffix).c_str(), "Gamma", 0.10, 3.0, 0.01);                  // default 1.0
	obs_properties_add_float_slider(grp, (std::string("contrast_") + suffix).c_str(), "Contrast", 0.10, 3.0, 0.01);            // default 1.0
	obs_properties_add_float_slider(grp, (std::string("brightness_") + suffix).c_str(), "Brightness", -1.0, 1.0, 0.01);        // default 0.0
	obs_properties_add_float_slider(grp, (std::string("saturation_") + suffix).c_str(), "Saturation", 0.00, 3.0, 0.01);        // default 1.0
	obs_properties_add_float_slider(grp, (std::string("hue_shift_") + suffix).c_str(), "Hue Shift (deg)", -180.0, 180.0, 0.1); // default 0.0
	obs_properties_add_float_slider(grp, (std::string("smooth_") + suffix).c_str(), "Smooth (bilateral)", 0.0, 100.0, 1.0);    // default 0.0
}

/*static*/
void ModifyAppearence::read_cat(obs_data_t* s, const char* suf, CategorySettings& out)
{
	float gamma = obs_data_get_double(s, (std::string("gamma_") + suf).c_str());
	float contrast = obs_data_get_double(s, (std::string("contrast_") + suf).c_str());
	float brightness = obs_data_get_double(s, (std::string("brightness_") + suf).c_str());
	float saturation = obs_data_get_double(s, (std::string("saturation_") + suf).c_str());
	float hue_shift = obs_data_get_double(s, (std::string("hue_shift_") + suf).c_str());
	float smooth = obs_data_get_double(s, (std::string("smooth_") + suf).c_str());

	gamma = (gamma < 0.0) ? (-gamma + 1.0) : (1.0 / (gamma + 1.0));
	contrast = (contrast < 0.0f) ? (1.0f / (-contrast + 1.0f)) : (contrast + 1.0f);
	out.gamma = gamma;

	// Now let's build our Contrast matrix.
	out.con_matrix = {contrast, 0.0f, 0.0f, 0.0f, 0.0f, contrast, 0.0f, 0.0f, 0.0f, 0.0f, contrast, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

	// Now let's build our Brightness matrix.
	// Earlier (in the function color_correction_filter_create) we set
	// this matrix to the identity matrix, so now we only need
	// to set the 3 variables that have changed.

	out.bright_matrix.t.x = brightness;
	out.bright_matrix.t.y = brightness;
	out.bright_matrix.t.z = brightness;

	static const float root3 = 0.57735f;
	static const float red_weight = 0.299f;
	static const float green_weight = 0.587f;
	static const float blue_weight = 0.114f;

	// Factor in the selected color weights.
	float one_minus_sat_red = (1.0f - saturation) * red_weight;
	float one_minus_sat_green = (1.0f - saturation) * green_weight;
	float one_minus_sat_blue = (1.0f - saturation) * blue_weight;
	float sat_val_red = one_minus_sat_red + saturation;
	float sat_val_green = one_minus_sat_green + saturation;
	float sat_val_blue = one_minus_sat_blue + saturation;

	// Now we build our Saturation matrix.
	out.sat_matrix = {sat_val_red, one_minus_sat_red, one_minus_sat_red, 0.0f, one_minus_sat_green, sat_val_green, one_minus_sat_green, 0.0f, one_minus_sat_blue, one_minus_sat_blue, sat_val_blue, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};

	// Hue is the radian of 0 to 360 degrees.
	float half_angle = 0.5f * (float)(hue_shift / (180.0f / M_PI));

	// Pseudo-Quaternion To Matrix.
	float rot_quad1 = root3 * (float)sin(half_angle);
	struct vec3 rot_quaternion;
	vec3_set(&rot_quaternion, rot_quad1, rot_quad1, rot_quad1);
	float rot_quaternion_w = (float)cos(half_angle);

	struct vec3 cross;
	vec3_mul(&cross, &rot_quaternion, &rot_quaternion);
	struct vec3 square;
	vec3_mul(&square, &rot_quaternion, &rot_quaternion);
	struct vec3 wimag;
	vec3_mulf(&wimag, &rot_quaternion, rot_quaternion_w);

	vec3_mulf(&square, &square, 2.0f);
	struct vec3 diag;
	vec3_sub(&diag, &out.half_unit, &square);
	struct vec3 a_line;
	vec3_add(&a_line, &cross, &wimag);
	struct vec3 b_line;
	vec3_sub(&b_line, &cross, &wimag);

	// Now we build our Hue and Opacity matrix.
	static const float opacity = 1.f;
	out.hue_op_matrix = {diag.x * 2.0f, b_line.z * 2.0f, a_line.y * 2.0f, 0.0f, a_line.z * 2.0f, diag.y * 2.0f, b_line.x * 2.0f, 0.0f, b_line.y * 2.0f, a_line.x * 2.0f, diag.z * 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, opacity};

	// Now get the overlay color multiply data.
	static const uint32_t color_multiply = 0x00FFFFFF;
	struct vec4 color_multiply_v4;
	vec4_from_rgba_srgb(&color_multiply_v4, color_multiply);

	// Now get the overlay color add data.
	static const uint32_t color_add = (uint32_t)0x00000000;
	struct vec4 color_add_v4;
	vec4_from_rgba_srgb(&color_add_v4, color_add);

	// Now let's build our Color 'overlay' matrix.
	// Earlier (in the function color_correction_filter_create) we set
	// this matrix to the identity matrix, so now we only need
	// to set the 6 variables that have changed.
	out.color_matrix.x.x = color_multiply_v4.x;
	out.color_matrix.y.y = color_multiply_v4.y;
	out.color_matrix.z.z = color_multiply_v4.z;

	out.color_matrix.t.x = color_add_v4.x;
	out.color_matrix.t.y = color_add_v4.y;
	out.color_matrix.t.z = color_add_v4.z;

	// First we apply the Contrast & Brightness matrix.
	matrix4_mul(&out.final_matrix, &out.con_matrix, &out.bright_matrix);

	// Now we apply the Saturation matrix.
	matrix4_mul(&out.final_matrix, &out.final_matrix, &out.sat_matrix);

	// Next we apply the Hue+Opacity matrix.
	matrix4_mul(&out.final_matrix, &out.final_matrix, &out.hue_op_matrix);

	// Lastly we apply the Color Wash matrix.
	matrix4_mul(&out.final_matrix, &out.final_matrix, &out.color_matrix);
}
