#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <obs.h>
#include <obs-module.h>
#include <graphics/matrix4.h>
#include <opencv2/core/types.hpp>

#include "OnnxModel.h"

/*static*/
class ModifyAppearence
{
public:
	static ModifyAppearence &instance()
	{
		static ModifyAppearence a;
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
	ModifyAppearence();
	~ModifyAppearence();

	struct CategorySettings
	{		
		float gamma = 0.f;
		float contrast = 0.f;
		float brightness = 0.f;
		float saturation = 0.f;
		float smooth = 0.f;

		gs_effect_t* maskEffect = nullptr;
	};

	struct ModData
	{
		gs_effect_t* effect = nullptr;
		obs_source_t* source = nullptr;

		gs_texrender_t* texrender = nullptr;
		gs_stagesurf_t* stagesurface = nullptr;

		CategorySettings cat[OnnxModel::Category::CATEGORY_NUM_CAT];

		float temporalSmooth = 0.0f;
	};

private:
	static bool show_only_selected_group(obs_properties_t* props, obs_property_t* list, obs_data_t* settings);
	static void add_category_controls(obs_properties_t* props, const char* group_name, const char* group_label, const char* suffix);
	static void read_cat(obs_data_t* s, const char* suf, CategorySettings& out);

	std::vector<OnnxModel::Category> m_cats = {OnnxModel::CATEGORY_HAIR, OnnxModel::CATEGORY_BODY_SKIN, OnnxModel::CATEGORY_FACE_SKIN, OnnxModel::CATEGORY_CLOTHES };
};
