#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <obs.h>
#include <obs-module.h>
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
		// Immitating OBS color correction 
		struct matrix4
		{
			struct vec4 {
				union {
					struct {
						float x, y, z, w;
					};
					float ptr[4];
					__m128 m;
				};
			};

			struct vec4 x, y, z, t;
		};
		
		gs_eparam_t* gamma_param = nullptr;
		gs_eparam_t* final_matrix_param = nullptr;

		float gamma = 0.f;

		struct matrix4 con_matrix;
		struct matrix4 bright_matrix;
		struct matrix4 sat_matrix;
		struct matrix4 hue_op_matrix;
		struct matrix4 color_matrix;
		struct matrix4 final_matrix;

		struct vec3 half_unit;
	};

	struct ModData
	{
		gs_effect_t* effect = nullptr;
		obs_source_t* source = nullptr;

		gs_texrender_t* texrender = nullptr;
		gs_stagesurf_t* stagesurface = nullptr;
		gs_effect_t* maskEffect = nullptr;

		CategorySettings cat[OnnxModel::Category::CATEGORY_NUM_CAT];

		float temporalSmooth = 0.0f;
	};

private:
	static bool show_only_selected_group(obs_properties_t* props, obs_property_t* list, obs_data_t* settings);
	static void add_category_controls(obs_properties_t* props, const char* group_name, const char* group_label, const char* suffix);
	static void read_cat(obs_data_t* s, const char* suf, CategorySettings& out);

	std::vector<OnnxModel::Category> m_cats = { OnnxModel::CATEGORY_HAIR, OnnxModel::CATEGORY_BODY_SKIN, OnnxModel::CATEGORY_FACE_SKIN };
};
