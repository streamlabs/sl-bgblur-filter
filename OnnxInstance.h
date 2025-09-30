#pragma once

#include "OnnxModel.h"

#include <obs.h>
#include <obs-module.h>

#include <string>

#include <opencv2/core/types.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

class OnnxInstance
{
public:
	static OnnxInstance& instance()
	{
		static OnnxInstance a;
		return a;
	}

	void init(const std::wstring& onnxModelPath);
	bool update(obs_source_t* source, gs_texrender_t* texrender, gs_stagesurf_t* stagesurface, OnnxModel::Category cat);

private:
	OnnxInstance();
	~OnnxInstance();

public:
	uint32_t m_maskWidth = 0;
	uint32_t m_maskHeight = 0;
	clock_t m_lastModelRun = 0;

	// This needs to match for all segm categories
	float m_temporalSmoothFactor = 0.5f;

	// Inference / Model configuration
	std::unique_ptr<OnnxModel> m_model;

	// OBS / Graphics handles
	obs_source_t* m_source = nullptr;
	gs_effect_t* m_maskEffect = nullptr;
	gs_effect_t* m_kawaseBlurEffect = nullptr;

	// Frame data
	std::map<OnnxModel::Category, cv::Mat> m_lastSmallMask;
	std::map<OnnxModel::Category, cv::Mat> m_lastFullMask;
	std::map<OnnxModel::Category, cv::Mat> m_lastFullBGRA;
	std::map<OnnxModel::Category, cv::Mat> m_lastOnnxOutput;
};

#define MASK_EFFECT_PATH "mask_alpha_filter.effect"
#define KAWASE_BLUR_EFFECT_PATH "kawase_blur.effect"
