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

	void init(obs_source_t* source, const std::wstring& onnxModelPath);
	bool update();

private:
	OnnxInstance();
	~OnnxInstance();

public:
	uint32_t m_maskWidth = 0;
	uint32_t m_maskHeight = 0;

	// Inference / Model configuration
	std::unique_ptr<OnnxModel> m_model;

	// OBS / Graphics handles
	obs_source_t *m_source = nullptr;
	gs_texrender_t *m_texrender = nullptr;
	gs_stagesurf_t *m_stagesurface = nullptr;
	gs_effect_t *m_maskEffect = nullptr;
	gs_effect_t *m_kawaseBlurEffect = nullptr;

	// Frame data
	std::map<OnnxModel::Category, cv::Mat> m_lastSmallMask;
	std::map<OnnxModel::Category, cv::Mat> m_lastFullMask;
	std::map<OnnxModel::Category, cv::Mat> m_lastFullBGRA;
	std::map<OnnxModel::Category, cv::Mat> m_lastOnnxOutput;

	// State flags
	bool m_isDisabled = false;

	// Threshold / Masking controls
	bool m_enableThreshold = true;
	float m_threshold = 0.5f;
	cv::Scalar m_backgroundColor{0, 0, 0, 0};
	float m_contourFilter = 0.05f;
	float m_smoothContour = 1.0f;
	float m_feather = 0.0f;
	clock_t m_lastModelRun = 0;

	// Similarity & temporal smoothing
	float m_temporalSmoothFactor = 0.5f;
	float m_imageSimilarityThreshold = 35.0f;
	bool m_enableImageSimilarity = true;

	// Blur / Depth settings
	int64_t m_blurBackground = 10;
	float m_blurFocusPoint = 0.1f;
	float m_blurFocusDepth = 0.0f;
	bool m_enableFocalBlur = false;
};

#define MASK_EFFECT_PATH "mask_alpha_filter.effect"
#define KAWASE_BLUR_EFFECT_PATH "kawase_blur.effect"
