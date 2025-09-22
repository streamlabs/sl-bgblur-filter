#pragma once

#include <obs.h>
#include <obs-module.h>

#include <opencv2/core/types.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

#include "OnnxModel.h"

#define MASK_EFFECT_PATH "mask_alpha_filter.effect"
#define KAWASE_BLUR_EFFECT_PATH "kawase_blur.effect"

struct FilterData
{
public:
	// Inference / Model configuration 
	std::unique_ptr<OnnxModel> model;
	std::mutex modelMutex;

	// OBS / Graphics handles
	obs_source_t *source = nullptr;
	gs_texrender_t *texrender = nullptr;
	gs_stagesurf_t *stagesurface = nullptr;
	gs_effect_t *maskEffect = nullptr;
	gs_effect_t *kawaseBlurEffect = nullptr;

	// Frame data
	cv::Mat lastSmallBackgroundMask;
	cv::Mat lastFullBackgroundMask;
	cv::Mat lastFullBGRA;
	cv::Mat inputBGRA;
	std::map<OnnxModel::Category, cv::Mat> lastOnnxOutput;

	// State flags
	bool isDisabled = false;

	// Threshold / Masking controls
	bool enableThreshold = true; 
	float threshold = 0.5f;      
	cv::Scalar backgroundColor{0, 0, 0, 0};
	float contourFilter = 0.05f; 
	float smoothContour = 1.0f;  
	float feather = 0.0f;        
	clock_t lastModelRun = 0;

	// Similarity & temporal smoothing
	float temporalSmoothFactor = 0.5f;     
	float imageSimilarityThreshold = 35.0f;
	bool enableImageSimilarity = true;     

	// Blur / Depth settings
	int64_t blurBackground = 10; 
	float blurFocusPoint = 0.1f; 
	float blurFocusDepth = 0.0f; 
	bool enableFocalBlur = false;
};
