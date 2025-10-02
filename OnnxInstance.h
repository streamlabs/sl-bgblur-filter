#pragma once

#include "OnnxModel.h"

#include <obs.h>
#include <obs-module.h>

#include <string>

#include <opencv2/core/types.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

class OnnxInstance;

class Onnx
{
public:
	static Onnx &instance()
	{
		static Onnx a;
		return a;
	}

	OnnxInstance* get(obs_source_t *source);

	void registerIncrementSource(obs_source_t *source);
	void unregisterDeIncrementSource(obs_source_t *source);

private:
	Onnx() = default;
	Onnx(const Onnx &) = delete;
	Onnx &operator=(const Onnx &) = delete;

private:
	// int is a ++, -- counter at each registration. A source might be using the same onnx instance from 5 filters or something like that
	std::map<obs_source_t*, std::pair<int, std::unique_ptr<OnnxInstance>>> m_instances;
};

class OnnxInstance
{
friend class Onnx;

public:
	OnnxInstance();
	~OnnxInstance();

public:
	void init(const std::wstring& onnxModelPath);
	bool update(obs_source_t* source, gs_texrender_t* texrender, gs_stagesurf_t* stagesurface, OnnxModel::Category cat);

public:
	uint32_t m_maskWidth = 0;
	uint32_t m_maskHeight = 0;
	uint64_t m_lastUpdate = 0;
	clock_t m_lastModelRun = 0;
	
	// This needs to match for all segm categories
	float m_temporalSmoothFactor = 0.5f;

	// Inference / Model configuration
	std::unique_ptr<OnnxModel> m_model;

	// OBS / Graphics handles
	gs_effect_t* m_maskEffect = nullptr;
	gs_effect_t* m_kawaseBlurEffect = nullptr;

	// Frame data
	std::map<OnnxModel::Category, cv::Mat> m_lastSmallMask;
	std::map<OnnxModel::Category, cv::Mat> m_lastFullMask;
	std::map<OnnxModel::Category, cv::Mat> m_lastFullBGRA;
	std::map<OnnxModel::Category, cv::Mat> m_lastOnnxOutput;

private:
	bool getRGBAFromStageSurface(gs_texrender_t* texrender, gs_stagesurf_t* stagesurface, obs_source_t* source, uint32_t& width, uint32_t& height, cv::Mat& outputBGRA);
};

#define MASK_EFFECT_PATH "mask_alpha_filter.effect"
#define KAWASE_BLUR_EFFECT_PATH "kawase_blur.effect"
