#include "OnnxInstance.h"
#include "BgBlur.h"

/**
* Onnx
*/

OnnxInstance* Onnx::get(obs_source_t* source)
{
	auto it = m_instances.find(source);

	if (it == m_instances.end())
		return nullptr;

	return it->second.second.get();
}

void Onnx::registerIncrementSource(obs_source_t* source)
{
	auto it = m_instances.find(source);

	if (it == m_instances.end())
		m_instances[source] = std::make_pair(1, std::make_unique<OnnxInstance>());
	else
		it->second.first++;
}

void Onnx::unregisterDeIncrementSource(obs_source_t* source)
{
	auto it = m_instances.find(source);

	if (it == m_instances.end())
		return;

	it->second.first--;

	if (it->second.first <= 0)
		m_instances.erase(it);
}

/**
* OnnxInstance
*/

OnnxInstance::OnnxInstance()
{

}

OnnxInstance::~OnnxInstance()
{

}

void OnnxInstance::init(const std::wstring &onnxModelPath)
{
	//::MessageBox(0, onnxModelPath.c_str(), onnxModelPath.c_str(), 0);

	m_model = std::make_unique<OnnxModel>(onnxModelPath.c_str());
	
	if (!m_model->isGood())
	{
		blog(LOG_ERROR, "OnnxModel failed to init properly.");
		m_model = nullptr;
	}
}

bool OnnxInstance::update(obs_source_t* source, gs_texrender_t* texrender, gs_stagesurf_t* stagesurface, const std::vector<OnnxModel::Category>& cats)
{
	// More than 1 filter may feed off of this. We don't want to duplicate the workload.
	if (m_lastUpdate == obs_get_video_frame_time())
		return true;

	if (m_model == nullptr)
		return false;

	cv::Mat fullBGRA;

	if (!getRGBAFromStageSurface(texrender, stagesurface, source, m_maskWidth, m_maskHeight, fullBGRA))
		return false;

	bool boolQueryOnnx = true;

	// 10fps is fine because we use temporal smoothing
	if (boolQueryOnnx && ::clock() - m_lastModelRun < 50)
		boolQueryOnnx = false;

	cv::Mat mask;

	if (boolQueryOnnx)
	{
		m_model->runImage(fullBGRA, cv::COLOR_BGRA2RGB, m_lastOnnxOutput);
		m_lastModelRun = ::clock();
	}

	for (auto& cat : cats)
	{
		auto ref = m_lastOnnxOutput[cat].clone();

		if (ref.empty())
			return false;

		if (m_temporalSmoothFactor <= 0 || m_lastSmallMask[cat].empty() || m_lastSmallMask[cat].size() != ref.size() || m_lastSmallMask[cat].type() != ref.type())
		{
			m_lastSmallMask[cat] = ref.clone();
		}
		else if (m_temporalSmoothFactor > 0)
		{
			const double f = std::clamp(m_temporalSmoothFactor, 0.0f, 1.0f);
			cv::addWeighted(m_lastSmallMask[cat], f, ref, 1.0 - f, 0.0, m_lastSmallMask[cat]);
		}

		mask = m_lastSmallMask[cat].clone();

		// edit as needed
		static float smoothContour = 1.0f;

		if (smoothContour > 0.0)
		{
			int k = (int)(3 + 11);
			if ((k & 1) == 0)
				++k;

			cv::stackBlur(mask, mask, cv::Size(k, k));
		}

		// Resize mask back to input image size
		cv::resize(mask, mask, fullBGRA.size());

		// If we smoothed, re-binarize
		if (smoothContour > 0.0)
			mask = mask > 128;

		m_lastFullMask[cat] = mask;
		m_lastFullBGRA[cat] = fullBGRA.clone();
	}
	m_lastUpdate = obs_get_video_frame_time();
	return true;
}

/*static*/
bool OnnxInstance::getRGBAFromStageSurface(gs_texrender_t*& texrender, gs_stagesurf_t*& stagesurface, obs_source_t* source, uint32_t &width, uint32_t &height, cv::Mat &outputBGRA)
{
	// Captures a live video frame from a source, renders it to a texture, transfers it onto
	//	a staging surface, maps it into CPU-accessible memory, then it wraps the pixel buffer into an OpenCV cv::Mat (BGRA format)

	if (!obs_source_enabled(source))
		return false;

	obs_source_t* target = obs_filter_get_target(source);

	if (!target)
		return false;

	width = obs_source_get_base_width(target);
	height = obs_source_get_base_height(target);

	if (width == 0 || height == 0)
		return false;

	gs_texrender_reset(texrender);

	if (!gs_texrender_begin(texrender, width, height))
		return false;

	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(texrender);

	if (stagesurface)
	{
		uint32_t stagesurf_width = gs_stagesurface_get_width(stagesurface);
		uint32_t stagesurf_height = gs_stagesurface_get_height(stagesurface);

		if (stagesurf_width != width || stagesurf_height != height)
		{
			gs_stagesurface_destroy(stagesurface);
			stagesurface = nullptr;
		}
	}

	if (!stagesurface)
		stagesurface = gs_stagesurface_create(width, height, GS_BGRA);

	gs_stage_texture(stagesurface, gs_texrender_get_texture(texrender));

	uint8_t* video_data;
	uint32_t linesize;

	if (!gs_stagesurface_map(stagesurface, &video_data, &linesize))
		return false;

	outputBGRA = cv::Mat(height, width, CV_8UC4, video_data, linesize);

	gs_stagesurface_unmap(stagesurface);
	return true;
}
