#include "OnnxInstance.h"
#include "BgBlur.h"

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

bool OnnxInstance::update(obs_source_t* source, gs_texrender_t* texrender, gs_stagesurf_t* stagesurface, OnnxModel::Category cat)
{
	// More than 1 filter may feed off of this. We don't want to duplicate the workload.
	if (m_lastUpdate == obs_get_video_frame_time())
		return true;

	if (m_model == nullptr)
		return false;

	cv::Mat fullBGRA;

	if (!BgBlurGraphics::getRGBAFromStageSurface(texrender, stagesurface, source, m_maskWidth, m_maskHeight, fullBGRA) || !m_maskEffect)
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
	m_lastUpdate = obs_get_video_frame_time();
	return true;
}
