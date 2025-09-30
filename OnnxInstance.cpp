#include "OnnxInstance.h"
#include "BgBlur.h"

OnnxInstance::OnnxInstance()
{

}

OnnxInstance::~OnnxInstance()
{

}

void OnnxInstance::init(obs_source_t *source, const std::wstring &onnxModelPath)
{
	//::MessageBox(0, onnxModelPath.c_str(), onnxModelPath.c_str(), 0);

	m_source = source;
	m_texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
	m_model = std::make_unique<OnnxModel>(onnxModelPath.c_str());
	
	if (!m_model->isGood())
	{
		blog(LOG_ERROR, "OnnxModel failed to init properly.");
		m_model = nullptr;
	}
}

bool OnnxInstance::update()
{
	if (m_model == nullptr)
		return false;

	if (!BgBlurGraphics::getRGBAFromStageSurface(m_maskWidth, m_maskHeight) || !m_maskEffect)
		return false;

	bool boolQueryOnnx = true;
	cv::Mat fullBGRA = m_inputBGRA.clone();

	// 10fps is fine because we use temporal smoothing
	if (boolQueryOnnx && ::clock() - m_lastModelRun < 50)
		boolQueryOnnx = false;

	cv::Mat backgroundMask;

	if (boolQueryOnnx)
	{
		m_model->runImage(fullBGRA, cv::COLOR_BGRA2RGB, m_lastOnnxOutput);
		m_lastModelRun = ::clock();
	}

	auto bgRef = m_lastOnnxOutput[OnnxModel::CATEGORY_BACKGROUND_INVERSE].clone();

	if (bgRef.empty())
		return false;

	if (m_temporalSmoothFactor <= 0 || m_lastSmallBackgroundMask.empty() || m_lastSmallBackgroundMask.size() != bgRef.size() || m_lastSmallBackgroundMask.type() != bgRef.type())
	{
		m_lastSmallBackgroundMask = bgRef.clone();
	}
	else if (m_temporalSmoothFactor > 0)
	{
		const double f = std::clamp(m_temporalSmoothFactor, 0.0f, 1.0f);
		cv::addWeighted(m_lastSmallBackgroundMask, f, bgRef, 1.0 - f, 0.0, m_lastSmallBackgroundMask);
	}

	backgroundMask = m_lastSmallBackgroundMask.clone();

	if (m_smoothContour > 0.0)
	{
		int k = (int)(3 + 11 * m_smoothContour);
		if ((k & 1) == 0)
			++k;

		cv::stackBlur(backgroundMask, backgroundMask, cv::Size(k, k));
	}

	// Resize mask back to input image size
	cv::resize(backgroundMask, backgroundMask, fullBGRA.size());

	// If we smoothed, re-binarize
	if (m_smoothContour > 0.0)
		backgroundMask = backgroundMask > 128;

	m_lastFullBackgroundMask = backgroundMask;
	m_lastFullBGRA = fullBGRA.clone();
	return true;
}
