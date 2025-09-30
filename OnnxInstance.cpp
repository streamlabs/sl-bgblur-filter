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

	cv::Mat fullBGRA;

	if (!BgBlurGraphics::getRGBAFromStageSurface(m_maskWidth, m_maskHeight, fullBGRA) || !m_maskEffect)
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

	for (int i = 0; i < OnnxModel::CATEGORY_NUM_CAT; ++i)
	{
		auto cat = (OnnxModel::Category)i;
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

		if (m_smoothContour > 0.0)
		{
			int k = (int)(3 + 11 * m_smoothContour);
			if ((k & 1) == 0)
				++k;

			cv::stackBlur(mask, mask, cv::Size(k, k));
		}

		// Resize mask back to input image size
		cv::resize(mask, mask, fullBGRA.size());

		// If we smoothed, re-binarize
		if (m_smoothContour > 0.0)
			mask = mask > 128;

		m_lastFullMask[cat] = mask;
		m_lastFullBGRA[cat] = fullBGRA.clone();
	}

	return true;
}
