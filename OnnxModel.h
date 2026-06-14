#pragma once

#include <util\platform.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/types.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>
#include <dml_provider_factory.h>

#include <map>

class OnnxModel
{
public:
	enum Category
	{
		CATEGORY_BACKGROUND_INVERSE,
		CATEGORY_HAIR,
		CATEGORY_BODY_SKIN,
		CATEGORY_FACE_SKIN,
		CATEGORY_CLOTHES,
		CATEGORY_OTHERS,

		CATEGORY_NUM_CAT = CATEGORY_OTHERS
	};

public:
	OnnxModel(const std::wstring &onnxPath);
	~OnnxModel();

	bool isGood() const { return m_session != nullptr; }

	bool runImageDisk(const std::string& imgPath, const std::string& out_imgPath);
	bool runImage(const cv::Mat& image, const int cv, std::map<Category, cv::Mat>& output);

private:
	std::wstring getTempFilePath(const std::wstring &fileName);

	Ort::Env m_env;
	std::unique_ptr<Ort::Session> m_session;
	Ort::MemoryInfo m_memInfo;

	std::vector<const char *> m_inputNamesCstr;
	std::vector<const char *> m_outputNamesCstr;

	std::vector<std::string> m_inputNamesStr;
	std::vector<std::string> m_outputNamesStr;
};
