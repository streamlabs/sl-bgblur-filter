#pragma once

#include <util\platform.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/types.hpp>
#include <onnxruntime_cxx_api.h>
#include <cpu_provider_factory.h>

class OnnxModel
{
public:
	OnnxModel(const std::wstring& onnxPath);
	~OnnxModel();

	void runImage(const std::string &imgPath);

private:
	Ort::Env m_env;
	std::unique_ptr<Ort::Session> m_session;
	Ort::MemoryInfo m_memInfo;

	std::vector<const char *> m_inputNamesCstr;
	std::vector<const char *> m_outputNamesCstr;

	std::vector<std::string> m_inputNamesStr;
	std::vector<std::string> m_outputNamesStr;
};
