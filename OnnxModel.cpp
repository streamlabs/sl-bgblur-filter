#include <Windows.h>
#include "OnnxModel.h"

#include <obs.hpp>

OnnxModel::OnnxModel(const std::wstring &onnxPath) :
	m_memInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
	m_env(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "segmentation"))
{
	try
	{
		Ort::AllocatorWithDefaultOptions allocator;

		// Init ONNX
		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(1);

		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		session_options.DisableMemPattern();
		session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

		OrtDmlApi *dmlApi = nullptr;
		Ort::ThrowOnError(Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, (const void **)&dmlApi));
		Ort::ThrowOnError(dmlApi->SessionOptionsAppendExecutionProvider_DML(session_options, 0));

		m_session = std::make_unique<Ort::Session>(m_env, onnxPath.c_str(), session_options);

		size_t numInputs = m_session->GetInputCount();
		size_t numOutputs = m_session->GetOutputCount();

		// Fetch input names
		for (size_t i = 0; i < numInputs; i++)
		{
			Ort::AllocatedStringPtr nameAllocated = m_session->GetInputNameAllocated(i, allocator);
			m_inputNamesStr.push_back(nameAllocated.get());
			m_inputNamesCstr.push_back(m_inputNamesStr.back().c_str());
		}

		// Fetch output names
		for (size_t i = 0; i < numOutputs; i++)
		{
			Ort::AllocatedStringPtr nameAllocated = m_session->GetOutputNameAllocated(i, allocator);
			m_outputNamesStr.push_back(nameAllocated.get());
			m_outputNamesCstr.push_back(m_outputNamesStr.back().c_str());
		}
	}
	catch (const Ort::Exception &e)
	{
		std::string msg = "ONNX Runtime error: " + std::string(e.what());
		printf("%s\n", msg.c_str());
		blog(LOG_ERROR, "%s", msg.c_str());
	}

}

OnnxModel::~OnnxModel()
{
	m_session = nullptr;
}

void OnnxModel::runImage(const cv::Mat &image, const int cv, std::map<Category, cv::Mat> &output)
{
	try
	{
		static int h = 256, w = 256;
		static std::vector<int64_t> input_dims = {1, h, w, 3};
		static size_t input_tensor_size = h * w * 3;

		cv::Mat resized, rgb;
		cv::resize(image, resized, cv::Size(w, h));
		cv::cvtColor(resized, rgb, cv);

		// Convert to float32 NHWC
		rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

		// Create input tensor
		std::vector<float> inputTensors(input_tensor_size);
		std::memcpy(inputTensors.data(), rgb.data, input_tensor_size * sizeof(float));

		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(m_memInfo, inputTensors.data(), input_tensor_size, input_dims.data(), input_dims.size());
		std::array<Ort::Value, 1> ort_inputs{std::move(input_tensor)};

		// Run
		std::vector<Ort::Value> outputTensors = m_session->Run(Ort::RunOptions{nullptr}, m_inputNamesCstr.data(), ort_inputs.data(), ort_inputs.size(), m_outputNamesCstr.data(), 1);

		// Extract output (assume [1, H, W, C])
		float *output_data = outputTensors.front().GetTensorMutableData<float>();
		std::vector<int64_t> output_shape = outputTensors.front().GetTensorTypeAndShapeInfo().GetShape();

		int out_h = static_cast<int>(output_shape[1]);
		int out_w = static_cast<int>(output_shape[2]);
		int num_classes = static_cast<int>(output_shape[3]);

		// Save per-category masks
		for (int c = 0; c < num_classes; c++)
		{
			cv::Mat mask(out_h, out_w, CV_32F);

			for (int y = 0; y < out_h; y++)
			{
				for (int x = 0; x < out_w; x++)
					mask.at<float>(y, x) = output_data[(y * out_w * num_classes) + (x * num_classes) + c];
			}

			// Normalize 0–255
			double minVal, maxVal;
			cv::minMaxLoc(mask, &minVal, &maxVal);
			cv::Mat mask_u8;
			mask.convertTo(mask_u8, CV_8U, 255.0 / (maxVal - minVal + 1e-6), -minVal);

			output[(Category)c] = mask_u8;
		}
	}
	catch (const Ort::Exception &e)
	{
		std::string msg = "ONNX Runtime error: " + std::string(e.what());
		printf("%s\n", msg.c_str());
		blog(LOG_ERROR, "%s", msg.c_str());
	}
}

void OnnxModel::runImageDisk(const std::string& imgPath)
{
	static std::vector<std::string> categories = {"background", "hair", "body-skin", "face-skin", "clothes", "others"};

	std::map<Category, cv::Mat> output;
	runImage(cv::imread(imgPath), cv::COLOR_BGR2RGB, output);

	for (auto& itr : output)
	{
		std::string out_path = "C:\\Users\\srogers\\Desktop\\onxtest/" + categories[itr.first] + ".png";
		cv::imwrite(out_path, itr.second);
	}
}
