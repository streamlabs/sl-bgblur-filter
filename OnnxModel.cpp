#include <Windows.h>
#include "OnnxModel.h"

#include <obs.hpp>

OnnxModel::OnnxModel(const std::wstring &onnxPath) :
	m_memInfo(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
	m_env(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "segmentation"))
{
	try
	{
		// Init ONNX
		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(1);
		m_session = std::make_unique<Ort::Session>(m_env, onnxPath.c_str(), session_options);

		m_inputNamesStr = m_session->GetInputNames();
		m_outputNamesStr = m_session->GetOutputNames();

		for (auto &n : m_inputNamesStr)
			m_inputNamesCstr.push_back(n.c_str());

		for (auto &n : m_outputNamesStr)
			m_outputNamesCstr.push_back(n.c_str());
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

void OnnxModel::runImage(const std::string &imgPath)
{

	try
	{

		// Categories (same order as training)
		std::vector<std::string> categories = {"background", "hair", "body-skin", "face-skin", "clothes", "others"};

		// Load image
		cv::Mat image_bgr = cv::imread(imgPath);


		int h = 256, w = 256;
		cv::Mat resized, rgb;
		cv::resize(image_bgr, resized, cv::Size(w, h));
		cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

		// Convert to float32 NHWC
		rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

		std::vector<int64_t> input_dims = {1, h, w, 3}; // NHWC
		size_t input_tensor_size = h * w * 3;
		std::vector<float> input_tensor_values(input_tensor_size);

		// Copy data (Mat is row-major NHWC already)
		std::memcpy(input_tensor_values.data(), rgb.data, input_tensor_size * sizeof(float));

		// Create input tensor
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(m_memInfo, input_tensor_values.data(), input_tensor_size, input_dims.data(), input_dims.size());

		std::array<Ort::Value, 1> ort_inputs{std::move(input_tensor)};

		auto tbefore = ::clock();

		std::vector<Ort::Value> output_tensors = m_session->Run(Ort::RunOptions{nullptr}, m_inputNamesCstr.data(), ort_inputs.data(), ort_inputs.size(), m_outputNamesCstr.data(), 1);

		printf("took %dms\n", ::clock() - tbefore);

		// Extract output (assume [1, H, W, C])
		float *output_data = output_tensors.front().GetTensorMutableData<float>();
		std::vector<int64_t> output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

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
				{
					mask.at<float>(y, x) = output_data[(y * out_w * num_classes) + (x * num_classes) + c];
				}
			}

			// Normalize 0–255
			double minVal, maxVal;
			cv::minMaxLoc(mask, &minVal, &maxVal);
			cv::Mat mask_u8;
			mask.convertTo(mask_u8, CV_8U, 255.0 / (maxVal - minVal + 1e-6), -minVal);


			std::string out_path = "C:\\Users\\srogers\\Desktop\\onxtest/" + categories[c] + ".png";
			cv::imwrite(out_path, mask_u8);
		}
	}
	catch (const Ort::Exception &e)
	{
		std::string msg = "ONNX Runtime error: " + std::string(e.what());
		printf("%s\n", msg.c_str());
		blog(LOG_ERROR, "%s", msg.c_str());
	}


}
