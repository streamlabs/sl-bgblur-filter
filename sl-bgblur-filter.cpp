#include <Windows.h>

#include <obs.hpp>
#include <obs-module.h>
#include <obs-config.h>
#include <util\platform.h>

#include "BgBlur.h"
#include "ModifyAppearence.h"
#include "OnnxModel.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("sl-bgblur-filter", "en-US")
MODULE_EXPORT const char *obs_module_description(void)
{
	return "SLABS BG Remover";
}

bool obs_module_load(void)
{
	// BgBlur
	struct obs_source_info bgblur = {};
	bgblur.id = "sl-bgblur-filter";
	bgblur.type = OBS_SOURCE_TYPE_FILTER;
	bgblur.output_flags = OBS_SOURCE_VIDEO;
	bgblur.get_name = BgBlur::obs_getname;
	bgblur.create = BgBlur::obs_create;
	bgblur.destroy = BgBlur::obs_destroy;
	bgblur.get_defaults = BgBlur::obs_defaults;
	bgblur.get_properties = BgBlur::obs_properties;
	bgblur.update = BgBlur::obs_update_settings;
	bgblur.activate = BgBlur::obs_activate;
	bgblur.deactivate = BgBlur::obs_deactivate;
	bgblur.video_tick = BgBlur::obs_video_tick;
	bgblur.video_render = BgBlur::obs_video_render;
	obs_register_source(&bgblur);

	// ModifyAppearence
	struct obs_source_info modapp = {};
	modapp.id = "sl-modapp-filter";
	modapp.type = OBS_SOURCE_TYPE_FILTER;
	modapp.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_SRGB;
	modapp.get_name = ModifyAppearence::obs_getname;
	modapp.create = ModifyAppearence::obs_create;
	modapp.destroy = ModifyAppearence::obs_destroy;
	modapp.get_defaults = ModifyAppearence::obs_defaults;
	modapp.get_properties = ModifyAppearence::obs_properties;
	modapp.update = ModifyAppearence::obs_update_settings;
	modapp.activate = ModifyAppearence::obs_activate;
	modapp.deactivate = ModifyAppearence::obs_deactivate;
	modapp.video_tick = ModifyAppearence::obs_video_tick;
	modapp.video_render = ModifyAppearence::obs_video_render;
	modapp.video_get_color_space = ModifyAppearence::obs_video_get_color_space;
	obs_register_source(&modapp);

	return true;
}

void obs_module_post_load(void)
{

}

void obs_module_unload(void)
{
	;
}

/*

	// Init ONNX Runtime
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "segmentation");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);

	try
	{
		Ort::Session session(env, L"C:\\Users\\srogers\\Desktop\\onxtest\\test.onnx", session_options);	

		// Categories (same order as training)
		std::vector<std::string> categories = {"background", "hair", "body-skin", "face-skin", "clothes", "others"};

		// Load image
		cv::Mat image_bgr = cv::imread("C:\\Users\\srogers\\Desktop\\onxtest\\zuck_original.jpg");

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
		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_dims.data(), input_dims.size());

		// Run inference
		std::array<Ort::Value, 1> ort_inputs{std::move(input_tensor)};

		auto inputNames = session.GetInputNames();
		auto outputNames = session.GetOutputNames();

		std::vector<const char *> inNames;
		inNames.reserve(inputNames.size());
		std::vector<const char *> outNames;
		outNames.reserve(outputNames.size());
		for (auto &n : inputNames)
			inNames.push_back(n.c_str());
		for (auto &n : outputNames)
			outNames.push_back(n.c_str());

		std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, // run options
								     inNames.data(),           // input names
								     ort_inputs.data(),        // input tensors
								     ort_inputs.size(),        // number of inputs
								     outNames.data(),          // output names
								     1                         // number of outputs
		);


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
		std::string msg = "ONNX Runtime error: ";
		msg += e.what(); // get error string

		printf("%s\n", msg.c_str());
		MessageBoxA(nullptr, msg.c_str(),"ONNX Runtime Exception", MB_ICONERROR | MB_OK);
	}
	catch (const std::exception &e)
	{
		// fallback for other exceptions
		std::string msg = "Unexpected error: ";
		msg += e.what();
		printf("%s\n", msg.c_str());

		MessageBoxA(nullptr, msg.c_str(), "Exception", MB_ICONERROR | MB_OK);
	}

	return 0;

*/
