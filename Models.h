#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <memory>
#include <vector>

//#define MODEL_MEDIAPIPE "mediapipe.onnx"
//#define MODEL_SINET "SINet_Softmax_simple.onnx"
//#define MODEL_SELFIE "selfie_segmentation.onnx"
//#define MODEL_RVM "rvm_mobilenetv3_fp32.onnx"
//#define MODEL_PPHUMANSEG "pphumanseg_fp32.onnx"
//#define MODEL_DEPTH_TCMONODEPTH "tcmonodepth_tcsmallnet_192x320.onnx"
//#define MODEL_RMBG "bria_rmbg_1_4_qint8.onnx"

class Model
{
public:
	virtual ~Model() = default;

	// Names / shapes
	virtual void populateInputOutputNames(const std::unique_ptr<Ort::Session> &session, std::vector<Ort::AllocatedStringPtr> &inputNames, std::vector<Ort::AllocatedStringPtr> &outputNames)
	{
		Ort::AllocatorWithDefaultOptions allocator;
		inputNames.clear();
		outputNames.clear();
		inputNames.push_back(session->GetInputNameAllocated(0, allocator));
		outputNames.push_back(session->GetOutputNameAllocated(0, allocator));
	}

	virtual bool populateInputOutputShapes(const std::unique_ptr<Ort::Session> &session, std::vector<std::vector<int64_t>> &inputDims, std::vector<std::vector<int64_t>> &outputDims)
	{
		inputDims.clear();
		outputDims.clear();
		inputDims.push_back(std::vector<int64_t>());
		outputDims.push_back(std::vector<int64_t>());

		// Output
		{
			const Ort::TypeInfo outInfo = session->GetOutputTypeInfo(0);
			const auto outTensorInfo = outInfo.GetTensorTypeAndShapeInfo();
			outputDims[0] = outTensorInfo.GetShape();

			for (auto &i : outputDims[0])
			{
				if (i == -1)
					i = 1;
			}
		}

		// Input
		{
			const Ort::TypeInfo inInfo = session->GetInputTypeInfo(0);
			const auto inTensorInfo = inInfo.GetTensorTypeAndShapeInfo();
			inputDims[0] = inTensorInfo.GetShape();

			for (auto &i : inputDims[0])
			{
				if (i == -1)
					i = 1;
			}
		}

		return inputDims[0].size() >= 3 && outputDims[0].size() >= 3;
	}

	// Tensor buffers
	virtual void allocateTensorBuffers(const std::vector<std::vector<int64_t>> &inputDims, const std::vector<std::vector<int64_t>> &outputDims, std::vector<std::vector<float>> &outputTensorValues, std::vector<std::vector<float>> &inputTensorValues, std::vector<Ort::Value> &inputTensor,
					   std::vector<Ort::Value> &outputTensor)
	{
		outputTensorValues.clear();
		outputTensor.clear();
		inputTensorValues.clear();
		inputTensor.clear();

		Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);

		for (size_t i = 0; i < inputDims.size(); ++i)
		{
			inputTensorValues.emplace_back(vectorProduct(inputDims[i]), 0.0f);
			inputTensor.push_back(Ort::Value::CreateTensor<float>(memInfo, inputTensorValues[i].data(), inputTensorValues[i].size(), inputDims[i].data(), inputDims[i].size()));
		}
		for (size_t i = 0; i < outputDims.size(); ++i)
		{
			outputTensorValues.emplace_back(vectorProduct(outputDims[i]), 0.0f);
			outputTensor.push_back(Ort::Value::CreateTensor<float>(memInfo, outputTensorValues[i].data(), outputTensorValues[i].size(), outputDims[i].data(), outputDims[i].size()));
		}
	}

	// IO sizes / pre-post
	virtual void getNetworkInputSize(const std::vector<std::vector<int64_t>> &inputDims, uint32_t &inputWidth, uint32_t &inputHeight)
	{
		// Default BHWC
		inputWidth = (uint32_t)inputDims[0][2];
		inputHeight = (uint32_t)inputDims[0][1];
	}

	virtual void prepareInputToNetwork(cv::Mat &resizedImage, cv::Mat &preprocessedImage) { preprocessedImage = resizedImage / 255.f; }

	virtual void postprocessOutput(cv::Mat &output) { (void)output; }

	virtual void loadInputToTensor(const cv::Mat &preprocessedImage, uint32_t inputWidth, uint32_t inputHeight, std::vector<std::vector<float>> &inputTensorValues) { preprocessedImage.copyTo(cv::Mat(inputHeight, inputWidth, CV_32FC3, &(inputTensorValues[0][0]))); }

	virtual cv::Mat getNetworkOutput(const std::vector<std::vector<int64_t>> &outputDims, std::vector<std::vector<float>> &outputTensorValues)
	{
		// Default BHWC → CV_32F(C)
		const uint32_t W = (uint32_t)outputDims[0].at(2);
		const uint32_t H = (uint32_t)outputDims[0].at(1);
		const int Ctype = CV_MAKE_TYPE(CV_32F, (int)outputDims[0].at(3));
		return cv::Mat(H, W, Ctype, outputTensorValues[0].data());
	}

	virtual void assignOutputToInput(std::vector<std::vector<float>> &, std::vector<std::vector<float>> &) {}

	// Inference
	virtual void runNetworkInference(const std::unique_ptr<Ort::Session> &session, const std::vector<Ort::AllocatedStringPtr> &inputNames, const std::vector<Ort::AllocatedStringPtr> &outputNames, const std::vector<Ort::Value> &inputTensor, std::vector<Ort::Value> &outputTensor)
	{
		if (inputNames.empty() || outputNames.empty() || inputTensor.empty() || outputTensor.empty())
			return;

		std::vector<const char *> inNames;
		inNames.reserve(inputNames.size());
		std::vector<const char *> outNames;
		outNames.reserve(outputNames.size());
		for (auto &n : inputNames)
			inNames.push_back(n.get());
		for (auto &n : outputNames)
			outNames.push_back(n.get());

		session->Run(Ort::RunOptions{nullptr}, inNames.data(), inputTensor.data(), (size_t)inNames.size(), outNames.data(), outputTensor.data(), (size_t)outNames.size());
	}

protected:
	template<typename T>
	static inline T vectorProduct(const std::vector<T> &v)
	{
		T product = 1;

		for (auto &i : v)
		{
			if (i > 0)
				product *= i; // treat 0/-1 as dynamic, map to 1
		}

		return product;
	}

	static inline void hwc_to_chw(cv::InputArray src, cv::OutputArray dst)
	{
		std::vector<cv::Mat> channels;
		cv::split(src, channels);

		for (auto &img : channels)
			img = img.reshape(1, 1);

		cv::hconcat(channels, dst);
	}

	static inline void chw_to_hwc_32f(cv::InputArray src, cv::OutputArray dst)
	{
		const cv::Mat srcMat = src.getMat();
		const int channels = srcMat.channels();
		const int height = srcMat.rows;
		const int width = srcMat.cols;
		const int dtype = srcMat.type();
		(void)dtype;

		const int channelStride = height * width;
		cv::Mat flat = srcMat.reshape(1, 1);

		std::vector<cv::Mat> chs(channels);

		for (int i = 0; i < channels; ++i)
			chs[i] = cv::Mat(height, width, CV_MAKE_TYPE(CV_32F, 1), flat.ptr<float>(0) + i * channelStride);
		
		cv::merge(chs, dst);
	}
};

class ModelMediaPipe : public Model
{
public:
	cv::Mat getNetworkOutput(const std::vector<std::vector<int64_t>> &outputDims, std::vector<std::vector<float>> &outputTensorValues) override
	{
		const uint32_t W = (uint32_t)outputDims[0].at(2);
		const uint32_t H = (uint32_t)outputDims[0].at(1);
		return cv::Mat(H, W, CV_32FC2, outputTensorValues[0].data());
	}
	void postprocessOutput(cv::Mat &outputImage) override
	{
		std::vector<cv::Mat> splitv;
		cv::split(outputImage, splitv);
		outputImage = splitv[1]; // keep channel 1
	}
};

class ModelBCHW : public Model
{
public:
	ModelBCHW(/* args */) {}
	~ModelBCHW() {}

	virtual void prepareInputToNetwork(cv::Mat &resizedImage, cv::Mat &preprocessedImage)
	{
		resizedImage = resizedImage / 255.0;
		hwc_to_chw(resizedImage, preprocessedImage);
	}

	virtual void postprocessOutput(cv::Mat &output)
	{
		cv::Mat outputTransposed;
		chw_to_hwc_32f(output, outputTransposed);
		outputTransposed.copyTo(output);
	}

	virtual void getNetworkInputSize(const std::vector<std::vector<int64_t>> &inputDims, uint32_t &inputWidth, uint32_t &inputHeight)
	{
		// BCHW
		inputWidth = (int)inputDims[0][3];
		inputHeight = (int)inputDims[0][2];
	}

	virtual cv::Mat getNetworkOutput(const std::vector<std::vector<int64_t>> &outputDims, std::vector<std::vector<float>> &outputTensorValues)
	{
		// BCHW
		uint32_t outputWidth = (int)outputDims[0].at(3);
		uint32_t outputHeight = (int)outputDims[0].at(2);
		int32_t outputChannels = CV_MAKE_TYPE(CV_32F, (int)outputDims[0].at(1));

		return cv::Mat(outputHeight, outputWidth, outputChannels, outputTensorValues[0].data());
	}

	virtual void loadInputToTensor(const cv::Mat &preprocessedImage, uint32_t, uint32_t, std::vector<std::vector<float>> &inputTensorValues) { inputTensorValues[0].assign(preprocessedImage.begin<float>(), preprocessedImage.end<float>()); }
};

class ModelPPHumanSeg : public ModelBCHW
{
public:
	ModelPPHumanSeg(/* args */) {}
	~ModelPPHumanSeg() {}

	virtual void prepareInputToNetwork(cv::Mat &resizedImage, cv::Mat &preprocessedImage)
	{
		resizedImage = (resizedImage / 256.0 - cv::Scalar(0.5, 0.5, 0.5)) / cv::Scalar(0.5, 0.5, 0.5);

		hwc_to_chw(resizedImage, preprocessedImage);
	}

	virtual cv::Mat getNetworkOutput(const std::vector<std::vector<int64_t>> &outputDims, std::vector<std::vector<float>> &outputTensorValues)
	{
		uint32_t outputWidth = (int)outputDims[0].at(2);
		uint32_t outputHeight = (int)outputDims[0].at(1);
		int32_t outputChannels = CV_32FC2;

		return cv::Mat(outputHeight, outputWidth, outputChannels, outputTensorValues[0].data());
	}

	virtual void postprocessOutput(cv::Mat &outputImage)
	{
		// take 1st channel
		std::vector<cv::Mat> outputImageSplit;
		cv::split(outputImage, outputImageSplit);

		cv::normalize(outputImageSplit[1], outputImage, 1.0, 0.0, cv::NORM_MINMAX);
	}
};

struct ORTModelData
{
	std::unique_ptr<Ort::Session> session;
	std::unique_ptr<Ort::Env> env;
	std::vector<Ort::AllocatedStringPtr> inputNames;
	std::vector<Ort::AllocatedStringPtr> outputNames;
	std::vector<Ort::Value> inputTensor;
	std::vector<Ort::Value> outputTensor;
	std::vector<std::vector<int64_t>> inputDims;
	std::vector<std::vector<int64_t>> outputDims;
	std::vector<std::vector<float>> outputTensorValues;
	std::vector<std::vector<float>> inputTensorValues;
};
