#include "BgBlur.h"

#include <util\platform.h>

#include <filesystem>
#include <wchar.h>
#include <windows.h>

#include "Models.h"

#include "FilterData.h"

#include <opencv2/imgcodecs.hpp>

/*static*/
bool BgBlurGraphics::runFilterModelInference(FilterData *tf, const cv::Mat &imageBGRA, cv::Mat &output)
{
	// Preprocesses a BGRA video frame, letterboxes to preserve aspect ratio for the network,
	// runs inference, then removes padding and maps the mask back to the original size.

	if (tf->session.get() == nullptr || tf->model.get() == nullptr)
		return false;

	cv::Mat imageRGB;
	cv::cvtColor(imageBGRA, imageRGB, cv::COLOR_BGRA2RGB);

	uint32_t inputWidth = 0, inputHeight = 0;
	tf->model->getNetworkInputSize(tf->inputDims, inputWidth, inputHeight);

	//Aspect-ratio preserving resize with padding (inline of resizeWithPadding)
	const int srcWidth = imageRGB.cols;
	const int srcHeight = imageRGB.rows;
	const float srcAspect = static_cast<float>(srcWidth) / static_cast<float>(srcHeight);
	const float dstAspect = static_cast<float>(inputWidth) / static_cast<float>(inputHeight);

	cv::Mat dst;
	int fittedW = 0, fittedH = 0, padX = 0, padY = 0;

	if (fabs(srcAspect - dstAspect) < 1e-3f) // treat as equal
	{
		// Aspect ratios match: resize directly, no padding
		fittedW = static_cast<int>(inputWidth);
		fittedH = static_cast<int>(inputHeight);
		cv::resize(imageRGB, dst, cv::Size(fittedW, fittedH));
		padX = 0;
		padY = 0;
	}
	else if (srcAspect > dstAspect)
	{
		// Fit by width
		fittedW = static_cast<int>(inputWidth);
		fittedH = std::max(1, static_cast<int>(static_cast<float>(inputWidth) / srcAspect));
		cv::resize(imageRGB, dst, cv::Size(fittedW, fittedH));
		padX = 0;
		padY = (inputHeight - fittedH) / 2;
	}
	else
	{
		// Fit by height
		fittedH = static_cast<int>(inputHeight);
		fittedW = std::max(1, static_cast<int>(static_cast<float>(inputHeight) * srcAspect));
		cv::resize(imageRGB, dst, cv::Size(fittedW, fittedH));
		padY = 0;
		padX = (inputWidth - fittedW) / 2;
	}

	// Create padded RGB image the network expects
	cv::Mat resizedImageRGB(static_cast<int>(inputHeight), static_cast<int>(inputWidth), imageRGB.type(), cv::Scalar(0, 0, 0));
	dst.copyTo(resizedImageRGB(cv::Rect(padX, padY, dst.cols, dst.rows)));

	//cv::imwrite("C:\\Users\\srogers\\Desktop\\input.png", resizedImageRGB);

	cv::Mat resizedImage32F, preprocessedImage;
	resizedImageRGB.convertTo(resizedImage32F, CV_32F);
	tf->model->prepareInputToNetwork(resizedImage32F, preprocessedImage);
	tf->model->loadInputToTensor(preprocessedImage, inputWidth, inputHeight, tf->inputTensorValues);

	
		auto tbefore = ::clock();

	tf->model->runNetworkInference(tf->session, tf->inputNames, tf->outputNames, tf->inputTensor, tf->outputTensor);

		printf("took %dms\n", ::clock() - tbefore);

	cv::Mat outputImage = tf->model->getNetworkOutput(tf->outputDims, tf->outputTensorValues);

	tf->model->assignOutputToInput(tf->outputTensorValues, tf->inputTensorValues);
	tf->model->postprocessOutput(outputImage); // typically CV_32F mask in [0..1]

	fittedW = std::max(1, std::min(fittedW, static_cast<int>(inputWidth)));
	fittedH = std::max(1, std::min(fittedH, static_cast<int>(inputHeight)));
	padX = std::max(0, std::min(padX, static_cast<int>(inputWidth) - fittedW));
	padY = std::max(0, std::min(padY, static_cast<int>(inputHeight) - fittedH));

	cv::Rect contentRoi(padX, padY, fittedW, fittedH);
	cv::Mat croppedMask = outputImage(contentRoi);

	cv::Mat maskAtOriginalSize;
	const int interp = cv::INTER_LINEAR; // use INTER_NEAREST if mask is already binary
	cv::resize(croppedMask, maskAtOriginalSize, cv::Size(srcWidth, srcHeight), 0, 0, interp);

	maskAtOriginalSize.convertTo(output, CV_8U, 255.0);

	// Debug: print out bytes and dimensions
	//printf("output image size: %dx%d, channels=%d, bytes=%zu\n", maskAtOriginalSize.cols, maskAtOriginalSize.rows, maskAtOriginalSize.channels(), maskAtOriginalSize.total() * maskAtOriginalSize.elemSize());

	// ---- 9) Save mask to file
	//if (!cv::imwrite("C:\\Users\\srogers\\Desktop\\mask.png", output))
	//	printf("Failed to save mask image!\n");

	return true;
}


/*static*/
bool BgBlurGraphics::getRGBAFromStageSurface(FilterData *tf, uint32_t &width, uint32_t &height)
{
	// Captures a live video frame from a source, renders it to a texture, transfers it onto
	//	a staging surface, maps it into CPU-accessible memory, then it wraps the pixel buffer into an OpenCV cv::Mat (BGRA format)

	if (!obs_source_enabled(tf->source))
		return false;

	obs_source_t *target = obs_filter_get_target(tf->source);

	if (!target)
		return false;

	width = obs_source_get_base_width(target);
	height = obs_source_get_base_height(target);

	if (width == 0 || height == 0)
		return false;

	gs_texrender_reset(tf->texrender);

	if (!gs_texrender_begin(tf->texrender, width, height))
		return false;

	struct vec4 background;
	vec4_zero(&background);
	gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
	gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
	gs_blend_state_push();
	gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
	obs_source_video_render(target);
	gs_blend_state_pop();
	gs_texrender_end(tf->texrender);

	if (tf->stagesurface)
	{
		uint32_t stagesurf_width = gs_stagesurface_get_width(tf->stagesurface);
		uint32_t stagesurf_height = gs_stagesurface_get_height(tf->stagesurface);

		if (stagesurf_width != width || stagesurf_height != height)
		{
			gs_stagesurface_destroy(tf->stagesurface);
			tf->stagesurface = nullptr;
		}
	}

	if (!tf->stagesurface)
		tf->stagesurface = gs_stagesurface_create(width, height, GS_BGRA);

	gs_stage_texture(tf->stagesurface, gs_texrender_get_texture(tf->texrender));

	uint8_t *video_data;
	uint32_t linesize;

	if (!gs_stagesurface_map(tf->stagesurface, &video_data, &linesize))
		return false;

	{
		std::lock_guard<std::mutex> lock(tf->inputBGRALock);
		tf->inputBGRA = cv::Mat(height, width, CV_8UC4, video_data, linesize);
	}

	gs_stagesurface_unmap(tf->stagesurface);
	return true;
}

/*static*/
gs_texture_t* BgBlurGraphics::blurBackground(FilterData *tf, uint32_t width, uint32_t height, gs_texture_t *alphaTexture)
{
	if (tf->blurBackground == 0 || !tf->kawaseBlurEffect)
		return nullptr;

	gs_texture_t *blurredTexture = gs_texture_create(width, height, GS_BGRA, 1, nullptr, 0);
	gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	gs_eparam_t *image = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "image");
	gs_eparam_t *focalmask = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "focalmask");
	gs_eparam_t *xOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "xOffset");
	gs_eparam_t *yOffset = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "yOffset");
	gs_eparam_t *blurIter = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "blurIter");
	gs_eparam_t *blurTotal = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "blurTotal");
	gs_eparam_t *blurFocusPointParam = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "blurFocusPoint");
	gs_eparam_t *blurFocusDepthParam = gs_effect_get_param_by_name(tf->kawaseBlurEffect, "blurFocusDepth");

	for (int i = 0; i < (int)tf->blurBackground; i++)
	{
		gs_texrender_reset(tf->texrender);

		if (!gs_texrender_begin(tf->texrender, width, height))
		{
			blog(LOG_INFO, "BgBlurGraphics::blurBackground - Could not open background blur texrender!");
			return blurredTexture;
		}

		gs_effect_set_texture(image, blurredTexture);
		gs_effect_set_texture(focalmask, alphaTexture);
		gs_effect_set_float(xOffset, ((float)i + 0.5f) / (float)width);
		gs_effect_set_float(yOffset, ((float)i + 0.5f) / (float)height);
		gs_effect_set_int(blurIter, i);
		gs_effect_set_int(blurTotal, (int)tf->blurBackground);
		gs_effect_set_float(blurFocusPointParam, tf->blurFocusPoint);
		gs_effect_set_float(blurFocusDepthParam, tf->blurFocusDepth);

		struct vec4 background;
		vec4_zero(&background);
		gs_clear(GS_CLEAR_COLOR, &background, 0.0f, 0);
		gs_ortho(0.0f, static_cast<float>(width), 0.0f, static_cast<float>(height), -100.0f, 100.0f);
		gs_blend_state_push();
		gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

		const char *blur_type = "Draw";

		while (gs_effect_loop(tf->kawaseBlurEffect, blur_type))
			gs_draw_sprite(blurredTexture, 0, width, height);
		
		gs_blend_state_pop();
		gs_texrender_end(tf->texrender);
		gs_copy_texture(blurredTexture, gs_texrender_get_texture(tf->texrender));
	}

	return blurredTexture;
}

/*static*/
int BgBlurGraphics::createOrtSession(FilterData *tf)
{
	if (tf->model.get() == nullptr)
	{
		printf("BgBlur::createOrtSession null model\n");
		blog(LOG_ERROR, "BgBlur::createOrtSession null model");
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_INVALID_MODEL;
	}

	Ort::SessionOptions sessionOptions;
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

	//if (tf->useGPU != USEGPU_CPU)
	//{
	//	sessionOptions.DisableMemPattern();
	//	sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
	//}
	//else
	{
		sessionOptions.SetInterOpNumThreads(1);
		sessionOptions.SetIntraOpNumThreads(1);
	}

	auto modelFilepath = (std::filesystem::path(obs_get_module_binary_path(obs_current_module())).parent_path() / tf->modelSelection);

	if (!std::filesystem::exists(modelFilepath))
	{
		printf("tf->modelSelection not found at\n");
		blog(LOG_ERROR, "tf->modelSelection not found at %s", modelFilepath.string().c_str());
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_FILE_NOT_FOUND;
	}

	tf->modelFilepath = modelFilepath.wstring();

	try
	{
		tf->session = std::make_unique<Ort::Session>(*tf->env, tf->modelFilepath.c_str(), sessionOptions);
	}
	catch (const std::exception &e)
	{
		printf("%s\n", e.what());
		blog(LOG_ERROR, "%s", e.what());
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_STARTUP;
	}

	Ort::AllocatorWithDefaultOptions allocator;

	tf->model->populateInputOutputNames(tf->session, tf->inputNames, tf->outputNames);

	if (!tf->model->populateInputOutputShapes(tf->session, tf->inputDims, tf->outputDims))
	{
		printf("Unable to get model input and output shapes\n");
		blog(LOG_ERROR, "Unable to get model input and output shapes");
		return OBS_BGREMOVAL_ORT_SESSION_ERROR_INVALID_INPUT_OUTPUT;
	}

	// Allocate buffers
	tf->model->allocateTensorBuffers(tf->inputDims, tf->outputDims, tf->outputTensorValues, tf->inputTensorValues, tf->inputTensor, tf->outputTensor);
	return OBS_BGREMOVAL_ORT_SESSION_SUCCESS;
}
