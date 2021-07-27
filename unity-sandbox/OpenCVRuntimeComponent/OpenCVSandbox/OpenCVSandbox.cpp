// OpenCVSandbox.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <onnxruntime_cxx_api.h>


void floatArray2opencvMat3D(float* ptr, int dim_1, int dim_2, int dim_3, cv::Mat& out)
{
	int size[3] = { dim_1, dim_2, dim_3 };
	out = cv::Mat(3, size, CV_32F, ptr).clone();
}

void floatArray2opencvMat(float* ptr, int dim_1, int dim_2, cv::Mat& out)
{
	int size[2] = { dim_1, dim_2 };
	out = cv::Mat(2, size, CV_32F, ptr).clone();
}

void NormalizeMatForPyTorch(cv::Mat src, cv::Mat& dst, cv::Scalar norm_mean, cv::Scalar norm_std)
{
	// Resize image
	cv::resize(src, dst, cv::Size(256, 256), cv::INTER_NEAREST);

	// Convert the matrix format (avoid rounding errors)
	dst.convertTo(dst, CV_32F);

	// Center crop the opencv matrices from (256, 256) to (224, 224) for input to the network
	int wh = 224;
	int wh_c = 128;
	int x = wh_c - (wh / 2); // 16
	int y = wh_c - (wh / 2); // 16
	dst(cv::Rect(x, y, wh, wh)).copyTo(dst);

	// Transform the image (copy the ToTensor behaviour)
	dst /= 255.0f;

	// Normalize with parameters
	dst -= norm_mean;
	dst /= norm_std;
}

// https://github.com/microsoft/onnxruntime/issues/3310
// Create expected ORT format (channel blocks by row): https://answers.opencv.org/question/64837
// [512, 512] -> [3, 512, 512]
std::vector<float> transposeOpencvMat(
	cv::Mat srcMat,
	cv::Mat& dstMat,
	cv::Size image_size)
{
	// Specify the input tensor size
	std::vector<float> imgArray;
	std::array<int64_t, 4> input_shape_{ 1, 3, image_size.height, image_size.width };
	const size_t input_tensor_size = image_size.height * image_size.width * 3;

	// Split the matrix into channels and push each channel 
	// into the vector of floats
	std::vector<cv::Mat> channels;
	cv::split(srcMat, channels);
	cv::Mat image_by_channel;
	for (size_t i = 0; i < channels.size(); i++)
		image_by_channel.push_back(channels[i]);

	if (!image_by_channel.isContinuous())
		image_by_channel = image_by_channel.clone();

	for (size_t i = 0; i < input_tensor_size; i++)
		imgArray.push_back(image_by_channel.at<float>(i));

	// Convert vector of floats to an opencv mat [3, 512, 512]
	// and return 
	floatArray2opencvMat3D(imgArray.data(), 3, image_size.height, image_size.width, dstMat);

	return imgArray;
}

/// <summary>
/// https://github.com/microsoft/onnxruntime/issues/2832
/// https://github.com/microsoft/onnxruntime/blob/521dc757984fbf9770d0051997178fbb9565cd52/samples/c_cxx/MNIST/MNIST.cpp#L38
/// </summary>
/// <param name="imgRGB"></param>
/// <returns></returns>
bool runInference(
	int n_classes,
	const wchar_t onnx_model_path[],
	const cv::Mat& imgRGB, cv::Size image_size,
	std::array<int64_t, 4> input_shape_,
	cv::Mat& output_mat)
{
	// Create the environment and load model from string
	Ort::Env env{ ORT_LOGGING_LEVEL_WARNING, "test" };
	Ort::Session session_
	{
		env,
		onnx_model_path,
		Ort::SessionOptions{nullptr}
	};

	// Define the input and outputs
	std::vector<const char*> input_names = { "input" };
	std::vector<const char*> output_names = { "output" };

	// Specify the input tensor size
	const size_t input_tensor_size = image_size.height * image_size.width * 3;

	// Convert opencv CHW (3, 512, 512) matrix to vector of floats for input tensor
	std::vector<float> imgArray;
	int height = image_size.height;
	int width = image_size.width;
	int channels = 3;
	for (int c = 0; c < channels; ++c)
	{
		for (int i = 0; i < height; ++i)
		{
			for (int j = 0; j < width; ++j)
			{
				imgArray.push_back(imgRGB.at<float>(c, i, j));
			}
		}
	}

	// Create the input and output tensors
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		imgArray.data(),
		input_tensor_size,
		input_shape_.data(), 4);
	assert(input_tensor.IsTensor());

	// score model & input tensor, get back output tensor
	auto output_tensors = session_.Run(
		Ort::RunOptions{ nullptr },
		input_names.data(),
		&input_tensor, 1,
		output_names.data(), 1);
	assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

	float* outputs = output_tensors.at(0).GetTensorMutableData<float>();
	floatArray2opencvMat(outputs, 1, n_classes, output_mat);

	return true;
}

/// <summary>
/// Compare this sample to the scratchpad.py script to ensure that the prediction results
/// for PyTorch versus OpenCV in python and c++ are similar
/// </summary>
/// <returns></returns>
int main()
{
	// Change these parameters before running on your system
	// Currently testing with (1, 3, 224, 224) sized input sequence
	cv::Size image_size(224, 224);
	std::array<int64_t, 4> input_shape_{ 1, 3, image_size.height, image_size.width };
	cv::String input_image = "C:/git/public/surgeon-assist-net/pytorch-sandbox/onnx-models/video41_25.jpg";
	cv::String normalized_image = "C:/git/public/surgeon-assist-net/pytorch-sandbox/onnx-models/video41_25-opencv-transforms-cpp.jpg";
	const wchar_t* onnx_model = L"C:/git/public/surgeon-assist-net/pytorch-sandbox/onnx-models/b0_lite_1.onnx";

	// Load a sample image
	cv::Mat image = cv::imread(input_image);
	cv::imshow("image", image);

	// Convert the color channel orientation from BGR to RGB
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	// For pytorch custom phantom data weights
	cv::Scalar norm_mean(0.40521515692759497, 0.27927462047480039, 0.27426218748099274);
	cv::Scalar norm_std(0.20460533490722591, 0.17244239120062696, 0.16623196974782356);

	// Run
	cv::Mat openCvImage;
	NormalizeMatForPyTorch(image, openCvImage, norm_mean, norm_std);

	// show results
	cv::imshow("openCvImage", openCvImage);
	cv::imwrite(normalized_image, openCvImage);

	double min, max;
	minMaxLoc(openCvImage, &min, &max);
	cv::Scalar result = mean(openCvImage);
	double avg = (result.val[0] + result.val[1] + result.val[2]) / 3;
	std::cout << "openCvImage.shape = " << openCvImage.size() << std::endl;
	std::cout << "openCvImage max = " << max << std::endl;
	std::cout << "openCvImage min = " << min << std::endl;
	std::cout << "openCvImage avg = " << avg << std::endl;
	std::cout << "openCvImage" << std::endl;

	int k = cv::waitKey(0); // Wait for a keystroke in the window

	// WC -> CHW format [STUCK]
	cv::Mat openCvImageCHW;
	std::cout << "openCvImage.shape = " << openCvImage.size << std::endl;
	auto floatCHW = transposeOpencvMat(openCvImage, openCvImageCHW, image_size);

	// Perform inference on loaded data
	cv::Mat outputs;
	int n_classes = 7;
	bool isRun = runInference(
		n_classes,
		onnx_model,
		openCvImageCHW,
		image_size,
		input_shape_,
		outputs);

	// First image of test dataset video41_25, want:
	// local_output: tensor([[ 0.1532,  5.9982, -0.3814, -4.1019,  0.9841,  1.8995, -4.3837]], device = 'cuda:0')

	// Get:
	// outputs: [[0.15243995, 5.9982438, -0.38196307, -4.1016378, 0.98468649, 1.8998184, -4.383872]]
	std::cout << "\noutputs: [" << outputs << "]" << std::endl;
}
