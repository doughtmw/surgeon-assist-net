#include "pch.h"
#include "CvUtils.h"
#include "Trace.h"
#include "BufferHelpers.h"
#include "FormattedInputDataAndShape.h"

using namespace OpenCVRuntimeComponent;
using namespace Platform;

CvUtils::CvUtils(){}

FormattedInputDataAndShape^ OpenCVRuntimeComponent::CvUtils::SoftwareBitmapToPreprocessedTensorFloat(
	Windows::Graphics::Imaging::SoftwareBitmap^ softwareBitmap,
	int image_dim)
{
	cv::Size image_size(image_dim, image_dim);

	// Return empty bitmap of 224 x 224 size if null
	if (softwareBitmap == nullptr)
	{
		auto sb = ref new Windows::Graphics::Imaging::SoftwareBitmap(
			Windows::Graphics::Imaging::BitmapPixelFormat::Bgra8,
			image_dim,
			image_dim);

		dbg::trace(L"OpenCVRuntimeComponent::CvUtils::SoftwareBitmapToPreprocessedTensorFloat: Input sb is null, returning empty tensorfloat.");

		// Return empty tensorfloat
		std::vector<float> zeroFloatVecCHW = std::vector<float>();
		for (int i = 0; i < (image_dim * image_dim * 3); i++)
			zeroFloatVecCHW.push_back(1.0f);

		Windows::Foundation::Collections::IVector<int>^ zeroInputShape = ref new Platform::Collections::Vector<int>({ 1, 3, image_dim, image_dim });
		Windows::Foundation::Collections::IVector<float>^ zeroInputData = ref new Platform::Collections::Vector<float>(std::move(zeroFloatVecCHW));
		return ref new FormattedInputDataAndShape(zeroInputData, zeroInputShape, true);
	}

	// Convert incoming bitmap to OpenCV mat
	cv::Mat wrappedSoftwareBitmap;
	WrapHoloLensSoftwareBitmapWithCvMat(softwareBitmap, wrappedSoftwareBitmap);
	dbg::trace(L"OpenCVRuntimeComponent::CvUtils::SoftwareBitmapToPreprocessedTensorFloat: Wrapped software bitmap with opencv mat.");

	// Convert the color channel orientation from BGR to RGB
	cv::cvtColor(wrappedSoftwareBitmap, wrappedSoftwareBitmap, cv::COLOR_BGR2RGB);
	dbg::trace(L"OpenCVRuntimeComponent::CvUtils::SoftwareBitmapToPreprocessedTensorFloat: BGR to RGB color channel conversion.");

	// For pytorch custom phantom data weights
	cv::Vec3f norm_mean(0.40521515692759497, 0.27927462047480039, 0.27426218748099274);
	cv::Vec3f norm_std(0.20460533490722591, 0.17244239120062696, 0.16623196974782356);

	// Normalize the incoming opencv mat
	cv::Mat openCvImage;
	ResizeAndNormalizeMatForPyTorch(image_size, wrappedSoftwareBitmap, openCvImage, norm_mean, norm_std);
	dbg::trace(L"OpenCVRuntimeComponent::CvUtils::SoftwareBitmapToPreprocessedTensorFloat: Resized, normalized with custom weights and zero padded.");

	// HW -> CHW format for correct predictions by network
	// Convert from [512, 512] to [3, 512, 512]
	std::vector<float> floatVecCHW = transposeOpencvMat(openCvImage, image_size);
	dbg::trace(L"OpenCVRuntimeComponent::CvUtils::SoftwareBitmapToPreprocessedTensorFloat: Converted HW -> CHW format.");

	// Create a tensorfloat from the vector
	// Batch support sample: https://github.com/microsoft/Windows-Machine-Learning/blob/master/Samples/BatchSupport/BatchSupport/SampleHelper.cpp
	Windows::Foundation::Collections::IVector<int>^ inputShape = ref new Platform::Collections::Vector<int>({ 1, 3, image_dim, image_dim });
	Windows::Foundation::Collections::IVector<float>^ inputData = ref new Platform::Collections::Vector<float>(std::move(floatVecCHW));
	//auto inputTensorFloat = Windows::AI::MachineLearning::TensorFloat::CreateFromIterable(
	//	inputShape,
	//	inputData);

	return ref new FormattedInputDataAndShape(inputData, inputShape, false);
}

// Taken directly from the OpenCVHelpers in HoloLensForCV repo.
// https://github.com/microsoft/HoloLensForCV
void CvUtils::WrapHoloLensSoftwareBitmapWithCvMat(
	Windows::Graphics::Imaging::SoftwareBitmap^ softwareBitmap,
	cv::Mat& wrappedImage)
{
	// Confirm that the sensor frame is not null
	if (softwareBitmap != nullptr)
	{
		Windows::Graphics::Imaging::BitmapBuffer^ bitmapBuffer =
			softwareBitmap->LockBuffer(
				Windows::Graphics::Imaging::BitmapBufferAccessMode::Read);

		uint32_t pixelBufferDataLength = 0;

		uint8_t* pixelBufferData =
			Io::GetTypedPointerToMemoryBuffer<uint8_t>(
				bitmapBuffer->CreateReference(),
				pixelBufferDataLength);

		int32_t wrappedImageType;

		switch (softwareBitmap->BitmapPixelFormat)
		{
		case Windows::Graphics::Imaging::BitmapPixelFormat::Bgra8:
			wrappedImageType = CV_8UC4;
			dbg::trace(
				L"WrapHoloLensSensorFrameWithCvMat: CV_8UC4 pixel format");
			break;

		case Windows::Graphics::Imaging::BitmapPixelFormat::Gray16:
			wrappedImageType = CV_16UC1;
			dbg::trace(
				L"WrapHoloLensSensorFrameWithCvMat: CV_16UC1 pixel format");
			break;

		case Windows::Graphics::Imaging::BitmapPixelFormat::Gray8:
			wrappedImageType = CV_8UC1;
			dbg::trace(
				L"WrapHoloLensSensorFrameWithCvMat: CV_8UC1 pixel format");
			break;

		default:
			dbg::trace(
				L"WrapHoloLensSensorFrameWithCvMat: unrecognized softwareBitmap pixel format, falling back to CV_8UC1");

			wrappedImageType = CV_8UC1;
			break;
		}

		wrappedImage = cv::Mat(
			softwareBitmap->PixelHeight,
			softwareBitmap->PixelWidth,
			wrappedImageType,
			pixelBufferData);

	}

	// Otherwise return an empty sensor frame
	else
	{
		uint8_t* pixelBufferData = new uint8_t();

		wrappedImage = cv::Mat(
			0,
			0,
			CV_8UC4,
			pixelBufferData);

		dbg::trace(
			L"WrapHoloLensSensorFrameWithCvMat: frame was null, returning empty matrix of CV_8UC4 pixel format.");
	}
}

// Wrap OpenCV Mat of type CV_8UC1 with SensorFrame.
Windows::Graphics::Imaging::SoftwareBitmap^ CvUtils::WrapCvMatWithHoloLensSensorFrame(
	cv::Mat& from)
{
	int32_t pixelHeight = from.rows;
	int32_t pixelWidth = from.cols;

	Windows::Graphics::Imaging::SoftwareBitmap^ bitmap =
		ref new Windows::Graphics::Imaging::SoftwareBitmap(
			Windows::Graphics::Imaging::BitmapPixelFormat::Bgra8,
			pixelWidth, pixelHeight,
			Windows::Graphics::Imaging::BitmapAlphaMode::Ignore);

	Windows::Graphics::Imaging::BitmapBuffer^ bitmapBuffer =
		bitmap->LockBuffer(Windows::Graphics::Imaging::BitmapBufferAccessMode::ReadWrite);

	auto reference = bitmapBuffer->CreateReference();
	unsigned char* dstPixels = GetPointerToPixelData(reference);
	memcpy(dstPixels, from.data, from.step.buf[1] * from.cols * from.rows);

	// Return a new sensor frame of photovideo type
	//HoloLensForCV::SensorFrame^ sf =
	//	ref new HoloLensForCV::SensorFrame(HoloLensForCV::SensorType::PhotoVideo, dt, bitmap);
	return bitmap;
}

// https://github.com/microsoft/Windows-universal-samples/blob/master/Samples/CameraOpenCV/shared/OpenCVBridge/OpenCVHelper.cpp
// https://stackoverflow.com/questions/34198259/winrt-c-win10-opencv-hsv-color-space-image-display-artifacts/34198580#34198580
// Get pointer to memory buffer reference. 
unsigned char* CvUtils::GetPointerToPixelData(Windows::Foundation::IMemoryBufferReference^ reference)
{
	Microsoft::WRL::ComPtr<Windows::Foundation::IMemoryBufferByteAccess> bufferByteAccess;

	reinterpret_cast<IInspectable*>(reference)->QueryInterface(IID_PPV_ARGS(&bufferByteAccess));

	unsigned char* pixels = nullptr;
	unsigned int capacity = 0;
	bufferByteAccess->GetBuffer(&pixels, &capacity);

	return pixels;
}

/// <summary>
/// Apply defined mean and std normalization parameters to the opencv mat
/// to replicate the scaling behaviour during preprocessing of data
/// with the PyTorch training approach.
/// </summary>
/// <param name="src"></param>
/// <param name="dst"></param>
/// <param name="norm_mean"></param>
/// <param name="norm_std"></param>
void OpenCVRuntimeComponent::CvUtils::ResizeAndNormalizeMatForPyTorch(
	cv::Size image_size,
	cv::Mat src, cv::Mat& dst, cv::Scalar norm_mean, cv::Scalar norm_std)
{
	// Resize image
	resize(src, dst, cv::Size(256, 256), cv::INTER_NEAREST);

	// Convert the matrix format (avoid rounding errors)
	dst.convertTo(dst, CV_32F);

	// Center crop the opencv matrices from (256, 256) to (224, 224) for input to the network
	int wh = image_size.width;
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

/// <summary>
/// https://social.msdn.microsoft.com/Forums/en-US/f18048c4-5d13-47f6-bf4e-c2cbceab256e/quickest-way-to-convert-platformcollectionsvector-to-a-stdvector?forum=winappswithnativecode
/// https://stackoverflow.com/questions/48207847/how-to-create-cvmat-from-buffer-array-of-t-data-using-a-template-function/48207940#48207940
/// </summary>
void OpenCVRuntimeComponent::CvUtils::WinVectorToCvMat(
	Windows::Foundation::Collections::IVector<float>^ vec,
	int dim_1, int dim_2, int dim_3,
	cv::Mat& mat)
{
	// Convert foundation collections vector to std vector
	std::vector<float> std_vector =
		Windows::Foundation::Collections::to_vector(vec);

	// Get pointer to data
	float* ptr = std_vector.data();

	// Allocate the matrix from pointer
	//mat = cv::Mat(data_row, data_col, CV_32F, ptr).clone();
	int size[3] = { dim_1, dim_2, dim_3 };
	mat = cv::Mat(3, size, CV_32F, ptr).clone();
}

std::vector<float> OpenCVRuntimeComponent::CvUtils::transposeOpencvMat(
	cv::Mat srcMat,
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
	//floatArray2opencvMat3D(imgArray.data(), 3, image_size.height, image_size.width, dstMat);
	return imgArray;
}