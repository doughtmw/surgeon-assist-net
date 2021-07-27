#pragma once

namespace OpenCVRuntimeComponent
{
    public ref class CvUtils sealed
    {
    public:
        CvUtils();
        FormattedInputDataAndShape^ SoftwareBitmapToPreprocessedTensorFloat(Windows::Graphics::Imaging::SoftwareBitmap^ softwareBitmap, int image_dim);

    private:
        void WrapHoloLensSoftwareBitmapWithCvMat(Windows::Graphics::Imaging::SoftwareBitmap^ softwareBitmap, cv::Mat& openCVImage);
        Windows::Graphics::Imaging::SoftwareBitmap^ WrapCvMatWithHoloLensSensorFrame(cv::Mat& from);
        unsigned char* GetPointerToPixelData(Windows::Foundation::IMemoryBufferReference^ reference);
        void ResizeAndNormalizeMatForPyTorch(cv::Size image_dim, cv::Mat src, cv::Mat& dst, cv::Scalar norm_mean, cv::Scalar norm_std);
        void WinVectorToCvMat(Windows::Foundation::Collections::IVector<float>^ vec, int dim_1, int dim_2, int dim_3, cv::Mat& mat);
        std::vector<float> transposeOpencvMat(cv::Mat srcMat, cv::Size image_size);

    };
}
