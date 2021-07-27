#pragma once

namespace OpenCVRuntimeComponent
{
	public ref class FormattedInputDataAndShape sealed
	{
	public:
		FormattedInputDataAndShape(
			Windows::Foundation::Collections::IVector<float>^ inputData,
			Windows::Foundation::Collections::IVector<int>^ inputShape,
			bool isNullInput);

		property Windows::Foundation::Collections::IVector<float>^ InputData;
		property Windows::Foundation::Collections::IVector<int>^ InputShape;
		property bool IsNullInput;
	};
}