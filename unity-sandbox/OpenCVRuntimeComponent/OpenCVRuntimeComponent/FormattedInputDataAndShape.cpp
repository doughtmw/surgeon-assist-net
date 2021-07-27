#include "pch.h"
#include "FormattedInputDataAndShape.h"

OpenCVRuntimeComponent::FormattedInputDataAndShape::FormattedInputDataAndShape(
	Windows::Foundation::Collections::IVector<float>^ inputData,
	Windows::Foundation::Collections::IVector <int>^ inputShape,
	bool isNullInput)
{
	InputData = inputData;
	InputShape = inputShape;
	IsNullInput = isNullInput;
}
