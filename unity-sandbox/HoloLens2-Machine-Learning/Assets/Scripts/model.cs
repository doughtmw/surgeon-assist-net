// Adapted from the WinML MNIST sample 
// https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/MNIST

#if ENABLE_WINMD_SUPPORT 
using System;
using System.Threading.Tasks;
using Windows.Storage.Streams;
using Windows.AI.MachineLearning;
//using Microsoft.AI.MachineLearning; // Requires .dll and .winmd files from the Microsoft.AI.MachineLearning nuget package to be copied to Assets/Plugins/ARM/

public sealed class CustomNetworkInput
{
    public TensorFloat features; // (3, 224, 224)
}

public sealed class CustomNetworkOutput
{
    public TensorFloat prediction; // Cholec80: (7) OR SurgicalTasks: (5)
}

public sealed class CustomNetworkModel
{
    private LearningModel model;
    private LearningModelSession session;
    private LearningModelBinding binding;
    public static async Task<CustomNetworkModel> CreateFromStreamAsync(IRandomAccessStreamReference stream)
    {
        // Run on the CPU
        var device = new LearningModelDevice(LearningModelDeviceKind.Default);

        CustomNetworkModel learningModel = new CustomNetworkModel();
        learningModel.model = await LearningModel.LoadFromStreamAsync(stream);
        learningModel.session = new LearningModelSession(learningModel.model, device);
        learningModel.binding = new LearningModelBinding(learningModel.session);
        return learningModel;
    }

    public async Task<CustomNetworkOutput> EvaluateAsync(CustomNetworkInput input)
    {
        // Ensure the input and output fields are bound to the correct
        // layer names in the onnx model
        binding.Bind("input", input.features);
        var result = await session.EvaluateAsync(binding, "0");
        var output = new CustomNetworkOutput();
        output.prediction = result.Outputs["output"] as TensorFloat;
        return output;
    }
}

#endif
