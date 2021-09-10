// Adapted from the WinML MNIST sample and Rene Schulte's repo 
// https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/MNIST
// https://github.com/reneschulte/WinMLExperiments/

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

#if ENABLE_WINMD_SUPPORT
using Windows.AI.MachineLearning;
//using Microsoft.AI.MachineLearning; // Requires .dll and .winmd files from the Microsoft.AI.MachineLearning nuget package to be copied to Assets/Plugins/ARM/
using Windows.Storage.Streams;
using Windows.Media;
using Windows.Storage;
using Windows.Graphics.Imaging;
using System.Diagnostics;
#endif


public enum ModelType
{
    Cholec80,
    SurgicalTasks
}

public struct NetworkResult
{
    public NetworkResult(int pred, string predLabel, float prob, long time)
    {
        Prediction = pred;
        PredictionLabel = predLabel;
        PredictionProbability = prob;
        PredictionTime = time;
    }

    public int Prediction { get; }
    public string PredictionLabel { get; }
    public float PredictionProbability { get; }
    public long PredictionTime { get; }
}

public class NetworkModel
{
    public float DetectionThreshold = 0.5f;

    private List<string> _labels = new List<string>();

#if ENABLE_WINMD_SUPPORT
    private CustomNetworkModel _customNetworkModel;
    private CustomNetworkInput _customNetworkInput = new CustomNetworkInput();
    private CustomNetworkOutput _customNetworkOutput = new CustomNetworkOutput();
#endif 

    /// <summary>
    /// Asynchronously load the onnx model from Visual Studio assets folder 
    /// </summary>
    /// <returns></returns>
    public async Task LoadModelAsync(string modelFileName, string labelsFileName)
    {
        try
        {
            // Parse imagenet labels from label json file
            // https://github.com/reneschulte/WinMLExperiments/
            var labelsTextAsset = Resources.Load(labelsFileName) as TextAsset;
            using (var streamReader = new StringReader(labelsTextAsset.text))
            {
                string line = "";
                char[] charToTrim = { '\"', ' ' };
                while (streamReader.Peek() >= 0)
                {
                    line = streamReader.ReadLine();
                    line.Trim(charToTrim);
                    var indexAndLabel = line.Split(':');
                    if (indexAndLabel.Count() == 2)
                    {
                        _labels.Add(indexAndLabel[1]);
                    }
                }
            }

#if ENABLE_WINMD_SUPPORT
            // Load onnx model from Visual studio assets folder, build VS project in Unity
            // then add the onnx model to the Assets folder in visual studio solution
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/" + modelFileName));
            _customNetworkModel = await CustomNetworkModel.CreateFromStreamAsync(modelFile as IRandomAccessStreamReference);
            UnityEngine.Debug.Log("LoadModelAsync: Onnx model loaded successfully.");
#endif 
        }

        catch
        {
#if ENABLE_WINMD_SUPPORT
            _customNetworkModel = null;
            UnityEngine.Debug.Log("LoadModelAsync: Onnx model failed to load.");
#endif 
            throw;
        }
    }

#if ENABLE_WINMD_SUPPORT
    public async Task<NetworkResult> EvaluateTensorFloatAsync(
        OpenCVRuntimeComponent.FormattedInputDataAndShape formattedInputDataAndShape)
    {
        // Sometimes on HL RS4 the D3D surface returned is null, so simply skip those frames
        if (_customNetworkModel == null)
        {
            UnityEngine.Debug.Log("EvaluateVideoFrameAsync: No detection, null frame or model not initialized.");
            return new NetworkResult(-1, "None", 0f, 0); ;
        }

        // Cache the input video frame to network input
        var inputTensor = TensorFloat.CreateFromIterable(
            formattedInputDataAndShape.InputShape.Select(i => (long)i).ToList(),
            formattedInputDataAndShape.InputData);
        _customNetworkInput.features = inputTensor;

        // Perform network model inference using the input data tensor, cache output and time operation
        var stopwatch = Stopwatch.StartNew();
        _customNetworkOutput = await _customNetworkModel.EvaluateAsync(_customNetworkInput);
        stopwatch.Stop();

        // Convert prediction to datatype
        var outVec = _customNetworkOutput.prediction.GetAsVectorView().ToList();

        // LINQ query to check for highest probability digit
        if (outVec.Max() > DetectionThreshold)
        {
            // Get the index of max probability value
            var maxProb = outVec.Max();
            var maxIndex = outVec.IndexOf(maxProb);

            UnityEngine.Debug.Log($"EvaluateVideoFrameAsync: Prediction [{_labels[maxIndex]}] prob: [{maxProb * 100} %] time: [{stopwatch.ElapsedMilliseconds} ms]");

            // Return the detections
            return new NetworkResult(maxIndex, _labels[maxIndex], maxProb, stopwatch.ElapsedMilliseconds);
        }
        else
        {
            return new NetworkResult(-1, "No prediction exceeded probability threshold.", 0f, stopwatch.ElapsedMilliseconds); ;
        }
    }
#endif
}