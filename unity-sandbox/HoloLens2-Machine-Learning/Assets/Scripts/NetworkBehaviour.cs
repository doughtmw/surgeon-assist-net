// Adapted from the WinML MNIST sample and Rene Schulte's repo 
// https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/MNIST
// https://github.com/reneschulte/WinMLExperiments/

using System;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UI;

#if ENABLE_WINMD_SUPPORT
using Windows.Graphics.Imaging;
#endif 

public class NetworkBehaviour : MonoBehaviour
{
    // Public fields
    public ModelType ModelType;
    public MediaCaptureUtility.MediaCaptureProfiles MediaCaptureProfiles;
    public MediaCaptureUtility.NetworkImageDimension NetworkImageDimension;
    public float ProbabilityThreshold = 0.5f;
    public Text StatusBlock;

    // Private fields
    private NetworkModel _networkModel;
    private MediaCaptureUtility _mediaCaptureUtility;
    private bool _isRunning = false;
    private int _numClasses = 0;
    private int _image_dim = 0;

#if ENABLE_WINMD_SUPPORT
    // OpenCV winrt dll 
    OpenCVRuntimeComponent.CvUtils CvUtils = new OpenCVRuntimeComponent.CvUtils();
#endif

    #region UnityMethods
    async void Start()
    {

        switch (NetworkImageDimension)
        {
            case MediaCaptureUtility.NetworkImageDimension._224x224:
                _image_dim = 224;
                break;
        }

        // Set the data source from labels type enum
        switch (ModelType)
        {
            case ModelType.Cholec80:
                _numClasses = 7;
                break;
            case ModelType.SurgicalTasks:
                _numClasses = 5;
                break;
            default:
                break;
        }
        try
        {
            // Create a new instance of the network model class
            // and asynchronously load the onnx model
            _networkModel = new NetworkModel();
            switch (ModelType)
            {
                case ModelType.Cholec80:
                    await _networkModel.LoadModelAsync("b0_lite_1.onnx", "Cholec80Labels.json");
                    break;
                case ModelType.SurgicalTasks:
                    await _networkModel.LoadModelAsync("model_custom.onnx", "SurgicalTaskLabels.json");
                    break;
                default:
                    break;
            }
            StatusBlock.text = $"Loaded model. Starting camera...";

#if ENABLE_WINMD_SUPPORT
            // Configure camera to return frames fitting the model input size
            try
            {
                Debug.Log("Creating MediaCaptureUtility and initializing frame reader.");
                _mediaCaptureUtility = new MediaCaptureUtility();
                await _mediaCaptureUtility.InitializeMediaFrameReaderAsync(MediaCaptureProfiles);
                StatusBlock.text = $"Camera started. Running!";

                Debug.Log("Successfully initialized frame reader.");
            }
            catch (Exception ex)
            {
                StatusBlock.text = $"Failed to start camera: {ex.Message}. Using loaded/picked image.";
            }

            // Run processing loop in separate parallel Task, get the latest frame
            // and asynchronously evaluate
            Debug.Log("Begin performing inference in frame grab loop.");

            _isRunning = true;
            await Task.Run(async () =>
            {
                while (_isRunning)
                {
                    if (_mediaCaptureUtility.IsCapturing)
                    {
                        using (var videoFrame = _mediaCaptureUtility.GetLatestFrame())
                        {
                            await EvaluateFrame(videoFrame, _image_dim);
                        }
                    }
                    else
                    {
                        return;
                    }
                }
            });
#endif 
        }
        catch (Exception ex)
        {
            StatusBlock.text = $"Error init: {ex.Message}";
            Debug.LogError($"Failed to start model inference: {ex}");
        }
    }

    private async void OnDestroy()
    {
        _isRunning = false;
        if (_mediaCaptureUtility != null)
        {
            await _mediaCaptureUtility.StopMediaFrameReaderAsync();
        }
    }
    #endregion

#if ENABLE_WINMD_SUPPORT
    private async Task EvaluateFrame(SoftwareBitmap softwareBitmap, int image_dim)
    {
        try
        {
            // Format the current software bitmap (rescale and crop) using OpenCV helper
            var formattedInputDataAndShape = CvUtils.SoftwareBitmapToPreprocessedTensorFloat(
                softwareBitmap,
                image_dim);

            if (!formattedInputDataAndShape.IsNullInput)
            {
                // Get the current network prediction from model and input frame
                var result = await _networkModel.EvaluateTensorFloatAsync(
                    formattedInputDataAndShape);

                // Update the UI with prediction
                UnityEngine.WSA.Application.InvokeOnAppThread(() =>
                {
                    StatusBlock.text = $"Label: {result.PredictionLabel} " +
                    $"Probability: {Math.Round(result.PredictionProbability, 3) * 10}% " +
                    $"Inference time: {result.PredictionTime} ms";
                }, false);
            }
            else
            {
                Debug.Log("Received null TensorFloat.");
            }
        }
        catch (Exception ex)
        {
            Debug.Log($"Exception {ex}");
        }
    }

#endif
}