// Script adapted from Rene Schulte's repo: https://github.com/reneschulte/WinMLExperiments/blob/master/HoloVision20/Assets/Scripts/MediaCapturer.cs

using System;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

#if ENABLE_WINMD_SUPPORT
using Windows.Media;
using Windows.Media.Capture;
using Windows.Storage.Streams;
using Windows.Media.Capture.Frames;
using Windows.Media.MediaProperties;
using Windows.Media.Devices;
using Windows.Graphics.Imaging;
using Windows.Devices.Enumeration;
#endif // ENABLE_WINMD_SUPPORT

public class MediaCaptureUtility
{
    public bool IsCapturing { get; set; }

    // https://docs.microsoft.com/en-us/windows/mixed-reality/develop/platform-capabilities-and-apis/locatable-camera
    public enum MediaCaptureProfiles
    {
        HL2_2272x1278,
        HL2_896x504,
    }

    public enum NetworkImageDimension
    {
        _224x224,
    }

#if ENABLE_WINMD_SUPPORT
    private MediaCapture _mediaCapture;
    private MediaFrameReader _mediaFrameReader;

    /// <summary>
    /// Method to start media frame reader at desired resolution.
    /// </summary>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <returns></returns>
    public async Task InitializeMediaFrameReaderAsync(MediaCaptureProfiles mediaCaptureProfiles)
    {
        // Check state of media capture object 
        if (_mediaCapture == null || _mediaCapture.CameraStreamState == CameraStreamState.Shutdown || _mediaCapture.CameraStreamState == CameraStreamState.NotStreaming)
        {
            if (_mediaCapture != null)
            {
                _mediaCapture.Dispose();
            }

            // Get the media capture description and request media capture profile
            int width = 0;
            int height = 0;
            switch (mediaCaptureProfiles)
            {
                case MediaCaptureProfiles.HL2_2272x1278:
                    width = 2272;
                    height = 1278;
                    break;
                case MediaCaptureProfiles.HL2_896x504:
                    width = 896;
                    height = 504;
                    break;
                default:
                    width = 0;
                    height = 0;
                    break;
            }

            // Create the media capture and media capture frame source from description
            var mediaCaptureAndFrameSource = await GetMediaCaptureForDescriptionAsync(
                MediaFrameSourceKind.Color, width, height, 30);

            // Convert the pixel formats
            var subtype = MediaEncodingSubtypes.Bgra8;

            // Create the media frame reader with specified description and subtype
            _mediaFrameReader = await mediaCaptureAndFrameSource.capture.CreateFrameReaderAsync(
                mediaCaptureAndFrameSource.source,
                subtype);
            _mediaFrameReader.AcquisitionMode = MediaFrameReaderAcquisitionMode.Realtime;

            await _mediaFrameReader.StartAsync();
            Debug.Log("InitializeMediaFrameReaderAsync: Successfully started media frame reader.");

            IsCapturing = true;
        }
    }

    /// <summary>
    /// Retrieve the latest video frame from the media frame reader
    /// </summary>
    /// <returns>VideoFrame object with current frame's software bitmap</returns>
    public SoftwareBitmap GetLatestFrame()
    //public VideoFrame GetLatestFrame()
    {
        // The overloads of CreateFrameReaderAsync with the format arguments will actually return a copy so we don't have to copy again
        var mediaFrameReference = _mediaFrameReader.TryAcquireLatestFrame();
        var softwareBitmap = mediaFrameReference?.VideoMediaFrame?.SoftwareBitmap;
        Debug.Log("GetLatestFrame: Successfully retrieved software bitmap.");
        return softwareBitmap;
    }
#endif

    /// <summary>
    /// Asynchronously stop media capture and dispose of resources
    /// </summary>
    /// <returns></returns>
    public async Task StopMediaFrameReaderAsync()
    {
#if ENABLE_WINMD_SUPPORT
        if (_mediaCapture != null && _mediaCapture.CameraStreamState != CameraStreamState.Shutdown)
        {
            await _mediaFrameReader.StopAsync();
            _mediaFrameReader.Dispose();
            _mediaCapture.Dispose();
            _mediaCapture = null;
        }
        IsCapturing = false;
#endif
    }


#if ENABLE_WINMD_SUPPORT
    /// <summary>
    /// https://mtaulty.com/page/5/
    /// Provide an input width, height and framerate to request for the 
    /// media capture initialization. Return a media capture and media
    /// frame source object.
    /// </summary>
    /// <param name="sourceKind"></param>
    /// <param name="width"></param>
    /// <param name="height"></param>
    /// <param name="frameRate"></param>
    /// <returns></returns>
    async Task<(MediaCapture capture, MediaFrameSource source)> GetMediaCaptureForDescriptionAsync(
            MediaFrameSourceKind sourceKind,
            int width,
            int height,
            int frameRate)
    {
        MediaCapture mediaCapture = null;
        MediaFrameSource frameSource = null;

        var allSources = await MediaFrameSourceGroup.FindAllAsync();

        // Ignore frame rate here on the description as both depth streams seem to tell me they are
        // 30fps whereas I don't think they are (from the docs) so I leave that to query later on.
        var sourceInfo =
            allSources.SelectMany(group => group.SourceInfos)
            .FirstOrDefault(
                si =>
                    (si.MediaStreamType == MediaStreamType.VideoRecord) &&
                    (si.SourceKind == sourceKind) && 
                    (si.VideoProfileMediaDescription.Any(
                        desc =>
                            desc.Width == width &&
                            desc.Height == height &&
                            desc.FrameRate == frameRate)));

        if (sourceInfo != null)
        {
            var sourceGroup = sourceInfo.SourceGroup;

            mediaCapture = new MediaCapture();

            await mediaCapture.InitializeAsync(
               new MediaCaptureInitializationSettings()
               {
                   // Want software bitmaps
                   MemoryPreference = MediaCaptureMemoryPreference.Cpu,
                   SourceGroup = sourceGroup,
                   StreamingCaptureMode = StreamingCaptureMode.Video
               }
            );
            frameSource = mediaCapture.FrameSources[sourceInfo.Id];

            var selectedFormat = frameSource.SupportedFormats.First(
                format => format.VideoFormat.Width == width && format.VideoFormat.Height == height &&
                format.FrameRate.Numerator / format.FrameRate.Denominator == frameRate);

            await frameSource.SetFormatAsync(selectedFormat);
        }
        return (mediaCapture, frameSource);
    }
#endif
}