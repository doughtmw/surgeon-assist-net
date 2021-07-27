import argparse
import cv2
import os
print(cv2.__version__)
import numpy as np
from decord import VideoReader
from decord import cpu, gpu

# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', default='F:/Data/cholec80/videos/', type=str, help='folder for input cholec80 video data [F:/Data/cholec80/videos/]')
parser.add_argument('--out_dir', default='data_256', type=str, help='output folder for extracted data [surgeon-assist-net/data/xx]')
parser.add_argument('--img_size', default='256,256', type=str, help='image size for rescaling video frames [(256,256)]')
args = parser.parse_args()


# Adapted from https://gist.github.com/HaydenFaulkner/3aa69130017d6405a8c0580c63bee8e6#file-video_to_frames_decord-py
def extract_frames(width, height, video_filename, video_path, frames_dir, overwrite=False, start=-1, end=-1, every=1):
    """
    Extract frames from a video using decord's VideoReader
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """
    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, _ = os.path.split(video_path)  # get the video path and filename from the path

    assert os.path.exists(video_path)  # assert the video file exists

    # load the VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = len(vr)

    frames_list = list(range(start, end, every))
    saved_count = 0

    if every > 25 and len(frames_list) < 1000:  # this is faster for every > 25 frames and can fit in memory
        frames = vr.get_batch(frames_list).asnumpy()

        for index, frame in zip(frames_list, frames):  # lets loop through the frames until the end
            save_path = os.path.join(frames_dir, "{}_{}.jpg".format(video_filename, index))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(
                    save_path,
                    cv2.resize(
                        cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR),
                        (width, height),
                        interpolation=cv2.INTER_CUBIC))  # save the extracted image
                saved_count += 1  # increment our counter by one

    else:  # this is faster for every <25 and consumes small memory
        for index in range(start, end):  # lets loop through the frames until the end
            frame = vr[index]  # read an image from the capture

            if index % every == 0:  # if this is a frame we want to write out based on the 'every' argument
                save_path = os.path.join(frames_dir, "{}_{}.jpg".format(video_filename, index))  # create the save path
                if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                    cv2.imwrite(
                        save_path,
                        cv2.resize(
                            cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR),
                            (width, height),
                            interpolation=cv2.INTER_CUBIC))  # save the extracted image
                    saved_count += 1  # increment our counter by one

    return saved_count  # and return the count of the images we saved


# Adapted from https://gist.github.com/HaydenFaulkner/3aa69130017d6405a8c0580c63bee8e6#file-video_to_frames_decord-py
def video_to_frames(width, height, video_path, frames_dir, overwrite=False, every=1):
    """
    Extracts the frames from a video
    :param width: width of input video
    :param height: height of input video
    :param video_path: path to the video
    :param frames_dir: directory to save the frames
    :param overwrite: overwrite frames if they exist?
    :param every: extract every this many frames
    :return: path to the directory where the frames were saved, or None if fails
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path
    video_filename = video_filename[:-4] # remove end tag

    # make directory to save frames, its a sub dir in the frames_dir with the video name
    os.makedirs(frames_dir, exist_ok=True)

    print("Extracting frames from {}".format(video_filename))

    extract_frames(width, height, video_filename, video_path, frames_dir, start=25, every=every)  # let's now extract the frames

    return os.path.join(frames_dir, video_filename)  # when done return the directory containing the frames

if __name__ == '__main__':
    # Data frame size for resampling
    img_size = tuple(map(int, str(args.img_size).split(',')))
    height = int(img_size[0])
    width = int(img_size[1])
    print("Image size: ", height, "x", width)

    # List of input videos and their directory
    # Directory structure
    # surgeon-assist-net/
    # ... data/
    #     ... args.out_dir/
    out_dir = args.out_dir
    print("Out directory: ", out_dir)

    # Location of downloaded cholec80 videos
    # Download data from http://camma.u-strasbg.fr/datasets
    vid_dir = args.video_dir
    print("Video directory: ", vid_dir)

    # Iterate through each video in the directory
    vid_list = []
    for file in os.listdir(vid_dir):
        if file.endswith('.mp4'):
            vid_list.append(file)
    print(vid_list)

    for i in range(0, len(vid_list)):
        # Extract every 25th frame in the dataset, beginning with the 25th frame
        # Resize to 256 x 256 using cubic interpolation
        video_to_frames(width, height, video_path=vid_dir + vid_list[i], frames_dir=out_dir, overwrite=True, every=25)
