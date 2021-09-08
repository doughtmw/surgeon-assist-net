# https://stackoverflow.com/questions/65617755/how-to-replicate-pytorch-normalization-in-opencv-or-numpy
import argparse
import cv2
import numpy as np
from PIL import Image
import onnxruntime as rt
import time

# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--onnx_model_name', default='onnx-models/b0_lite_10.onnx', type=str, help='default onnx model to use for inference testing [onnx-models/b0_lite_10.onnx]')
parser.add_argument('--img_size', default='224,224', type=str, help='image size [(224,224)]')
parser.add_argument('--seq', default=10, type=int, help='sequence length [1, 2, 5, 10]')
args = parser.parse_args()

img_size = tuple(map(int, str(args.img_size).split(',')))

# List of phases from training
phase_list = ['CalotTriangleDissection', 'CleaningCoagulation', 'ClippingCutting',
 'GallbladderDissection', 'GallbladderPackaging', 'GallbladderRetraction',
 'Preparation']

# https://stackoverflow.com/questions/65617755/how-to-replicate-pytorch-normalization-in-opencv-or-numpy
# https://stackoverflow.com/questions/45591502/simple-way-to-scale-channels-in-opencv
# Want to recreate image transforms from PyTorch in OpenCV
mean = [0.40521515692759497, 0.27927462047480039, 0.27426218748099274]
std = [0.20460533490722591, 0.17244239120062696, 0.16623196974782356]

# Create the onnx session with 1 thread on CPU
sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 1

# Specify providers when you use onnxruntime-gpu for CPU inference.
ort_session = rt.InferenceSession(args.onnx_model_name, sess_options, providers=['CPUExecutionProvider'])
print(ort_session.get_providers())

# Define a frame queue for input sequence
frame_queue = []

def main():

    # Open the sample video for CPU inference
    cap = cv2.VideoCapture("onnx-models/video41_Trim.mp4")

    # For writing video to file
    # FPS = 25
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    # out = cv2.VideoWriter('onnx-models/video41_inference.avi',cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))

    # Read the ground truth labels from file
    f = open("onnx-models/video41-phase.txt", "r")
    print(f.readline())
    gt_label = f.readline().split("\t")[1].rstrip("\n")

    while(True):
        count = 1

        ret, image = cap.read()
        begin_time = time.time()

        # Convert the colour channel orientation from BGR to RGB
        openCvImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # resize the image
        openCvImage = cv2.resize(openCvImage, (256, 256))
        openCvImage = openCvImage.astype(np.float32)

        # Crop the normalized image to (224, 224) from (256, 256) (as in the processing code)
        w = img_size[0]
        h = img_size[1]
        center = (128, 128) # (256, 256) / 2
        x = center[1] - w/2
        y = center[0] - h/2

        # Crop the normalized resized image to (224, 224)
        openCvImage = openCvImage[int(y):int(y+h), int(x):int(x+w)]

        # img = (img - mean) / stdDev (copy ToTensor behaviour)
        openCvImage /= 255.
        openCvImage -= mean
        openCvImage /= std

        cv2.imshow('opencv-transforms', openCvImage)

        # Reshape NWHC -> NCWH
        # [224, 224, 3] -> [3, 224, 224] -> [1, 3, 224, 224]
        openCvImage = openCvImage.transpose([2, 0, 1])

        # Reshape for infenrence
        openCvImage = openCvImage.reshape((1, 3, img_size[0], img_size[1])).astype(np.float32)

        # Check if we have achieved the correct sequence length
        if (len(frame_queue) == args.seq):

            # Grab the previous 10 frames from the sequence (hard coded for 10 frames now)
            # and concatenate into a proper format for inference
            openCvImageSequence = np.concatenate([
                frame_queue[9], frame_queue[8], frame_queue[7],
                frame_queue[6], frame_queue[5], frame_queue[4],
                frame_queue[3], frame_queue[2], frame_queue[1],
                frame_queue[0]], axis=0)

            # Get inputs and run onnx model
            input_name = ort_session.get_inputs()[0].name
            outputs = ort_session.run(None, {input_name: openCvImageSequence})
            onnx_time_diff = time.time() - begin_time

            # Reshape to [batch_size, n_classes]
            outputs = outputs[0][args.seq - 1::args.seq]
            curr_pred_int = np.argmax(outputs)
            print('onnx_time_diff:', round(onnx_time_diff * 1000, 2), '  onnx_pred:', phase_list[curr_pred_int], '  gt:', gt_label)

            # Check if the current prediction matches the actual prediction for the frame and show debug text
            # if correct, show green label and if incorrect show red label
            if (gt_label == phase_list[curr_pred_int]):
                cv2.putText(image, "Current prediction: [{}] in [{:.2f}] ms".format(phase_list[curr_pred_int], onnx_time_diff * 1000), (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                cv2.putText(image, "Current prediction: [{}] in [{:.2f}] ms".format(phase_list[curr_pred_int], onnx_time_diff * 1000), (150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.putText(image, "Ground truth: [{}]".format(gt_label), (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

            cv2.imshow('image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Write the frame to video
            # out.write(image)

            # Handle the 25 -> 1 FPS sampling of the labels 
            if count == 25:
                count = 1
                gt_label = f.readline().split("\t")[1].rstrip("\n")
            else:
                count += 1
            
            # Dequeue the last frame
            frame_queue.pop(0)
        else:
            # Enqeue the current frame
            frame_queue.append(openCvImage)

    cap.release()
    cv2.destroyAllWindows()

    print('\nDone...\n')
        
if __name__ == '__main__':
    main()