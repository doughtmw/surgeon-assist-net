# https://stackoverflow.com/questions/65617755/how-to-replicate-pytorch-normalization-in-opencv-or-numpy
import argparse
import cv2
import numpy as np
from PIL import Image
import onnxruntime as rt

from data.dataset import pil_loader
from data.transforms import augment_image, to_tensor_normalize

# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--onnx_model_name', default='onnx-models/b0_lite_1.onnx', type=str, help='default onnx model to use for inference testing [onnx-models/b0_lite_1.onnx]')
parser.add_argument('--images', default=['video41_25'], type=str, help='sample image for testing inference')
parser.add_argument('--img_size', default='224,224', type=str, help='image size [(224,224)]')
parser.add_argument('--seq', default=1, type=int, help='sequence length [1, 2, 5, 10]')
args = parser.parse_args()

img_size = tuple(map(int, str(args.img_size).split(',')))

# Compare the transforms applied to training data in PyTorch to the ones during inference with OpenCV and make
# sure that they are similar.
# NOTE: for some reason, when testing with the WinMlDashboard tool, cropped images give incorrect results. Need
# to use full uncropped images for testing purposes.

# https://stackoverflow.com/questions/65617755/how-to-replicate-pytorch-normalization-in-opencv-or-numpy
# https://stackoverflow.com/questions/45591502/simple-way-to-scale-channels-in-opencv
# Want to recreate image transforms from PyTorch in OpenCV
mean = [0.40521515692759497, 0.27927462047480039, 0.27426218748099274]
std = [0.20460533490722591, 0.17244239120062696, 0.16623196974782356]


def main():

    for image in args.images:
        print(image)

        # PyTorch normalization pipeline
        normalize = to_tensor_normalize()
        img_augment = augment_image(False, img_size[0])
        img_augment = img_augment.to_deterministic()

        # Load the image using pil
        ptImage = np.array(pil_loader('onnx-models/' + image + '.jpg'))
        ptImage = Image.fromarray(img_augment(image=ptImage))
        ptImage = normalize(ptImage).numpy()

        cv2.imshow('ptImage-transforms', ptImage.swapaxes(0,2).swapaxes(0,1))
        cv2.imwrite('onnx-models/' + image + '-pt-transforms.jpg', ptImage.swapaxes(0,2).swapaxes(0,1))

        # OpenCV pipeline
        # open the image as an OpenCV image
        openCvImage = cv2.imread('onnx-models/' + image + '.jpg')

        # Convert the colour channel orientation from BGR to RGB
        openCvImage = cv2.cvtColor(openCvImage, cv2.COLOR_BGR2RGB)

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
        cv2.imwrite('onnx-models/' + image + '-opencv-transforms.jpg', openCvImage)

        # Reshape NWHC -> NCWH
        # [224, 224, 3] -> [3, 224, 224] -> [1, 3, 224, 224]
        openCvImage = openCvImage.transpose([2, 0, 1])

        # show results
        print('\nptImage.shape = ' + str(ptImage.shape))
        print('ptImage max = ' + str(np.max(ptImage)))
        print('ptImage min = ' + str(np.min(ptImage)))
        print('ptImage avg = ' + str(np.mean(ptImage)))
        print('ptImage: ')

        # Reshape for infenrence
        openCvImage = openCvImage.reshape((args.seq, 3, img_size[0], img_size[1])).astype(np.float32)
        ptImage = ptImage.reshape((args.seq, 3, img_size[0], img_size[1])).astype(np.float32)

        print('\nopenCvImage.shape = ' + str(openCvImage.shape))
        print('openCvImage max = ' + str(np.max(openCvImage)))
        print('openCvImage min = ' + str(np.min(openCvImage)))
        print('openCvImage avg = ' + str(np.mean(openCvImage)))
        print('openCvImage: ')

        # Create the model
        # Test ONNX inference
        sess = rt.InferenceSession(args.onnx_model_name)
        input_name = sess.get_inputs()[0].name
        outputs = sess.run(None, {input_name: openCvImage})
        outputs_pt = sess.run(None, {input_name: ptImage})

        # First image of test dataset video41_25, want:
	    # local_output: tensor([[ 0.1532,  5.9982, -0.3814, -4.1019,  0.9841,  1.8995, -4.3837]], device = 'cuda:0')

        # Get:
        # outputs:  [array([[ 0.1524404, 5.998244 , -0.3819608, -4.101637,  0.9846859, 1.8998177, -4.383874 ]], dtype=float32)]
        print('\noutputs opencv normalized: ', outputs)

        # outputs pt:  [array([[ 0.15243927, 5.998246, -0.3819655, -4.1016374, 0.9846873, 1.8998195, -4.3838735 ]], dtype=float32)]
        print('\noutputs pytorch normalized: ', outputs_pt)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('\nDone...\n')
        
if __name__ == '__main__':
    main()