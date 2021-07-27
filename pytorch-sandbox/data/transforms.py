import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import random
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
import imgaug.parameters as iap


# Parameters for standardizing dataset
# From: https://github.com/ChinaYi/miccai19/blob/master/cfg.py
norm_mean = [0.40521515692759497, 0.27927462047480039, 0.27426218748099274]
norm_std = [0.20460533490722591, 0.17244239120062696, 0.16623196974782356]
ia.seed(1)

# Center crop the incoming image
def CenterCrop(height, width):
    crop = iaa.CropToFixedSize(height=height, width=width)
    crop.position = (iap.Deterministic(0.5), iap.Deterministic(0.5))
    pad = iaa.PadToFixedSize(height=height, width=width, pad_mode=ia.ALL, pad_cval=(0, 255))
    pad.position = (iap.Deterministic(0.5), iap.Deterministic(0.5))
    return iaa.Sequential([crop, pad])

# Create the transformations to apply to incoming images
def augment_image(use_transform, img_size):
    if use_transform:
        return iaa.Sequential([
                iaa.CropToFixedSize(width=224, height=224),
                iaa.flip.Fliplr(p=0.5),
                iaa.flip.Flipud(p=0.2),
                iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
                iaa.MultiplyBrightness(mul=(0.65, 1.35)),
                iaa.LinearContrast((0.75, 1.5)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, rotate=(-25, 25), shear=(-8, 8))
            ], random_order=True)
    else:
        return iaa.Sequential([
                CenterCrop(width=224, height=224)
            ], random_order=False)
  
def to_tensor_normalize():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)])

