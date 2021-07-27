import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Local imports
from data.transforms import augment_image, to_tensor_normalize


# Load images using PIL
def pil_loader(path):
    # Open images
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

# Adapted from https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
class Dataset(torch.utils.data.Dataset):

    # Create dataset, cache labels and list ids
    def __init__(self, list_IDs, labels, params):

        self.labels = labels
        self.list_IDs = list_IDs

        # (seq_len, x, y, 3)
        self.use_transform = params['use_transform']
        self.seq_len = params['seq_len']
        self.img_size = params['img_size']
        self.shuffle = params['shuffle']
        self.num_classes = params['num_classes']
        self.data_dir = params['data_dir']

        self.on_epoch_end()

        self.to_tensor_normalize = to_tensor_normalize()

    # Total number of samples
    def __len__(self):
        return int(len(self.list_IDs))

    # Generate one sample of data
    def __getitem__(self, index):

        # Initialization
        # X : (seq_len, 3, x, y)
        X = torch.empty((self.seq_len, 3, self.img_size[0], self.img_size[1]))
        y = 0

        # Generate indexes of the batch
        indexes = self.indexes[index]

        # Find list of IDs
        list_IDs_temp = self.list_IDs[indexes]
        # print('list_IDs_temp:',list_IDs_temp)

        # Create the transform with deterministic behaviour for batch
        img_augment = augment_image(self.use_transform, self.img_size)
        img_augment = img_augment.to_deterministic()

        # Generate data
        for j in range(0, self.seq_len):
            try:
                # Take jpg images from data dir as input, one phase per
                # time_series of images per batch
                # print('outer_ID:',ID)
                ID_idx = self.list_IDs.index(list_IDs_temp)
                ID_ts = self.list_IDs[ID_idx + j]
                img_name = self.data_dir + ID_ts + '.jpg'

                # Load the current image from image sequence
                img = np.array(pil_loader(img_name))
                # print('image:', j, 'name:', img_name, 'phase:', self.labels[ID_ts])

                # Store the X and y data increments in array
                # Augment the training data
                img_transformed = Image.fromarray(img_augment(image=img))
                X[j, ] = self.to_tensor_normalize(img_transformed)

                # Debug with a figure of image and corresponding phase
                # fig = plt.figure()

                # plt.subplot(1, 3, 1)
                # plt.imshow(img)

                # plt.subplot(1, 3, 2)
                # plt.imshow(img_transformed)

                # plt.subplot(1, 3, 3)
                # plt.imshow(X[j, ].permute(1, 2, 0))

                # plt.show()
                # plt.pause(5)

                if hasattr(img, 'close'):
                    img.close()

            except IndexError:
                X[j, ] = torch.from_numpy(np.zeros(self.img_size))

        # Get the phase label
        y = self.labels[list_IDs_temp]

        # print('y:', y)
        # print('X.shape', X.shape)
        # print('y.shape', y.shape)

        # Transform data
        return X, y


    # Called at end of the epoch
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            # https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
            uint64_seed = torch.initial_seed()
            ss = np.random.SeedSequence([uint64_seed])
            # More than 128 bits (4 32-bit words) would be overkill.
            np.random.seed(ss.generate_state(4))
            
            # htptps://discuss.pytorch.org/t/shuffling-a-tensor/25422/3
            np.random.shuffle(self.indexes)
