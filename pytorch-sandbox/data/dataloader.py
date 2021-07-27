import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import imgaug

# Local imports
from data.dataset import Dataset


# Correctly handle RNG to ensure different training distribution across epochs
# https://github.com/aleju/imgaug/issues/406
# https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562
def worker_init_fn(worker_id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    imgaug.seed(ss.generate_state(4))

# Format input data dict from csv file
def format_data_from_csv(data_dict):
    # Format the data into a dict and return
    formatted_data_dict = {}

    # Load the data from formatted csv files
    # Image: files
    train_imgs = pd.read_csv(data_dict['train_imgs'], usecols=['Files'], dtype=str)
    val_imgs = pd.read_csv(data_dict['val_imgs'], usecols=['Files'], dtype=str)
    test_imgs = pd.read_csv(data_dict['test_imgs'], usecols=['Files'], dtype=str)
    # print('train_imgs:',train_imgs)

    # Phase: files and phase
    train_phases = pd.read_csv(data_dict['train_phases'],usecols=['Files', 'Phase'],dtype=str)
    val_phases = pd.read_csv(data_dict['val_phases'],usecols=['Files', 'Phase'],dtype=str)
    test_phases = pd.read_csv(data_dict['test_phases'],usecols=['Files', 'Phase'],dtype=str)
    print('train_phases:',train_phases)

    # Encode the phase labels from strings to ints
    le = LabelEncoder()
    le.fit(train_phases['Phase'])
    print('le.classes_:', le.classes_)

    # Modify loaded data to reflect the label encoding
    train_phases['Phase'] = le.transform(train_phases['Phase'])
    val_phases['Phase'] = le.transform(val_phases['Phase'])
    test_phases['Phase'] = le.transform(test_phases['Phase'])
    # print('train_phases:',train_phases)

    # Datasets
    # Load images into train, val and test dictionary partitions
    # {'train': ['video01_25', 'video01_50', 'video01_75', 'video01_100']}
    # partition['train']: ['video01_25', 'video01_50', 'video01_75', 'video01_100']
    partition_train,partition_val,partition_test = {},{},{}

    partition_train['train'] = train_imgs['Files'].values.tolist()
    partition_val['validation'] = val_imgs['Files'].values.tolist()
    partition_test['test'] = test_imgs['Files'].values.tolist()

    # Compute class reweighting scheme
    if data_dict['use_weights']:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_phases['Phase'].values),
            y=train_phases['Phase'].values)
        class_weights_tensor = torch.from_numpy(class_weights)

        formatted_data_dict['class_weight'] = class_weights_tensor
        print('class_weights_tensor:',class_weights_tensor)
    else:
        n_classes = len(np.unique(train_phases['Phase'].values))
        class_weights_tensor = torch.from_numpy(np.ones(n_classes))
        formatted_data_dict['class_weight'] = class_weights_tensor
        print('class_weights_tensor:',class_weights_tensor)

    # Load the phase labels to format as:
    # {'video01_25': 'Preparation', 'video01_50': 'Preparation', 'video01_75': 'Preparation'}
    labels_train,labels_val,labels_test = {},{},{}

    labels_train = train_phases.set_index('Files','Phase').to_dict()['Phase']
    labels_val = val_phases.set_index('Files','Phase').to_dict()['Phase']
    labels_test = test_phases.set_index('Files','Phase').to_dict()['Phase']

    formatted_data_dict['partition_train'] = partition_train['train']
    formatted_data_dict['partition_val'] = partition_val['validation']
    formatted_data_dict['partition_test'] = partition_test['test']

    formatted_data_dict['labels_train'] = labels_train
    formatted_data_dict['labels_val'] = labels_val
    formatted_data_dict['labels_test'] = labels_test

    return formatted_data_dict

# def create_data_generators(data_dict, params, train_transforms, val_transforms, test_transforms):
def create_data_generators(data_dict, params):

    # Cache the data dict to params if not none
    if data_dict['class_weight'] is not None:
        params['class_weight'] = data_dict['class_weight']
    else:
        params['class_weight'] = None

    # Generators
    # Training generator
    training_set = Dataset(data_dict['partition_train'], data_dict['labels_train'], params)
    training_generator = torch.utils.data.DataLoader(
        training_set,
        batch_size=params['batch_size'],
        sampler=None,
        shuffle=False,
        num_workers=params['num_workers'],
        pin_memory=True,
        worker_init_fn=worker_init_fn)

    # Set shuffle and transforms to False for validation and test data
    params['shuffle'] = False
    params['use_transform'] = False

    # Validation generator
    validation_set = Dataset(data_dict['partition_val'], data_dict['labels_val'], params)
    validation_generator = torch.utils.data.DataLoader(
        validation_set,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['num_workers'],
        pin_memory=True)

    # Test generator
    test_set = Dataset(data_dict['partition_test'], data_dict['labels_test'], params)
    test_generator = torch.utils.data.DataLoader(
        test_set,
        batch_size=params['batch_size'],
        shuffle=False,
        num_workers=params['num_workers'],
        pin_memory=True)

    # Return the generators
    return training_generator, validation_generator, test_generator
