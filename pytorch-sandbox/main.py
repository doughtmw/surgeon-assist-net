import os
import argparse
import multiprocessing as mp
import torch
import numpy as np
from collections import OrderedDict
from ptflops import get_model_complexity_info

# Local imports
from data.dataloader import format_data_from_csv, create_data_generators
from model.model import create_model
from run.train import train
from run.test import test
from run.test_cpu_inference import test_cpu_inference
from run.test_pytorch_onnx_inference import test_pytorch_onnx_accuracy
from utils.utils import prepare_input, print_size_of_model, export_to_onnx


# CUDA for PyTorch
num_gpu = torch.cuda.device_count()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cholec80_256', type=str, help='training dataset to use [cholec80_256]')
parser.add_argument('--feat_ext', default='b0_lite', type=str, help='feature extractor to use [b0_lite, b0]')
parser.add_argument('--img_size', default='224,224', type=str, help='image size [(224,224)]')
parser.add_argument('--seq', default=10, type=int, help='sequence length [1, 2, 5, 10]')
parser.add_argument('--h_size', default=128, type=int, help='number of rnn cells [128]')
parser.add_argument('--trans', default=True, type=bool, help='transform training set [False, True]')
parser.add_argument('--batch_size', default=32, type=int, help='train batch size')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate for sgd')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd')
parser.add_argument('--epo', default=25, type=int, help='epochs to train')
parser.add_argument('--n_workers', default=mp.cpu_count(), type=int, help='number of cpus to use')
parser.add_argument('--ckpt', default=None, type=str, help='checkpoint to use for beginning training')
parser.add_argument('--best_weights', default=None, type=str, help='model weights used for evaluating network performance on the test set')
parser.add_argument('--other_model', default=None, type=str, help='benchmark other models - see main_benchmark.py')
args = parser.parse_args()

# Default parameters for network training
params = {'dataset': args.dataset,
          'feat_ext': args.feat_ext,
          'img_size': tuple(map(int, str(args.img_size).split(','))),
          'seq_len': args.seq,
          'hidden_size': args.h_size,
          'use_transform': args.trans,
          'num_classes': None,
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': args.n_workers,
          'learning_rate': args.lr,
          'momentum': args.momentum,
          'epochs': args.epo,
          'device': device,
          'use_cuda': use_cuda,
          'data_dir': None,
          'log_dir': 'logs/',
          'weights_dir': 'weights/',
          'target_names': None,
          'results_file': 'results_file.txt',
          'ckpt': args.ckpt,
          'other_model': args.other_model,
          'best_weights': args.best_weights}

# Parameters for data loading
# Cholec80 dataset
if params['dataset'] == 'cholec80_256':
    params['data_dir'] = '../data/data_256/'
    params['num_classes'] = 7
    params['target_names'] = [
        'CalotTriangleDissection', 'CleaningCoagulation', 'ClippingCutting',
        'GallbladderDissection', 'GallbladderPackaging', 'GallbladderRetraction',
        'Preparation']

    data_load = {
        'train_imgs': '../data/csv/img_data_train.csv',
        'val_imgs': '../data/csv/img_data_val.csv',
        'test_imgs': '../data/csv/img_data_test.csv',
        'train_phases': '../data/csv/phase_data_train.csv',
        'val_phases': '../data/csv/phase_data_val.csv',
        'test_phases': '../data/csv/phase_data_test.csv',
        'use_weights': False}

else:
    print('No dataset selected to load.')


def main():
    print('device: {} num_gpu: {}'.format(device, num_gpu))

    print(params)
    
    # Load and format the data for passing to generators
    formatted_data = format_data_from_csv(data_load)

    # Function call to create the data generators necessary for training
    training_generator, validation_generator, test_generator = create_data_generators(
        formatted_data,
        params)

    # Create the model to be used for training
    model = create_model(params)

    # Load checkpoint weights if using
    if params['ckpt'] is not None:
        state_dict = torch.load(params['ckpt'])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        # load params
        model.load_state_dict(new_state_dict)
        print("\nSuccessfully loaded pretrained weights from checkpoint.")

    # Get size, FLOPS and parameters in current model
    print_size_of_model(model)
    model_flops, model_params = get_model_complexity_info(
        model, input_res=(params['seq_len'], 3, params['img_size'][0], params['img_size'][1]),
        input_constructor=prepare_input,
        as_strings=True,
        print_per_layer_stat=False)
    print('Model flops:  ' + model_flops)
    print('Model params: ' + model_params)    

    # Begin training the model
    if (params["best_weights"] == None):
        params["best_weights"] = train(training_generator, validation_generator, model, params)

    # Test the model using the best model weights
    test(test_generator, params["best_weights"], model, params)
    
    # Test CPU inference performance
    # NOTE: efficientnets are known to be slow in their PyTorch implementation...
    # https://github.com/lukemelas/EfficientNet-PyTorch/issues/19
    # https://github.com/pytorch/pytorch/issues/18631
    test_cpu_inference(test_generator, model, params, params["best_weights"])

    # Export the model to onnx
    onnx_model_name = params['feat_ext'] + "_" + str(params['seq_len']) + ".onnx"
    export_to_onnx(model, params, onnx_model_name, params["best_weights"])

    # Test the ONNX inference performance on CPU
    test_pytorch_onnx_accuracy(test_generator, model, params, "onnx-models/" + onnx_model_name, params["best_weights"])

    print('\n Done.\n')
 
if __name__ == "__main__":
    mp.freeze_support()
    main()
