import numpy as np
import torch
import torchvision
import os
import onnx
import time
from collections import OrderedDict
import onnxruntime as rt


# Utility to print the size of the model
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

# Utility to count total (trainable) parameters in network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# Convert tensor to numpy for onnx runtime implementation
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# https://github.com/sovrasov/flops-counter.pytorch/issues/14
# Estimate FLOPS of model
def prepare_input(resolution):
    x = torch.FloatTensor(1, *resolution)
    return dict(x=x)


def export_to_onnx(model, params, onnx_model_name, best_model_str=None):
    CPU = torch.device('cpu')
    model.to(CPU)

    if best_model_str != None:
        # Load the best weights from training
        state_dict = torch.load(best_model_str)

        # Iterate across key value pair
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)

    # Set model to eval mode
    model.eval()

    # Create a random tensor of shape for testing
    random_input = np.random.rand(params['seq_len'], 3, params['img_size'][0], params['img_size'][1])
    img_resized = np.ascontiguousarray(random_input).astype(np.float32)
    
    # save this random input for re-using by TensorFlow
    np.save("onnx-models/input.npy", img_resized)

    # Get prediction on tensor
    with torch.no_grad():
        sample = torch.from_numpy(img_resized)
        prediction = model.forward(sample)

    # Export the onnx model
    # https://github.com/microsoft/onnxruntime/blob/master/docs/Versioning.md#version-matrix
    torch.onnx.export(model, sample, 'onnx-models/' + onnx_model_name, opset_version=9, input_names=["input"], output_names=["output"])

    # Confirm onnx predictions are correct after export
    img_resized = np.load("onnx-models/input.npy")
    img_resized = np.ascontiguousarray(img_resized).astype(np.float32)

    # Load the onnx model and bind to new session with input
    sess = rt.InferenceSession("onnx-models/" + onnx_model_name)
    input_name = sess.get_inputs()[0].name
    pred = sess.run(None, {input_name: img_resized})[0]

    print("PyTorch inference:")
    print(prediction.shape)
    print(prediction)
    
    print("ONNX inference:")
    print(pred.shape)
    print(pred)
