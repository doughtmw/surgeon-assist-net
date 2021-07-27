import onnx
import onnxruntime
import time
import numpy as np


# Utility to save the pytorch model as an onnx model a
# https://github.com/pytorch/pytorch/commit/11845cf2440c1abb0cb117bcee9532b26573e9c9
# https://github.com/onnx/tutorials/issues/137
# def pytorch_to_onnx(params, best_model_str, model, save_name, save_name_quantized):
def pytorch_to_onnx(params, best_model_str, model, save_name):

    CPU = torch.device('cpu')
    
    # To CPU
    model.to(CPU)
    model.eval()
    model.set_swish(memory_efficient=False)

    # Load the best weights from training
    state_dict = torch.load(best_model_str)

    # Iterate across key value pair
    # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict)

    # create some dummy input of the desired size
    dummy_input = torch.randn(params['seq_len'], 3, params['img_size'][0], params['img_size'][1], device=CPU)
    print('dummy_input.size:', dummy_input.size())

    # Be sure to be using PyTorch 1.4 (1.6 onnx model will not load in UWP)
    torch.onnx.export(
        model,
        dummy_input,
        save_name,
        export_params=True,
        opset_version=9, # To support current Windows build: https://docs.microsoft.com/en-us/windows/ai/windows-ml/onnx-versions
        verbose=False,
        input_names=['input_seq'],
        output_names=['output_pred'])

class Onnx_Eval():
    def __init__(self, save_name):
        super(Onnx_Eval, self).__init__()
        self.save_name = save_name
        
        # Print warning if user uses onnxruntime-gpu instead of onnxruntime package.
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            print("warning: onnxruntime-gpu is not built with OpenMP. You might try onnxruntime package to test CPU inference.")

        self.sess_options = onnxruntime.SessionOptions()
        self.sess_options.intra_op_num_threads = 1

        # Load onnx model for a new session
        # Specify providers when you use onnxruntime-gpu for CPU inference.
        self.ort_session = onnxruntime.InferenceSession(self.save_name, self.sess_options, providers=['CPUExecutionProvider'])

    # https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb
    def eval_pytorch_vs_onnx(self, torch_in, save_name):
        
        # compute ONNX Runtime output prediction
        ort_inputs = {self.ort_session.get_inputs()[0].name: torch_in.numpy()}

        begin_time = time.time()

        ort_outs = self.ort_session.run(None, ort_inputs)

        onnx_time_diff = time.time() - begin_time

        # Get the prediction and time difference
        onnx_pred = np.squeeze(np.asarray(ort_outs), axis=0)

        return {'onnx_time_diff': onnx_time_diff, 'onnx_pred': onnx_pred}