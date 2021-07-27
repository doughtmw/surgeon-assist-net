import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import onnxruntime as rt
import onnx

# https://discuss.pytorch.org/t/performance-drop-when-quantizing-efficientnet/90990/12
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.model = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            "tf_efficientnet_lite0",
            pretrained=True,
            exportable=True)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224) 
        y = self.model(x) 
        return y

# Convert a pretrained (on imagenet) efficientnet lite b0 model to onnx format and evaluate inputs
# Adapted from: 
# https://github.com/rwightman/gen-efficientnet-pytorch/issues/50

# random input of correct input shape
random_input = np.random.rand(1,3,224,224)
img_resized = np.ascontiguousarray(random_input).astype(np.float32)

# save this random input for re-using by TensorFlow
np.save("onnx-models/sample-models/sample-input.npy", img_resized)

# Create the model
model = Model()
model.eval()

# Prediction of input data
with torch.no_grad():
    sample = torch.from_numpy(img_resized)
    prediction = model.forward(sample)
    
print(prediction.shape)
print(prediction)

# Export model to onnx
torch.onnx.export(model, sample, 'onnx-models/sample-models/sample-efficientnet-lite-b0.onnx', opset_version=9, input_names=["input"], output_names=["output"])

# Confirm onnx predictions are correct after export
img_resized = np.load("onnx-models/sample-models/sample-input.npy")
img_resized = np.ascontiguousarray(img_resized).astype(np.float32)

# Load the onnx model and bind to new session with input
sess = rt.InferenceSession("onnx-models/sample-models/sample-efficientnet-lite-b0.onnx")
input_name = sess.get_inputs()[0].name
pred = sess.run(None, {input_name: img_resized})[0]
print(pred.shape)
print(pred)
