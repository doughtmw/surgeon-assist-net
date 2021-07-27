import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
import torch.nn.functional as F

# Local imports
from utils.utils import count_parameters


def create_model(params, feat_ext):

    if feat_ext == 'SV_RCNet':
        print('Using SV_RCNet.')
        return SV_RCNet(params)

    if feat_ext == 'PhaseNet':
        print('Using PhaseNet.')
        return PhaseNet(params)
        
    else:
        print('No feature extraction backbone selected.')
        return None

# SV-RCNet
# https://github.com/YuemingJin/MTRCNet-CL/blob/master/pytorch1.2.0/train_singlenet_phase.py
# Seq len 10
# 512 LSTM cells
class SV_RCNet(torch.nn.Module):
    def __init__(self, params):
        super(SV_RCNet, self).__init__()
        self.img_size = params['img_size']
        self.seq_len = params['seq_len']
        self.num_classes = params['num_classes']

        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        # self.fcDropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, self.num_classes)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

        print('count_parameters(self.features):', count_parameters(self.share)) #  0
        print('count_parameters(self.rnn):', count_parameters(self.lstm)) # 126,144
        print('count_parameters(self.fc):', count_parameters(self.fc)) # 231

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(-1, self.seq_len, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        # y = self.fcDropout(y)
        y = self.fc(y)
        return y

# PhaseNet
# https://arxiv.org/pdf/1602.03012.pdf
# seq len 1 
class PhaseNet(torch.nn.Module):
    def __init__(self, params):
        super(PhaseNet, self).__init__()
        self.img_size = params['img_size']
        self.seq_len = params['seq_len']
        self.num_classes = params['num_classes']

        alexnet = models.alexnet(pretrained=True)
        self.share = nn.Sequential(
            # stop at conv5
            *list(alexnet.features.children())[:-2]
        )
        self.fc_ = nn.Linear(43264, 4096)
        self.relu = nn.ReLU()
        # self.fcDropout = nn.Dropout(0.5)
        self.fc = nn.Linear(4096, self.num_classes)

        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.share.forward(x)
        # print('features:', x.size())
        x = x.view(-1, 43264)

        x = self.fc_(x)
        # print('fc_:', x.size())

        x = x.view(-1, self.seq_len, 4096)

        # y = self.fcDropout(y)
        x = self.relu(x)
        y = self.fc(x)
        y = y.view(-1, self.num_classes)

        return y
