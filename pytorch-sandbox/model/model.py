import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

# Local imports
from utils.utils import count_parameters


def create_model(params):
    feat_ext = str(params['feat_ext'])

    if feat_ext == 'b0_lite':
        print('Using effnet-lite b0 as feature extractor.')
        return effnetb0_lite_rnn(params)

    if feat_ext == 'b0':
        print('Using effnet b0 as feature extractor.')
        return effnetb0_rnn(params)
    
    else:
        print('No feature extraction backbone selected.')
        return None

# EfficientNet-Lite-B0
class effnetb0_lite_rnn(torch.nn.Module):
    def __init__(self, params):
        super(effnetb0_lite_rnn, self).__init__()
        self.img_size = params['img_size']
        self.seq_len = params['seq_len']
        self.hidden_size = params['hidden_size']
        self.num_classes = params['num_classes']

        # Feature extraction (no 1x1 conv layer, global pooling, dropout and fc head)
        effnetb0 = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            "efficientnet_lite0",
            pretrained=True,
            exportable=True)
        self.feat = torch.nn.Sequential(*list(effnetb0.children())[:-4])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.rnn = nn.GRU(input_size=1280, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        # Prediction structure
        self.pred = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.num_classes))

        # Initialize rnn weights
        init.xavier_normal_(self.rnn.all_weights[0][0])
        init.xavier_normal_(self.rnn.all_weights[0][1])

        print('count_parameters(self.feat):', count_parameters(self.feat))
        print('count_parameters(self.rnn):', count_parameters(self.rnn))
        print('count_parameters(self.pred):', count_parameters(self.pred))

    def forward(self, x):
        x = x.view(-1, 3, self.img_size[0], self.img_size[1]) 
        x = self.feat.forward(x) 
        x = self.avgpool(x) 
        x = x.view(-1, self.seq_len, 1280) 
        self.rnn.flatten_parameters()
        y, _ = self.rnn(x) 
        y = y.contiguous().view(-1, self.hidden_size) 
        y = self.pred(y) 

        return y

# EfficientNet-B0
class effnetb0_rnn(torch.nn.Module):
    def __init__(self, params):
        super(effnetb0_rnn, self).__init__()
        self.img_size = params['img_size']
        self.seq_len = params['seq_len']
        self.hidden_size = params['hidden_size']
        self.num_classes = params['num_classes']

        # Feature extraction (no 1x1 conv layer, global pooling, dropout and fc head)
        effnetb0 = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch",
            "efficientnet_b0",
            pretrained=True,
            exportable=True)
        self.feat = torch.nn.Sequential(*list(effnetb0.children())[:-4])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.rnn = nn.GRU(input_size=1280, hidden_size=self.hidden_size, num_layers=1, batch_first=True)

        # Prediction structure
        self.pred = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.num_classes))

        # Initialize rnn weights
        init.xavier_normal_(self.rnn.all_weights[0][0])
        init.xavier_normal_(self.rnn.all_weights[0][1])

        print('count_parameters(self.feat):', count_parameters(self.feat))
        print('count_parameters(self.rnn):', count_parameters(self.rnn))
        print('count_parameters(self.pred):', count_parameters(self.pred))

    def forward(self, x):
        x = x.view(-1, 3, self.img_size[0], self.img_size[1]) 
        x = self.feat.forward(x) 
        x = self.avgpool(x) 
        x = x.view(-1, self.seq_len, 1280)
        self.rnn.flatten_parameters()
        y, _ = self.rnn(x) 
        y = y.contiguous().view(-1, self.hidden_size) 
        y = self.pred(y) 
        return y

