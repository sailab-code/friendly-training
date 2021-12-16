import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

def conv_out_formula(inp, pad, dil, ker, stri):
    return floor((inp + 2 * pad - dil * (ker - 1) - 1) / stri + 1)

def calc_out_conv_layers(in_h, in_w, kernels, paddings, dilations, strides):
    out_h = in_h
    out_w = in_w
    h,w = [in_h], [in_w]
    for ker, pad, dil, stri in zip(kernels, paddings, dilations, strides):
        out_h = conv_out_formula(out_h, pad, dil, ker, stri)
        out_w = conv_out_formula(out_w, pad, dil, ker, stri)
        h.append(out_h)
        w.append(out_w)
    return h, w

class CNN(nn.Module):
    def __init__(self, input_size = (28,28,1), n_classes=10):
        super(CNN, self).__init__()
        n_channels = input_size[-1]
        input_dim = input_size[0]

        self.batch_norm = False
        self.conv1 = nn.Conv2d(n_channels, 32, 3, 1) #ker 3, stri 1, pad 0, dil 1
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        out, _ = calc_out_conv_layers(input_dim, input_dim, [3, 3], [0,0], [1,1],[1,1])
        
        self.fc1 = nn.Linear(conv_out_formula(out[-1], stri=2, dil=1, ker=2,pad=0)**2 * 64, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


class CNN2(nn.Module):
    def __init__(self,  input_size = (28,28,1), n_classes=10):
        super(CNN2, self).__init__()
        n_channels = input_size[-1]
        input_dim = input_size[0]

        self.batch_norm = False
        self.conv1 = nn.Conv2d(n_channels, 32, 4, 1)
        self.conv2 = nn.Conv2d(32, 48, 4, 1)
        self.conv3 = nn.Conv2d(48, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 2, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        out, _ = calc_out_conv_layers(input_dim, input_dim, kernels=[4, 4,3,2], paddings=[0,0,0,0], dilations=[1,1,1,1],strides=[1,1,1,1])
        self.fc1 = nn.Linear(conv_out_formula(out[-1], stri=2, dil=1, ker=2,pad=0)**2 * 64, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)

        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output
