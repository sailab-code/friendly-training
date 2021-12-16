import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor
activations = {'relu': nn.ReLU(), 'leaky-relu': nn.LeakyReLU(), 'tanh': nn.Tanh(), 'sigmoid':nn.Sigmoid()}

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


def calc_out_deconv_padding(out, inp, s, p, k):
    return out -1 - (inp-1)*s+2*p - 1*(k-1)


class SimplifierFF(torch.nn.Module):
    """This network learns to generate simplified input data."""

    def __init__(self, input_dimensionality, hidden, activation, sigmoid_postprocessing, target_conditioning, n_classes=10):
        super(SimplifierFF, self).__init__()
        self.sigmoid_postprocessing = sigmoid_postprocessing
        self.input_dimensionality = input_dimensionality
        self.target_conditioning = target_conditioning
        self.n_classes = n_classes
        hidden = [input_dimensionality] + hidden
        if target_conditioning: hidden[0] += n_classes
        N = len(hidden)
        activation = activations[activation]

        layers = []
        for i in range(N):
            layers.append(nn.Linear(hidden[i], hidden[i+1]))
            if i == N-2: break
            layers.append(activation)
        if self.sigmoid_postprocessing: layers.append(activations['sigmoid'])

        self.generator = nn.Sequential(*layers)

    def forward(self, X, y):
        orig_shape = X.shape
        if self.target_conditioning:
            target_encoded = y.new_zeros(X.size(0), self.n_classes)
            target_encoded[range(X.size(0)), y] = 1.0
            X = torch.cat((X, target_encoded), 1)
        X = X.view(X.size(0), -1)
        O = self.generator(X)
        O = O.view(orig_shape)
        return O

from .unet_parts import *

class SimplifierUNet(nn.Module):
    def __init__(self, sigmoid_postprocessing, target_conditioning, n_classes=10, n_filters_base = 64, n_deep = 2, bilinear=True, input_size = (28,28,1)):
        super(SimplifierUNet, self).__init__()
        n_channels = input_size[-1]
        image_width = input_size[0]
        self.sigmoid_postprocessing = sigmoid_postprocessing
        self.target_conditioning = target_conditioning
        self.input_dim = image_width
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_deep = n_deep
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels+(n_classes if self.target_conditioning else 0), n_filters_base)
        '''
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 1, bilinear)
        '''
        if self.n_deep == 2:
            self.down1 = Down(n_filters_base, 2*n_filters_base // factor)
            self.up1 = Up(2*n_filters_base, n_channels, bilinear)
        elif self.n_deep == 4:
            self.down1 = Down(n_filters_base, 2 * n_filters_base)
            self.down2 = Down(2 * n_filters_base, 2*2 * n_filters_base // factor)
            self.up1 = Up(2*2*n_filters_base, 2 * n_filters_base // factor, bilinear)
            self.up2 = Up(2 * n_filters_base, n_channels, bilinear)



    def forward(self, x, y):
        '''
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        '''
        if self.target_conditioning:
            target_encoded = y.new_zeros(x.size(0), self.n_classes, self.input_dim, self.input_dim)
            target_encoded[range(x.size(0)), y, :, :] = 1.0
            x = torch.cat((x, target_encoded), 1)
        x1 = self.inc(x)
        if self.n_deep == 2:
            x2 = self.down1(x1)
            x = self.up1(x2, x1)
        else:
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up1(x3, x2)
            x = self.up2(x, x1)

        if self.sigmoid_postprocessing: x = torch.sigmoid(x)

        return x
