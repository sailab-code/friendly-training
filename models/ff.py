import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Net(nn.Module):
    def __init__(self, input_size=(28, 28, 1), n_classes=10, init_mode="kaiming"):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(np.prod(input_size), 10)
        self.fc2 = nn.Linear(10, n_classes)
        self.initialize_params(init_mode)

    def initialize_params(self, init_mode):
        self.__initialize_params_on_layer(self.fc1, init_mode)
        self.__initialize_params_on_layer(self.fc2, init_mode)
        
    def __initialize_params_on_layer(self, l, init_mode):
        if init_mode == 'xavier':
            torch.nn.init.xavier_uniform_(l.weight)
            if l.bias is not None:
                torch.nn.init.zeros_(l.bias)
        elif init_mode == 'kaiming':
            torch.nn.init.kaiming_uniform_(l.weight, a=math.sqrt(5))
            if l.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(l.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(l.bias, -bound, bound)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
        
class Net2(nn.Module):
    def __init__(self, batch_norm=True, input_size=(28, 28, 1), n_classes=10, init_mode="kaiming"):
        super(Net2, self).__init__()
        self.batch_norm = batch_norm
        self.fc1 = nn.Linear(np.prod(input_size), 2500)
        self.fc2 = nn.Linear(2500, 2000)
        self.fc3 = nn.Linear(2000, 1500)
        self.fc4 = nn.Linear(1500, 1000)
        self.fc5 = nn.Linear(1000, 500)
        self.fc6 = nn.Linear(500, n_classes)
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(2500)
            self.bn2 = nn.BatchNorm1d(2000)
            self.bn3 = nn.BatchNorm1d(1500)
            self.bn4 = nn.BatchNorm1d(1000)
            self.bn5 = nn.BatchNorm1d(500)
        self.initialize_params(init_mode)

    def initialize_params(self, init_mode):
        self.__initialize_params_on_layer(self.fc1, init_mode)
        self.__initialize_params_on_layer(self.fc2, init_mode)
        self.__initialize_params_on_layer(self.fc3, init_mode)
        self.__initialize_params_on_layer(self.fc4, init_mode)
        self.__initialize_params_on_layer(self.fc5, init_mode)
        self.__initialize_params_on_layer(self.fc6, init_mode)

    def __initialize_params_on_layer(self, l, init_mode):
        if init_mode == 'xavier':
            torch.nn.init.xavier_uniform_(l.weight)
            if l.bias is not None:
                torch.nn.init.zeros_(l.bias)
        elif init_mode == 'kaiming':
            torch.nn.init.kaiming_uniform_(l.weight, a=math.sqrt(5))
            if l.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(l.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(l.bias, -bound, bound)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.batch_norm: x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        if self.batch_norm: x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        if self.batch_norm: x = self.bn3(x)
        x = torch.relu(x)
        
        x = self.fc4(x)
        if self.batch_norm: x = self.bn4(x)
        x = torch.relu(x)
        
        x = self.fc5(x)
        if self.batch_norm: x = self.bn5(x)
        x = torch.relu(x)
        
        x = self.fc6(x)
        return x
