import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleColorCNN(nn.Module):
    def __init__(self, params):    # num_classes
        super(SimpleColorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, params.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    from easydict import EasyDict as edict
    params = {'depth':28, 'widen_factor':2, 'num_classes':10, 'dropout': 0.0}
    params = edict(params)
    m = WideResNet(params)
    print(m)