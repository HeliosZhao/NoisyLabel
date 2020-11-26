import torch
import torch.nn as nn
import torch.nn.functional as F
from cleanselect import CleanSelect

def conv_3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)

    )

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = conv_3x3(in_planes=3, out_planes=64)
        self.conv1_2 = conv_3x3(in_planes=64, out_planes=64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = conv_3x3(in_planes=64, out_planes=128)
        self.conv2_2 = conv_3x3(in_planes=128, out_planes=128)

        self.conv3_1 = conv_3x3(in_planes=128, out_planes=196)
        self.conv3_2 = conv_3x3(in_planes=196, out_planes=196)

        self.fc1 = nn.Sequential(
            nn.Linear(3136, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Linear(256, 10)
        # self.softmax = nn.Softmax(dim=1)
        #
        # self._init_params()

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # prob = self.softmax(x)

        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='relu'
                )
