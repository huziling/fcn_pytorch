
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class SELayer(nn.Module):
    def __init__(self, inchannel,outchannel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, outchannel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        for i in range(x.size(1)):
            x[0, i, :, :] = x[0, i, :, :] * y[0, i, 0, 0].cpu().data.float()[0]

        return x

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()

        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        padding = pad_total // 2

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        # extra BatchNomalization and ReLU
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class SeparableAsppConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bn_momentum=0.1):
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, stride=stride\
            , padding=padding, dilation=dilation,groups=in_channels,bias=False)
        self.relu_dp = nn.ReLU(inplace=True)
        # self.bn_dp = SynchronizedBatchNorm2d(in_channels, momentum=bn_momentum)
        self.bn_dp = nn.SyncBatchNorm(in_channels,momentum=bn_momentum)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.bn = SynchronizedBatchNorm2d(in_channels, momentum=bn_momentum)
        self.bn = nn.SyncBatchNorm(out_channels,momentum=bn_momentum)
    
    def forward(self,x):
        x = self.depthwise(x)
        x = self.bn_dp(self.relu_dp(x))
        x = self.pointwise(x)
        x = self.bn(self.relu(x))
        return x
def AsppConv(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bn_momentum=0.1):
    asppconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
            # SynchronizedBatchNorm2d(out_channels, momentum=bn_momentum),
            nn.SyncBatchNorm(out_channels,momentum=bn_momentum),
            nn.ReLU()
        )
    return asppconv