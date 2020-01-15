
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .se_module import SELayer_2

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, padding=0,dilation=0, bn=False,Se=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding,dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.se = SELayer_2(in_channels, out_channels, 16) if Se else None


        # self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
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
        # x = self.relu(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class SeparableAsppConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bn_momentum=0.1):
        super(SeparableAsppConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, stride=stride\
            , padding=padding, dilation=dilation,groups=in_channels,bias=False)
        self.relu_dp = nn.ReLU(inplace=True)
        # self.bn_dp = SynchronizedBatchNorm2d(in_channels, momentum=bn_momentum)
        self.bn_dp = nn.BatchNorm2d(in_channels,momentum=bn_momentum)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.bn = SynchronizedBatchNorm2d(in_channels, momentum=bn_momentum)
        self.bn = nn.BatchNorm2d(out_channels,momentum=bn_momentum)
    
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
            nn.BatchNorm2d(out_channels,momentum=bn_momentum),
            nn.ReLU()
        )
    return asppconv