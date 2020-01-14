import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchsummary import summary
from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from .Convolution import SeparableConv2d
import math

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



class SKtune_unit_v1(nn.Module):
    def __init__(self, channels, M=2, G=32, r=16, stride=1 ,L=32):
        """ Constructor
        Args:
            in_channels: input channel dimensionality.
            out_channels: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKtune_unit_v1, self).__init__()
        d = max(int(channels//r), L)
        self.channels = channels
        self.gap1 = nn.AdaptiveAvgPool2d((1,1))
        # self.gap2 = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Sequential(nn.Conv2d(channels, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=False))
        self.fc2 = nn.Conv2d(d,channels*M,1,1,bias=False)
        self.softmax = nn.Softmax(dim=1)
      
    def forward(self,x,pre_x):
        batch_size = x.size(0)

        output = [x,pre_x]
        U = reduce(lambda x,y:x+y,output)
        s = self.gap1(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = a_b.view(batch_size,self.M,self.channels,-1)
        a_b = self.softmax(a_b)
        a_b = list(a_b.chunk(self.M,dim = 1))
        a_b = list(map(lambda x:x.view(batch_size,self.channels,1,1),a_b))
        V = list(map(lambda x,y:x*y, output,a_b))
        V = reduce(lambda x,y:x+y,V)

        return V
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1,sparable = False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                        padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1,sparable = False):
        super(Bottleneck, self).__init__()
        
        if sparable:
            self.conv1 = SeparableConv2d(inplanes, planes, kernel_size=1, bias=False)
            self.conv2 = SeparableConv2d(planes, planes, kernel_size=3, stride=stride,
            dilation=dilation, bias=False)
            self.conv3 = SeparableConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                        padding=dilation, dilation=dilation, bias=False)
            self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        
       
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        
       
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




