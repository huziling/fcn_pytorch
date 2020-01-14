import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from functools import reduce
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchsummary import summary
import os
# from .sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from .Convolution import SeparableConv2d
import math

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def show(keys):
    s = ''
    for i,k in enumerate(keys):
        s += k + ' '
        if i != 0 and i % 5 == 0:
            print(s)
            s = ''

class SKtune_unit(nn.Module):
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
        super(SKtune_unit, self).__init__()
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

    def __init__(self, inplanes, planes, stride=1, dilation=1, bn_momentum=0.1,sparable = False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False,padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                        padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1,sparable = False):
        super(Bottleneck, self).__init__()
        
        if sparable:
            pass
            # self.conv1 = SeparableConv2d(inplanes, planes, kernel_size=1, bias=False)
            # self.conv2 = SeparableConv2d(planes, planes, kernel_size=3, stride=stride,
            # dilation=dilation, bias=False)
            # self.conv3 = SeparableConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
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



class SKtune_block(nn.Module):
    def __init__(self,block,\
        inplanes, planes, stride=1, dilation=1, downsample=None, bn_momentum=0.1,sparable = False,M=2):
        super(SKtune_block, self).__init__()
        self.block = block(inplanes, planes,\
            stride=stride, dilation=dilation, bn_momentum=bn_momentum,sparable = sparable)
        self.parallel_block = block(inplanes, planes, \
            stride=stride, dilation=dilation, bn_momentum=bn_momentum,sparable = sparable)
        self.sk_layer = SKtune_unit(planes,M=M)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self,x):
        residual = x
        out = self.block(x)
        parallel_out = self.parallel_block(x)
        out = self.sk_layer(out,parallel_out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, bn_momentum=0.1, pretrained=False, output_stride=16,mode ='resnet50',sparable = False):
        self.mode = mode
        if output_stride == 16:
            dilations = [1, 1, 1, 2]
            strides = [1, 2, 2, 1]
        elif output_stride == 8:
            dilations = [1, 1, 2, 4]
            strides = [1, 2, 1, 1]
        elif output_stride == 32:
            dilations = [1, 1, 1, 1]
            strides = [1, 2, 2, 2]
        else:
            raise Warning("output_stride must be 8 or 16!")
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       bn_momentum=bn_momentum,sparable = sparable)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       bn_momentum=bn_momentum,sparable = sparable)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       bn_momentum=bn_momentum,sparable = sparable)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3],
                                       bn_momentum=bn_momentum,sparable = sparable)

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, sparable,stride=1, dilation=1, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_momentum),
            )

        layers = []
        layers.append(SKtune_block(block,self.inplanes, planes, stride, dilation, downsample, bn_momentum=bn_momentum,sparable=sparable))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(SKtune_block(block,self.inplanes, planes, dilation=dilation, bn_momentum=bn_momentum))

        return nn.Sequential(*layers)

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url(model_urls[self.mode],model_dir=os.path.join(os.getcwd(),"models/"))
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                # print("ok",k)
                model_dict[k] = v
            else:
                # print(k)
                tk = k.split('.',2)
                if len(tk) < 3:
                    continue
                k = tk[0]+'.'+tk[1]+'.block.'+tk[2]
                # print(k)
                if k in state_dict:
                    # print('ok',k)
                    model_dict[k] = v
         
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
        show(model_dict.keys())
        # show(pretrain_dict.keys())
        print("Having loaded imagenet-pretrained successfully!")
    def forward(self, x):
        # output = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # output['x1'] = x
        low_level_feat = x
        x = self.layer2(x)
        # output["x2"] = x
        x = self.layer3(x)
        # output["x3"] = x
        x = self.layer4(x)
        # output["x4"] = x
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x, low_level_feat
        # return output

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def resnet50(bn_momentum=0.1, pretrained=False, output_stride=16,sparable = False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], bn_momentum, pretrained, output_stride,mode ='resnet50',sparable = sparable)
    return model


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images: bs * w * h * channel 
    :param means:
    :return:
    '''
    num_channels = images.data.shape[1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    for i in range(num_channels):
        images.data[:,i,:,:] -= means[i]

    return images


def resnet101(bn_momentum=0.1, pretrained=False, output_stride=16,sparable = False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], bn_momentum, pretrained, output_stride,mode ='resnet101',sparable = sparable)
    return model

def resnet18(bn_momentum=0.1, pretrained=False, output_stride=16,sparable = False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], bn_momentum, pretrained, output_stride,mode ='resnet18',sparable = sparable)
    return model

if __name__ == "__main__":
    with torch.no_grad():
        model = resnet50(pretrained=True)
        model.eval()
        k = []
        for item in model.state_dict().keys():
            if "tracked" in item or 'parallel' in item or 'sk' in item:
                continue
            k.append(item)
        show(k)