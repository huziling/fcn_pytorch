from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from se_module import SELayer, SEGet


def get_upsample_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]  # list (64 x 1), (1 x 64)
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)  # 64 x 64
    weight = np.zeros((in_channels, out_channels, kernel_size,
                       kernel_size), dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt

    return torch.from_numpy(weight).float()


class FCN32s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
        score = self.bn1(self.relu(self.deconv1(x5)))  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn2(self.relu(self.deconv2(score)))  # size=[n, 256, x.h/8, x.w/8]
        score = self.bn3(self.relu(self.deconv3(score)))  # size=[n, 128, x.h/4, x.w/4]
        score = self.bn4(self.relu(self.deconv4(score)))  # size=[n, 64, x.h/2, x.w/2]
        score = self.bn5(self.relu(self.deconv5(score)))  # size=[n, 32, x.h, x.w]
        score = self.classifier(score)  # size=[n, n_class, x.h, x.w]

        return score


class FCN16s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
        x4 = output['x4']  # size=[n, 512, x.h/16, x.w/16]

        score = self.relu(self.deconv1(x5))  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn1(score + x4)  # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.bn2(self.relu(self.deconv2(score)))  # size=[n, 256, x.h/8, x.w/8]
        score = self.bn3(self.relu(self.deconv3(score)))  # size=[n, 128, x.h/4, x.w/4]
        score = self.bn4(self.relu(self.deconv4(score)))  # size=[n, 64, x.h/2, x.w/2]
        score = self.bn5(self.relu(self.deconv5(score)))  # size=[n, 32, x.h, x.w]
        score = self.classifier(score)  # size=[n, n_class, x.h, x.w]

        return score


class FCN8s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

        init.xavier_uniform_(self.deconv1.weight)
        init.xavier_uniform_(self.deconv2.weight)
        init.xavier_uniform_(self.deconv3.weight)
        init.xavier_uniform_(self.deconv4.weight)
        init.xavier_uniform_(self.deconv5.weight)
        init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
        x4 = output['x4']  # size=[n, 512, x.h/16, x.w/16]
        x3 = output['x3']  # size=[n, 512, x.h/8, x.w/8]

        score = self.relu(self.deconv1(x5))  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn1(score + x4)  # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.relu(self.deconv2(score))  # size=[n, 256, x.h/8, x.w/8]
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=[n, 128, x.h/4, x.w/4]
        score = self.bn4(self.relu(self.deconv4(score)))  # size=[n, 64, x.h/2, x.w/2]
        score = self.bn5(self.relu(self.deconv5(score)))  # size=[n, 32, x.h, x.w]
        score = self.classifier(score)  # size=[n, n_class, x.h, x.w]

        return score

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                # if m.bias is not None:
                m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsample_weight(m.in_channels,
                                                     m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)  # copy not = ?


class FCN1s(nn.Module):
    def __init__(self, pretrained_net, n_class, Time=False, Space=False):
        super().__init__()
        self.n_class = n_class
        self.Time = Time
        self.Space = Space
        self.pretrained_net = pretrained_net
        self.S1 = SEGet(64, 2)
        self.S2 = SEGet(128, 16)
        self.S3 = SEGet(256, 16)
        self.S4 = SEGet(512, 16)
        self.S5 = SEGet(512, 16)

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.se1 = SELayer(512, 16)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.se2 = SELayer(256, 16)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.se3 = SELayer(128, 16)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.se4 = SELayer(64, 16)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.se5 = SELayer(32, 16)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
        x4 = output['x4']  # size=[n, 512, x.h/16, x.w/16]
        x3 = output['x3']  # size=[n, 512, x.h/8, x.w/8]
        x2 = output['x2']  # size=[n, 512, x.h/4, x.w/4]
        x1 = output['x1']  # size=[n, 512, x.h/2, x.w/2]

        if self.Space:
            S1 = self.S1(x1)
            S2 = self.S2(x2)
            S3 = self.S3(x3)
            S4 = self.S4(x4)
            S5 = self.S5(x4)

            # if self.Time:

        score = self.deconv1(x5)
        # if self.Space:
        #     score = score*S5
        # if self.Time:
        #     score = self.se1(score)
        score = self.relu(score)

        score = self.bn1(score + x4)
        if self.Space:
            score = score * S4
        score = self.deconv2(score)
        if self.Time:
            score = self.se2(score)
        score = self.relu(score)

        score = self.bn2(score + x3)
        if self.Space:
            score = score * S3
        score = self.deconv3(score)
        if self.Time:
            score = self.se3(score)
        score = self.relu(score)

        score = self.bn3(score + x2)
        if self.Space:
            score = score * S2
        score = self.deconv4(score)
        if self.Time:
            score = self.se4(score)
        score = self.relu(score)

        score = self.bn4(score + x1)
        if self.Space:
            score = score * S1
        score = self.deconv5(score)
        if self.Time:
            score = self.se5(score)
        score = self.bn5(score)
        score = self.classifier(score)

        return score


class FCNss(nn.Module):
    def __init__(self, pretrained_net, n_class, Time=False, Space=False):
        super().__init__()
        self.n_class = n_class
        self.Time = Time
        self.Space = Space
        self.pretrained_net = pretrained_net
        self.S1 = SEGet(64, 16)
        self.S2 = SEGet(128, 16)
        self.S3 = SEGet(256, 16)
        self.S4 = SEGet(512, 16)
        self.S5 = SEGet(512, 16)

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
        x4 = output['x4']  # size=[n, 512, x.h/16, x.w/16]
        x3 = output['x3']  # size=[n, 512, x.h/8, x.w/8]
        x2 = output['x2']  # size=[n, 512, x.h/4, x.w/4]
        x1 = output['x1']  # size=[n, 512, x.h/2, x.w/2]

        S1 = torch.sum(torch.sum(self.S1(x1)))
        S2 = torch.sum(torch.sum(self.S2(x2)))
        S3 = torch.sum(torch.sum(self.S3(x3)))
        S4 = torch.sum(torch.sum(self.S4(x4)))

        total_sum = S1 + S2 + S3 + S4

        s1 = torch.div(S1, total_sum)
        s2 = torch.div(S2, total_sum)
        s3 = torch.div(S3, total_sum)
        s4 = torch.div(S4, total_sum)

        score = self.relu(self.deconv1(x5))  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn1(score + x4 * s4)  # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.relu(self.deconv2(score))  # size=[n, 256, x.h/8, x.w/8]
        score = self.bn2(score + x3 * s3)
        score = self.relu(self.deconv3(score))  # size=[n, 128, x.h/4, x.w/4]
        score = self.bn3(score + x2 * s2)
        score = self.relu(self.deconv4(score))  # size=[n, 64, x.h/2, x.w/2]
        score = self.bn4(score + x1 * s1)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=[n, 32, x.h, x.w]
        score = self.classifier(score)  # size=[n, n_class, x.h, x.w]

        return score


class FCN1(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1,
                                          output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
        x4 = output['x4']  # size=[n, 512, x.h/16, x.w/16]
        x3 = output['x3']  # size=[n, 512, x.h/8, x.w/8]
        x2 = output['x2']  # size=[n, 512, x.h/4, x.w/4]
        x1 = output['x1']  # size=[n, 512, x.h/2, x.w/2]

        score = self.relu(self.deconv1(x5))  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn1(score + x4 * 0.4)  # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.relu(self.deconv2(score))  # size=[n, 256, x.h/8, x.w/8]
        score = self.bn2(score + x3 * 0.3)
        score = self.relu(self.deconv3(score))  # size=[n, 128, x.h/4, x.w/4]
        score = self.bn3(score + x2 * 0.2)
        score = self.relu(self.deconv4(score))  # size=[n, 64, x.h/2, x.w/2]
        score = self.bn4(score + x1 * 0.1)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=[n, 32, x.h, x.w]
        score = self.classifier(score)  # size=[n, n_class, x.h, x.w]

        return score


class FCNs(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.S1 = SEGet(64, 16)
        self.S2 = SEGet(128, 16)
        self.S3 = SEGet(256, 16)

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x4 = output['x4']  # size=[n, 512, x.h/16, x.w/16]
        x3 = output['x3']  # size=[n, 512, x.h/8, x.w/8]
        x2 = output['x2']  # size=[n, 512, x.h/4, x.w/4]
        x1 = output['x1']  # size=[n, 512, x.h/2, x.w/2]

        S1 = torch.sum(torch.sum(self.S1(x1)))
        S2 = torch.sum(torch.sum(self.S2(x2)))
        S3 = torch.sum(torch.sum(self.S3(x3)))

        total_sum = S1 + S2 + S3

        s1 = torch.div(S1, total_sum)
        s2 = torch.div(S2, total_sum)
        s3 = torch.div(S3, total_sum)

        score = self.relu(self.deconv1(x4))  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn1(score + x3 * s3)  # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.relu(self.deconv2(score))  # size=[n, 256, x.h/8, x.w/8]
        score = self.bn2(score + x2 * s2)
        score = self.relu(self.deconv3(score))  # size=[n, 128, x.h/4, x.w/4]
        score = self.bn3(score + x1 * s1)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.classifier(score)  # size=[n, n_class, x.h, x.w]

        return score


ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg_self': ((0, 5), (5, 10), (10, 17), (17, 24)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg_self': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            if model == 'vgg_self':
                pass
            else:
                exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output
