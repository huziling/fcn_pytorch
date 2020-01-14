from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from .sync_batchnorm import SynchronizedBatchNorm2d
from .se_module import SELayer, SEGet
from .Convolution import Conv2d
"""

"""


# def get_upsample_weight(in_channels, out_channels, kernel_size):
#     '''
#     make a 2D bilinear kernel suitable for upsampling
#     '''
#     factor = (kernel_size + 1) // 2
#     if kernel_size % 2 == 1:
#         center = factor - 1
#     else:
#         center = factor - 0.5
#     og = np.ogrid[:kernel_size, :kernel_size]  # list (64 x 1), (1 x 64)
#     filt = (1 - abs(og[0] - center) / factor) * \
#            (1 - abs(og[1] - center) / factor)  # 64 x 64
#     weight = np.zeros((in_channels, out_channels, kernel_size,
#                        kernel_size), dtype=np.float64)
#     weight[range(in_channels), range(out_channels), :, :] = filt

#     return torch.from_numpy(weight).float()


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

        score = self.relu(self.deconv1(x5))                  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn1(score + x4)                         # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.bn2(self.relu(self.deconv2(score)))     # size=[n, 256, x.h/8, x.w/8]
        score = self.bn3(self.relu(self.deconv3(score)))     # size=[n, 128, x.h/4, x.w/4]
        score = self.bn4(self.relu(self.deconv4(score)))     # size=[n, 64, x.h/2, x.w/2]
        score = self.bn5(self.relu(self.deconv5(score)))     # size=[n, 32, x.h, x.w]
        score = self.classifier(score)                       # size=[n, n_class, x.h, x.w]

        return score

class FCN_8s(nn.Module):
    def __init__(self, pretrained_net, n_class,upstride = 4):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = SynchronizedBatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = SynchronizedBatchNorm2d(512)
        if upstride == 4:
            self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn3 = SynchronizedBatchNorm2d(256)
            self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn4 = SynchronizedBatchNorm2d(128)
            self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn5 = SynchronizedBatchNorm2d(64)
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)
        self._init_weights()

    def forward(self, x):
        # print(x.size())
        output = self.pretrained_net.forward(x)
        x4 = output['x4']  # size=[n, 2048, x.h/32, x.w/32]
        x3 = output['x3']  # size=[n, 1024, x.h/16, x.w/16]
        x2 = output['x2']  # size=[n, 512, x.h/8, x.w/8]

        score = self.relu(self.deconv1(x4))                  # size=[n, 512, x.h/16, x.w/16]
        # print(score.size(),x3.size())
        score = self.bn1(score + x3)                         # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.relu(self.deconv2(score))               # size=[n, 256, x.h/8, x.w/8]
        score = self.bn2(score+x2)
        score = self.bn3(self.relu(self.deconv3(score)))     # size=[n, 128, x.h/4, x.w/4]
        score = self.bn4(self.relu(self.deconv4(score)))     # size=[n, 64, x.h/2, x.w/2]
        score = self.bn5(self.relu(self.deconv5(score)))     # size=[n, 32, x.h, x.w]
        score = self.classifier(score)                       # size=[n, n_class, x.h, x.w]

        return score

    def _init_weights(self):
        '''
        hide method, used just in class
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                # assert m.kernel_size[0] == m.kernel_size[1]
                torch.nn.init.kaiming_normal_(m.weight)
                # initial_weight = get_upsample_weight(m.in_channels,
                #             m.out_channels, m.kernel_size[0])
                # m.weight.data.copy_(initial_weight)                 # copy not = ?
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class FCN1s_resnet(nn.Module):
    def __init__(self, pretrained_net, n_class, Time=False, Space=False):
        super().__init__()
        self.n_class = n_class
        self.Time = Time
        self.Space = Space
        self.pretrained_net = pretrained_net
        if self.Space:
            self.S1 = SEGet(256, 16)
            self.S2 = SEGet(512, 16)
            self.S3 = SEGet(1024, 16)
            self.S4 = SEGet(2048, 16)

        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.se1 = SELayer(1024, 16)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.se2 = SELayer(512, 16)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.se3 = SELayer(256, 16)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.se4 = SELayer(128, 16)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.se5 = SELayer(64, 16)
        self.bn5 = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x4 = output['x4']  # size=[n, 2048, x.h/32, x.w/32]
        x3 = output['x3']  # size=[n, 1024, x.h/16, x.w/16]
        x2 = output['x2']  # size=[n, 512, x.h/8, x.w/8]
        x1 = output['x1']  # size=[n, 256, x.h/4, x.w/4]

        if self.Space:
            S1 = self.S2(x1)
            S2 = self.S3(x2)
            S3 = self.S4(x3)
            S4 = self.S5(x4)

            # if self.Time:

        score = self.deconv1(x4)
        if self.Space:
            score = score*S4
        if self.Time:
            score = self.se1(score)
        score = self.relu(score)
        score = self.bn1(score + x3)
        if self.Space:
            score = score * S3
        score = self.deconv2(score)
        if self.Time:
            score = self.se2(score)
        score = self.relu(score)

        score = self.bn2(score + x2)
        if self.Space:
            score = score * S2
        score = self.deconv3(score)
        if self.Time:
            score = self.se3(score)
        score = self.relu(score)

        score = self.bn3(score + x1)
        if self.Space:
            score = score * S1
        score = self.deconv4(score)
        if self.Time:
            score = self.se4(score)
        score = self.relu(score)

        score = self.bn4(score)
        score = self.deconv5(score)
        if self.Time:
            score = self.se5(score)
        score = self.bn5(score)
        score = self.classifier(score)

        return score

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
        self.smooth_conv1 = Conv2d(1024,512,kernel_size=3,same_padding=True)
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


# class FCN8s(nn.Module):
#     def __init__(self, pretrained_net, n_class):
#         super().__init__()
#         self.n_class = n_class
#         self.pretrained_net = pretrained_net
#         self.relu = nn.ReLU(inplace=True)
#         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn1 = nn.BatchNorm2d(512)
#         self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
#         self.bn5 = nn.BatchNorm2d(32)
#         self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

#         # self._init_weights()
#         #
#         # 1
#         init.xavier_uniform_(self.deconv1.weight)
#         # 2
#         init.xavier_uniform_(self.deconv2.weight)
#         # 3
#         init.xavier_uniform_(self.deconv3.weight)
#         init.xavier_uniform_(self.deconv4.weight)
#         init.xavier_uniform_(self.deconv5.weight)
#         init.xavier_uniform_(self.classifier.weight)

#     def forward(self, x):
#         output = self.pretrained_net.forward(x)
#         x5 = output['x5']  # size=[n, 512, x.h/32, x.w/32]
#         x4 = output['x4']  # size=[n, 512, x.h/16, x.w/16]
#         x3 = output['x3']  # size=[n, 512, x.h/8, x.w/8]

#         score = self.relu(self.deconv1(x5))                  # size=[n, 512, x.h/16, x.w/16]
#         score = self.bn1(score + x4)                         # element-wise add, size=[n, 512, x.h/16, x.w/16]
#         score = self.relu(self.deconv2(score))               # size=[n, 256, x.h/8, x.w/8]
#         score = self.bn2(score+x3)
#         score = self.bn3(self.relu(self.deconv3(score)))     # size=[n, 128, x.h/4, x.w/4]
#         score = self.bn4(self.relu(self.deconv4(score)))     # size=[n, 64, x.h/2, x.w/2]
#         score = self.bn5(self.relu(self.deconv5(score)))     # size=[n, 32, x.h, x.w]
#         score = self.classifier(score)                       # size=[n, n_class, x.h, x.w]

#         return score

#     def _init_weights(self):
#         '''
#         hide method, used just in class
#         '''
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 m.weight.data.zero_()
#                 # if m.bias is not None:
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.ConvTranspose2d):
#                 assert m.kernel_size[0] == m.kernel_size[1]
#                 initial_weight = get_upsample_weight(m.in_channels,
#                             m.out_channels, m.kernel_size[0])
#                 m.weight.data.copy_(initial_weight)                 # copy not = ?


