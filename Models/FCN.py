from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import torch.nn.functional as F
from torch.nn import init
import numpy as np

"""

"""


def get_upsample_weight(in_channels, out_channels, kernel_size):
    '''
    make a 2D bilinear kernel suitable for upsampling
    '''
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
        self.bn1 = nn.SyncBatchNorm(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.SyncBatchNorm(512)
        if upstride == 4:
            self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn3 = nn.SyncBatchNorm(256)
            self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            self.bn4 = nn.SyncBatchNorm(128)
            # self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
            # self.bn5 = nn.SyncBatchNorm(32)
        self.classifier = nn.Conv2d(128, n_class, kernel_size=1)
        self._init_weights()

    def forward(self, x):
        output = self.pretrained_net.forward(x)
        x4 = output['x4']  # size=[n, 2048, x.h/16, x.w/16]
        x3 = output['x3']  # size=[n, 1024, x.h/8, x.w/8]
        x2 = output['x2']  # size=[n, 512, x.h/8, x.w/8]

        score = self.relu(self.deconv1(x4))                  # size=[n, 512, x.h/16, x.w/16]
        score = self.bn1(score + x3)                         # element-wise add, size=[n, 512, x.h/16, x.w/16]
        score = self.relu(self.deconv2(score))               # size=[n, 256, x.h/8, x.w/8]
        score = self.bn2(score+x2)
        score = self.bn3(self.relu(self.deconv3(score)))     # size=[n, 128, x.h/4, x.w/4]
        score = self.bn4(self.relu(self.deconv4(score)))     # size=[n, 64, x.h/2, x.w/2]
        # score = self.bn5(self.relu(self.deconv5(score)))     # size=[n, 32, x.h, x.w]
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
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsample_weight(m.in_channels,
                            m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)                 # copy not = ?
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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


