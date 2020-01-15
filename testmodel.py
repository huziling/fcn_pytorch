import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import models
import loss
from torch.optim import Adam, SGD
from torchvision import transforms
import configparser
import logging
from sklearn import metrics
import time
from Model import DeepLab,resnet50,FCN1s_resnet,sk_resnet18,resnet18
from torchsummary import summary


if __name__ == '__main__':
    model = resnet18()
    for name, m in model.named_modules():
        if isinstance(m,nn.Conv2d):
            print(name,m)