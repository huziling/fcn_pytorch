#!/usr/bin/env python

import collections
import os.path as osp
import os
import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import cv2
import random
from XMLContext import GetContext
import configparser

cf = configparser.ConfigParser()
cf.read("./config.ini")  # 读取配置文件，如果写文件的绝对路径，就可以不用os模

max_height = int(cf.get("develop", "max_height"))

"""
https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/datasets/voc.py
"""

class DSSESegBase(data.Dataset):

    class_names_create = ['__background__', 'figure', 'table', 'text']
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])


    def __init__(self, root, split='test', transform=True,PType = '.jpg'):
        self.root = root
        self.split = split
        self._transform = transform
        self.PType = PType

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, 'DSSE')
        # dataset_dir = osp.join(self.root, 'VOC2007')

        self.files = collections.defaultdict(list)
        for img_name in os.listdir(osp.join(dataset_dir,'JPEGImages')):
            img_file = osp.join(dataset_dir, 'JPEGImages/%s' % img_name)
            lbl_name = img_name.replace('.jpg','_gt.npy')
            lbl_file = osp.join(dataset_dir, 'Annotation_img/%s' % lbl_name)
            self.files[split].append({
                'img': img_file,
                'lbl': lbl_file,
                })
        # for split_file in ['train', 'val']:
        #     imgsets_file = osp.join(
        #         dataset_dir, 'ImageSets/Segmentation/%s.txt' % split_file)
        #     for img_name in open(imgsets_file):
        #         img_name = img_name.strip()
        #         img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % img_name)
        #         lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % img_name)
        #         self.files[split_file].append({
        #             'img': img_file,
        #             'lbl': lbl_file,
        #         })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]    # 数据
        # load image
        img_file = data_file['img']
        # print(img_file)
        img = PIL.Image.open(img_file)
        if img.mode != 'RGB':
            img = img.convert("RGB")
        img = np.array(img, dtype=np.uint8)
        # load label
        # print(img.shape)
        
        lbl_file = data_file['lbl']
        # target = (GetContext(lbl_file, PType=self.PType))
        # lbl = target.numpy()
        lbl = np.load(lbl_file)
        img,lbl = self.resize(img,lbl)
        # print(lbl.shape,img.shape)
       
        # lbl = PIL.Image.open(lbl_file)
        # lbl = np.array(lbl, dtype=np.uint8)

        # lbl[lbl == 255] = 0
        # augment

        if self._transform:
            return self.transform(img, lbl,img_file)
        else:
            return img, lbl,img_file


    def transform(self, img, lbl,img_file):
        img = img[:, :, ::-1]          # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)   # whc -> cwh
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl,img_file

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)   # cwh -> whc
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]          # BGR -> RGB
        lbl = lbl.numpy()
        return img, lbl
    def resize(self, img, label):
        # print(s, img.shape)
        # h = int(max_height / 64) * 64
        # w = int(img.shape[1] * (max_height / img.shape[0]) / 64) * 64
        img = cv2.resize(img, (512, 384), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (512, 384), interpolation=cv2.INTER_NEAREST)
        return img, label
    # elif not self.predict: # for batch test, this is needed
    #     img, label = self.randomCrop(img, label)
    #     img, label = self.resize(img, label, VOCClassSeg.img_size)
    # else:
    #     pass


class DSSEClassSeg(DSSESegBase):

    # url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False,PType='.jpg'):
        super(DSSEClassSeg, self).__init__(root, split=split, transform=transform,PType=PType)


"""
vocbase = VOC2012ClassSeg(root="/home/yxk/Downloads/")

print(vocbase.__len__())
img, lbl = vocbase.__getitem__(0)
img = img[:, :, ::-1]
img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_LINEAR)
print(np.shape(img))
print(np.shape(lbl))

"""


