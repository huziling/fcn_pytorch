#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import torch
from torch.utils import data
import cv2
from XMLContext import GetContext
import configparser
import random
cf = configparser.ConfigParser()
cf.read("./config.ini")  # 读取配置文件，如果写文件的绝对路径，就可以不用os模

max_height = int(cf.get("develop", "max_height"))


class PDFClassSegBase(data.Dataset):
    def __init__(self, root, split, transform, filePath, PType):
        self.root = root
        self.split = split
        self._transform = transform
        self.filePath = filePath
        self.PType = PType
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

        dataset_dir = osp.join(self.root)
        self.files = collections.defaultdict(list)

        imgsets_file = osp.join(dataset_dir, 'layout/%s.txt' % split)
        for img_name in open(imgsets_file):
            img_name = img_name.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % img_name)
            lbl_file = osp.join(dataset_dir, 'Annotation_pic/%s.npy' % img_name)
            self.files[split].append({
                'img': img_file,
                'lbl': lbl_file,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]  # 数据
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        # target = (GetContext(lbl_file, PType=self.PType))
        # lbl = target.numpy()
        lbl = np.load(lbl_file)
        # augment
        # h = int(max_height / 64) * 64
        # w = int(img.shape[1] * (max_height / img.shape[0]) / 64) * 64

        # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        # lbl = cv2.resize(lbl, (w, h), interpolation=cv2.INTER_LINEAR)
        img,lbl = self.augmentation(img,lbl)
        if self._transform:
            return self.transform(img, lbl, img_file)
        else:
            return img, lbl, img_file

    def transform(self, img, lbl, img_file):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # whc -> cwh
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl, img_file

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)  # cwh -> whc
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]  # BGR -> RGB
        lbl = lbl.numpy()

        return img, lbl

    def randomFlip(self, img, label):
        if random.random() < 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)
        return img, label
    def resize(self, img, label):
        # print(s, img.shape)
        h = int(max_height / 64) * 64
        w = int(img.shape[1] * (max_height / img.shape[0]) / 64) * 64
        img = cv2.resize(img, (h, w), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST)
        return img, label

    def randomCrop(self, img, label):
        h, w, _ = img.shape
        short_size = min(w, h)
        rand_size = random.randrange(int(0.7 * short_size), short_size)
        x = random.randrange(0, w - rand_size)
        y = random.randrange(0, h - rand_size)

        return img[y:y + rand_size, x:x + rand_size], label[y:y + rand_size, x:x + rand_size]
    # data augmentaion
    def augmentation(self, img, lbl):
        img, lbl = self.randomFlip(img, lbl)
        #img, lbl = self.randomCrop(img, lbl)
        img, lbl = self.resize(img, lbl)
        return img, lbl

class ClassSeg(PDFClassSegBase):
    def __init__(self, root, split='train', transform=False, filePath='', PType='.jpg'):
        super(ClassSeg, self).__init__(root, split=split, transform=transform, filePath=filePath, PType=PType)
