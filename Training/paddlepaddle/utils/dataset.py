# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-17 10:46:07
LastEditors: TJUZQC
LastEditTime: 2020-11-17 12:48:55
Description: None
'''
from glob import glob
import logging
from os import listdir
from os.path import splitext, join

import cv2
import numpy as np
import paddle
from paddle.io import Dataset


class SegDataset(Dataset):
    """
    继承paddle.io.Dataset类
    """

    def __init__(self, imgs_dir, masks_dir, scale=1, train=True, classes=2):
        """
        实现构造函数，定义数据读取方式，划分训练和测试数据集
        """
        self.train = train
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.classes = classes
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        """
        实现__len__方法，返回数据集总数目
        """
        return len(self.ids)

    @classmethod
    def preprocess(cls, img: np.ndarray, scale: float) -> np.ndarray:
        w, h, c = img.shape
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        img = cv2.resize(img, (newW, newH))
        # img = img.resize((newW, newH))

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocess_mask(cls, img, scale, classes=2) -> np.ndarray:
        w, h, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        img = cv2.resize(img, (newW, newH))
        # img = img.resize((newW, newH))

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        if classes <= 2:
            if img_trans.max() > 1:
                img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(join(self.masks_dir, idx+'.*[png,jpg,tiff,tif]'))
        img_file = glob(join(self.imgs_dir, idx+'.*[png,jpg,tiff,tif]'))
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = cv2.imread(mask_file[0])
        img = cv2.imread(img_file[0])
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        img = self.preprocess(img, self.scale)
        mask = self.preprocess_mask(mask, self.scale, self.classes)

        return paddle.to_tensor(img), paddle.to_tensor(mask)


if __name__ == '__main__':
    # 测试定义的数据集
    train_dataset = SegDataset("G:\TJUZQC\code\python\PathologySegmentation\Training\pytorch\data\WSI\imgs",
                            "G:\TJUZQC\code\python\PathologySegmentation\Training\pytorch\data\WSI\masks")
    val_dataset = SegDataset("G:\TJUZQC\code\python\PathologySegmentation\Training\pytorch\data\WSI\imgs",
                            "G:\TJUZQC\code\python\PathologySegmentation\Training\pytorch\data\WSI\masks", train=False)

    print('=============train dataset=============')
    for data, label in train_dataset:
        print(data, label)

    print('=============evaluation dataset=============')
    for data, label in val_dataset:
        print(data, label)
