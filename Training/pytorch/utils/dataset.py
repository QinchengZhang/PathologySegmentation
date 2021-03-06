# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-16 14:03:12
LastEditors: TJUZQC
LastEditTime: 2020-11-20 19:23:20
Description: None
'''
import logging
from glob import glob
from os import listdir
from os.path import join, splitext

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, train=True, classes=2):
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
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale) -> np.ndarray:
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocess_mask(cls, pil_img, scale, classes=2) -> np.ndarray:
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if classes <= 2:
            if img_trans.max() > 1:
                img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(join(self.masks_dir, idx + '.*[png,jpg,tiff,tif]'))
        img_file = glob(join(self.imgs_dir, idx + '.*[png,jpg,tiff,tif]'))
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        img = self.preprocess(img, self.scale)
        mask = self.preprocess_mask(mask, self.scale, self.classes)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask)}
