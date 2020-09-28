# -*- coding: utf-8 -*-
'''
@Description: 
@Autor: TJUZQC
@Date: 2020-05-13 10:48:10
LastEditors: TJUZQC
LastEditTime: 2020-09-28 11:35:45
'''
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

import cv2

def __batch_resize(file_list, path_dst, start, end):
    if not os.path.exists(path_dst):
        os.mkdir(path_dst)
    if not os.path.exists(os.path.join(path_dst, 'imgs')):
        os.mkdir(os.path.join(path_dst, 'imgs'))
    if not os.path.exists(os.path.join(path_dst, 'masks')):
        os.mkdir(os.path.join(path_dst, 'masks'))
    print('getting start from {} to {}'.format(start, end))
    pbar = tqdm(file_list[start : end])
    for name in pbar:
        pbar.set_description("Processing {}:".format(name))
        file_name = os.path.basename(name)
        img = Image.open(name)
        img = img.resize((1024, 1024))
        img.save(os.path.join(path_dst, 'imgs', file_name))
        del img
        mask = Image.open(os.path.join(file_list, 'masks', file_name))
        mask = mask.resize((1024, 1024))
        mask.save(os.path.join(path_dst, 'masks', file_name))
        del mask