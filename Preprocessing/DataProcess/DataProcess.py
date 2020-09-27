# -*- coding: utf-8 -*-
'''
@Description: 
@Autor: TJUZQC
@Date: 2020-05-13 10:48:10
LastEditors: TJUZQC
LastEditTime: 2020-09-10 15:33:10
'''
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

import cv2

patch_path = 'patch'
patch_1024_path = 'patch_1024'
if not os.path.exists(patch_1024_path):
    os.mkdir(patch_1024_path)
if not os.path.exists(os.path.join(patch_1024_path, 'imgs')):
    os.mkdir(os.path.join(patch_1024_path, 'imgs'))
if not os.path.exists(os.path.join(patch_1024_path, 'masks')):
    os.mkdir(os.path.join(patch_1024_path, 'masks'))
pbar = tqdm(glob.glob(os.path.join(patch_path, 'imgs', '*.png')))
for name in pbar:
    pbar.set_description("Processing {}:".format(name))
    file_name = os.path.basename(name)
    img = Image.open(name)
    img = img.resize((1024, 1024))
    img.save(os.path.join(patch_1024_path, 'imgs', file_name))
    del img
    mask = Image.open(os.path.join(patch_path, 'masks', file_name))
    mask = mask.resize((1024, 1024))
    mask.save(os.path.join(patch_1024_path, 'masks', file_name))
    del mask