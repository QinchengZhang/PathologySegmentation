# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-24 16:18:01
LastEditors: TJUZQC
LastEditTime: 2020-11-24 17:15:53
Description: None
'''
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob

from paddleseg.datasets import Dataset
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class RemoteSensing(Dataset):
    """
    RemoteSensing dataset `https://www.datafountain.cn/competitions/475/datasets`.
    The folder structure is as follow:

        RemoteSensing
        |
        |--img_train
        |
        |--lab_train
        |
        |--train_list.txt
        |
        |--val_liat.txt

    Args:
        transforms (list): Transforms for image.
        dataset_root (str): Cityscapes dataset directory.
        mode (str): Which part of dataset to use. it is one of ('train', 'val'). Default: 'train'.
    """

    def __init__(self, transforms, dataset_root, mode='train'):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        self.file_list = list()
        self.mode = mode if mode in ['train', 'val'] else 'test'
        self.num_classes = 7
        self.ignore_index = 255

        if mode not in ['train', 'val', 'testA', 'testB']:
            raise ValueError(
                "mode should be 'train', 'val', 'testA' or 'testB', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        img_dir = os.path.join(self.dataset_root, 'img_train')
        label_dir = os.path.join(self.dataset_root, 'lab_train')
        list_file = open(os.path.join(dataset_root, f'{mode}_list.txt'))
        if self.dataset_root is None or not os.path.isdir(
                self.dataset_root) or not os.path.isdir(
                    img_dir) or not os.path.isdir(label_dir):
            raise ValueError(
                "The dataset is not Found or the folder structure is nonconfoumance."
            )
        
        

        # label_files = sorted(
        #     glob.glob(
        #         os.path.join(label_dir, mode, '*',
        #                      '*_gtFine_labelTrainIds.png')))
        # img_files = sorted(
        #     glob.glob(os.path.join(img_dir, mode, '*', '*_leftImg8bit.png')))
        if self.mode != 'test':
            self.file_list = [img_lab_path.strip().split(' ') for img_lab_path in list_file.readlines()]
        
        else:
            self.file_list = [[
                img_path.strip(), ''
            ] for img_path in list_file.readlines()]
