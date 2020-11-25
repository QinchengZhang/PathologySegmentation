# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-15 11:45:59
LastEditors: TJUZQC
LastEditTime: 2020-11-20 19:25:13
Description: None
'''
import random

from PIL import Image

import Augmentation.basicOps


class RandomChoice(object):
    def __init__(self, transforms):
        assert isinstance(transforms, list), 'transforms should be a list but got {}'.format(
            type(transforms))
        self.transforms = transforms

    def __call__(self, img):
        ops = random.randint(1, len(self.transforms))
        print(str(self.transforms[ops-1]))
        return self.transforms[ops-1](img)


class RandomChoiceWithMask(object):
    def __init__(self, transforms):
        assert isinstance(transforms, list), 'transforms should be a list but got {}'.format(
            type(transforms))
        self.transforms = transforms

    def __call__(self, img, mask):
        ops = random.randint(1, len(self.transforms))
        print(str(self.transforms[ops-1]))
        return self.transforms[ops-1](img, mask)


class ApplyOps(object):
    def __init__(self, transforms):
        assert isinstance(transforms, list), 'transforms should be a list but got {}'.format(
            type(transforms))
        self.transforms = transforms

    def __call__(self, img):
        ret_img = None
        for ops in self.transforms:
            ret_img = ops(img)
            print(str(ops))
        return ret_img


class ApplyOpsWithMask(object):
    def __init__(self, transforms):
        assert isinstance(transforms, list), 'transforms should be a list but got {}'.format(
            type(transforms))
        self.transforms = transforms

    def __call__(self, img, mask):
        ret_img, ret_mask = None, None
        for ops in self.transforms:
            ret_img, ret_mask = ops(img, mask)
            print(str(ops))
        return ret_img, ret_mask
