# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-05 12:29:09
LastEditors: TJUZQC
LastEditTime: 2021-01-05 13:49:49
Description: None
'''
from paddle.vision import transforms
from paddle.vision.transforms import BaseTransform
from paddle.vision.transforms import functional as F
import random


class FlipTransform(BaseTransform):
    def __init__(self, keys=None):
        super(FlipTransform, self).__init__(keys)

    def _apply_image(self, image):
        return F.hflip(image)
    
    def _apply_mask(self, mask):
        return F.hflip(mask)


class MirrorTransform(BaseTransform):
    def __init__(self, keys=None):
        super(MirrorTransform, self).__init__(keys)

    def _apply_image(self, image):
        return F.vflip(image)
    
    def _apply_mask(self, mask):
        return F.vflip(mask)


class RandomRotateTransform(BaseTransform):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0, keys=None):
        super(RandomRotateTransform, self).__init__(keys)
        assert isinstance(degrees, (tuple, list, int)), f'Argument degrees expects tuple, list or int but got {type(degrees)}'
        if isinstance(degrees, (tuple, list)):
            assert len(degrees) == 2, 'The length of argument degrees must be 2'
        self.degrees = random.randint(degrees[0], degrees[1]) if isinstance(degrees, (tuple, list)) else random.randint(-degrees, degrees)
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    def _apply_image(self, image):
        return F.rotate(image, self.degrees, self.resample, self.expand, self.center, self.fill)
    
    def _apply_mask(self, mask):
        return F.rotate(mask, self.degrees, self.resample, self.expand, self.center, self.fill)



class PaddingTransform(BaseTransform):
    def __init__(self, padding, fill=0, padding_mode='constant', keys=None):
        super(PaddingTransform, self).__init__(keys)
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def _apply_image(self, image):
        return F.pad(image, self.padding, self.fill, self.padding_mode)
    
    def _apply_mask(self, mask):
        return F.pad(mask, self.padding, 0)


class RandomOffsetTransform(BaseTransform):
    def __init__(self, offset, keys=None):
        super(RandomOffsetTransform, self).__init__(keys)
        assert isinstance(offset, (tuple, list, int)), f'Argument offset expects tuple, list or int but got {type(offset)}'
        if isinstance(offset, (tuple, list)):
            assert len(offset) == 2, 'The length of argument offset must be 2'
            if isinstance(offset[0], (tuple, list)):
                assert len(offset) == 2, 'The length of argument x_offset must be 2'
                self.x_offset = random.randint(offset[0][0], offset[0][1])
            else:
                self.x_offset = random.randint(-offset[0], offset[0])
                
            if isinstance(offset[1], (tuple, list)):
                assert len(offset) == 2, 'The length of argument y_offset must be 2'
                self.y_offset = random.randint(offset[1][0], offset[1][1])
            else:
                self.y_offset = random.randint(-offset[1], offset[1])
        else:
            self.x_offset = random.randint(-offset, offset)
            self.y_offset = random.randint(-offset, offset)
            print(self.x_offset, self.y_offset)
        self.padding = [0, 0, 0, 0] #left, top, right, bottom
        if self.x_offset >= 0:
            self.padding[0] = self.x_offset
        else:
            self.padding[2] = -self.x_offset
        if self.y_offset >= 0:
            self.padding[3] = self.y_offset
        else:
            self.padding[1] = -self.y_offset
        self.padding = tuple(self.padding)


    def _apply_image(self, image):
        _, height, width = F.to_tensor(image).shape
        ret = F.pad(image, self.padding)
        return F.crop(ret, top=abs(self.y_offset) if self.y_offset >= 0 else 0, left=0 if self.x_offset >= 0 else abs(self.x_offset), height=height, width=width)
    
    def _apply_mask(self, mask):
        _, height, width = F.to_tensor(mask).shape
        ret = F.pad(mask, self.padding)
        return F.crop(ret, top=abs(self.y_offset) if self.y_offset >= 0 else 0, left=0 if self.x_offset >= 0 else abs(self.x_offset), height=height, width=width)

# class ResizeOps(Baclass FlipOps(BaseTransform):
#     def __init__(self, size):
#         assert isinstance(size, tuple), f'argument type error!'
#         self.size = size
#         self.name = "basicOps.ResizeOps"

#     def __call__(self, img, mask=None):
#         assert isinstance(img, Image.Image), f'argument type error!'
#         if mask is not None:
#             assert isinstance(mask, Image.Image), f'argument type error!'
#             return img.resize(self.size), mask.resize(self.size)
#         else:
#             return img.resize(self.size)

#     def __str__(self):
#         return self.name
