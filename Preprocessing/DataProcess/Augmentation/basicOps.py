# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-15 11:15:29
LastEditors: TJUZQC
LastEditTime: 2020-09-17 11:59:47
Description: None
'''
from torchvision import transforms
from PIL import Image, ImageOps, ImageChops
import numpy as np
import cv2


class FlipOps(object):
    def __init__(self):
        self.name = "basicOps.FlipOps"
        pass

    def __call__(self, img, mask=None):
        assert isinstance(img, Image.Image), f'argument type error!'
        if mask is not None:
            assert isinstance(mask, Image.Image), f'argument type error!'
            return ImageOps.flip(img), ImageOps.flip(mask)
        else:
            return ImageOps.flip(img)

    def __str__(self):
        return self.name


class MirrorOps(object):
    def __init__(self):
        self.name = "basicOps.MirrorOps"
        pass

    def __call__(self, img, mask=None):
        assert isinstance(img, Image.Image), f'argument type error!'
        if mask is not None:
            assert isinstance(mask, Image.Image), f'argument type error!'
            return ImageOps.mirror(img), ImageOps.mirror(mask)
        else:
            return ImageOps.mirror(img)
    
    def __str__(self):
        return self.name


class RotateOps(object):
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        self.name = "basicOps.RotateOps"
        self.degrees = degrees
        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, img, mask=None):
        assert isinstance(img, Image.Image), f'argument type error!'
        transform = transforms.RandomRotation(
            self.degrees, resample=self.resample, expand=self.expand, center=self.center, fill=self.fill)
        if mask is not None:
            assert isinstance(mask, Image.Image), f'argument type error!'
            return transform(img), transform(mask)
        else:
            return transform(img)

    def __str__(self):
        return self.name


class AffineOps(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        self.name = "basicOps.AffineOps"
        self.degress = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    def __call__(self, img, mask=None):
        assert isinstance(img, Image.Image), f'argument type error!'
        transform = transforms.RandomAffine(
            self.degress, self.translate, self.scale, self.shear, self.resample, self.fillcolor)
        if mask is not None:
            assert isinstance(mask, Image.Image), f'argument type error!'
            return transform(img), transform(mask)
        else:
            return transform(img)

    def __str__(self):
        return self.name

class PaddingOps(object):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        self.name = "basicOps.PaddingOps"
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, mask):
        assert isinstance(img, Image.Image), f'argument type error!'
        transform = transforms.Pad(
            self.padding, fill=self.fill, padding_mode=self.padding_mode)
        if mask is not None:
            assert isinstance(mask, Image.Image), f'argument type error!'
            return transform(img), transform(mask)
        else:
            return transform(img)

    def __str__(self):
        return self.name


class OffsetOps(object):
    def __init__(self, xoffset, yoffset=None):
        self.name = "basicOps.OffsetOps"
        self.xoffset = xoffset
        if yoffset == None:
            self.yoffset = xoffset

    def __call__(self, img, mask=None):
        assert isinstance(img, Image.Image), f'argument type error!'
        if mask is not None:
            assert isinstance(mask, Image.Image), f'argument type error!'
            return ImageChops.offset(img, self.xoffset, self.yoffset), ImageChops.offset(mask, self.xoffset, self.yoffset)
        else:
            return ImageChops.offset(img, self.xoffset, self.yoffset)

    def __str__(self):
        return self.name


class DoNothing(object):
    def __init__(self):
        self.name = "basicOps.DoNothing"
        pass

    def __call__(self, img, mask=None):
        assert isinstance(img, Image.Image), f'argument type error!'
        if mask is not None:
            assert isinstance(mask, Image.Image), f'argument type error!'
            return img, mask
        else:
            return img

    def __str__(self):
        return self.name

class ResizeOps(object):
    def __init__(self, size):
        assert isinstance(size, tuple), f'argument type error!'
        self.size = size
        self.name = "basicOps.ResizeOps"
    
    def __call__(self, img, mask=None):
        assert isinstance(img, Image.Image), f'argument type error!'
        if mask is not None:
            assert isinstance(mask, Image.Image), f'argument type error!'
            return img.resize(self.size), mask.resize(self.size)
        else:
            return img.resize(self.size)

    def __str__(self):
        return self.name