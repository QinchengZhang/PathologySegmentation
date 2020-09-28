# -*- coding: utf-8 -*-
'''
Description: 
Autor: TJUZQC
Date: 2020-09-28 11:37:08
LastEditors: TJUZQC
LastEditTime: 2020-09-29 00:31:32
'''
import glob
import os
import threading

import multiresolutionimageinterface as mir
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as I

from Augmentation import augmentOps

IMAGEEXT = ['.png', '.jpg', '.tif', '.tiff', '.gif', '.bmp']


class NDPIImage(object):
    def __init__(self, path: str):
        reader = mir.MultiResolutionImageReader()
        try:
            self.img = reader.open(path)
        except FileNotFoundError as e:
            raise('file not found!')

    def __del__(self):
        self.img.close()

    def getNumberOfLevels(self):
        return self.img.getNumberOfLevels()

    def getDimensions(self):
        return self.img.getDimensions()

    def getLevelDimensions(self, level: int):
        return self.img.getLevelDimensions(level)

    def getLevelDownsample(self, level: int):
        self.img.getLevelDownsample(level)

    def getMinValue(self, channel: int = -1):
        self.img.getMinValue(channel)

    def getMaxValue(self, channel: int = -1):
        self.img.getMaxValue(channel)

    def close(self):
        self.__del__()

    def getPatch(self, posX: int, posY: int, width: int, height: int, level: int, dtype: type = np.uint8):
        return np.array(self.img.getUInt16Patch(posX, posY, width, height, level), dtype=dtype)

    def toTensor(self, level: int):
        min_width, min_height = self.getLevelDimensions(level)
        return self.getPatch(0, 0, min_width, min_height, level)

    def toImage(self, level: int):
        min_width, min_height = self.getLevelDimensions(level)
        return Image(array=self.getPatch(0, 0, min_width, min_height, level))

    def getSpacing(self):
        return self.img.getSpacing()

    def WritetoImage(self, output_path:str):
        writer = mir.MultiResolutionImageWriter()
        writer.writeImageToFile(self.img, output_path)


class Image(object):
    def __init__(self, path: str = None, array: np.ndarray = None):
        assert path is not None or array is not None, 'argument must be a path or an array'
        assert path is None or array is None, 'argument must be a path or an array'
        if path is not None:
            try:
                self.img = I.open(path)
            except FileNotFoundError as e:
                raise('file not found!')
        if array is not None:
            self.img = I.fromarray(array)
        self.size = self.img.size

    def __str__(self):
        return self.img.__str__()

    def getArray(self):
        return np.array(self.img)

    def save(self, path: str):
        assert os.path.splitext(
            path)[-1] in IMAGEEXT, f'image file extension name must in {IMAGEEXT}'
        pass

    def close(self):
        return self.__del__()

    def resize(self, width, height):
        self.img = self.img.resize((width, height))
        self.size = self.img.size


class XMLLabel(object):
    def __init__(self, path: str):
        assert os.path.splitext(
            path)[-1] == '.xml', f'xml file extension name must is xml'
        self.annotation_list = mir.AnnotationList()
        xml_repository = mir.XmlRepository(self.annotation_list)
        try:
            xml_repository.setSource(path)
            xml_repository.load()
        except FileNotFoundError as e:
            raise('file not found!')

    def __str__(self):
        return self.annotation_list.__str__()

    def toMask(self, output_path, dimensions, spacing, label_map, conversion_order):
        annotation_mask = mir.AnnotationToMask()
        annotation_mask.convert(self.annotation_list, output_path, img.getDimensions(
        ), img.getSpacing(), label_map, conversion_order)


if __name__ == '__main__':
    # item = WSIItem('F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.ndpi',
    #                xml_path='F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.xml')

    img = NDPIImage(
        'F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.ndpi')
    xml = XMLLabel(
        'F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.xml')
    img.WritetoImage('F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.tiff')
    print(xml)
