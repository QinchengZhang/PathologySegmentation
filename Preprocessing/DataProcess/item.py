# -*- coding: utf-8 -*-
'''
Description: 
Autor: TJUZQC
Date: 2020-09-28 11:37:08
LastEditors: TJUZQC
LastEditTime: 2020-10-09 17:36:28
'''
import glob
import os
import threading

import multiresolutionimageinterface as mir
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as I
from Augmentation import augmentOps
from libtiff import TIFF

IMAGEEXT = ['.png', '.jpg', '.tif', '.tiff', '.gif', '.bmp']


class WSIImage(object):
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
        writer.writeImageInformation(self.getDimensions()[0], self.getDimensions()[1])
        writer.writeImageToFile(self.img, output_path)


class Image(object):
    img = None
    size = None
    shape = None
    channel = None
    def __init__(self, path: str = None, array: np.ndarray = None):
        assert path is not None or array is not None, 'argument must be a path or an array'
        assert path is None or array is None, 'argument must be a path or an array'
        if path is not None:
            if os.path.splitext(path)[-1] in ['.tiff', '.tif']:
                try:
                    temp = TIFF.open(path)
                    self.img = temp.read_image()
                except FileNotFoundError as e:
                    raise('file not found!')
            else:
                try:
                    self.img = np.array(I.open(path))
                except FileNotFoundError as e:
                    raise('file not found!')
        if array is not None:
            self.img = array
        self.__getInformation()

    def __getInformation(self):
        self.shape = self.img.shape
        self.size = (self.img.shape[0],self.img.shape[1])
        self.channel = self.img.shape[2] if len(self.img.shape) == 3 else 1

    def __str__(self):
        return self.img.__str__()

    def getImage(self):
        return I.fromarray(self.img)

    def save(self, path: str):
        assert os.path.splitext(
            path)[-1] in IMAGEEXT, f'image file extension name must is one of {IMAGEEXT}'
        self.getImage().save(path)

    def close(self):
        return self.__del__()

    def resize(self, width, height):
        self.img = np.array(self.getImage().resize((width, height)))
        self.__getInformation()

class Mask(Image):
    def Max(self):
        return np.max(self.img)


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
        annotation_mask.convert(self.annotation_list, output_path, dimensions, spacing, label_map, conversion_order)
        mask = TIFF.open(output_path)
        return np.array(mask.read_image())

class Item(object):
    def __init__(self, img_path:str, xml_path:str, mask_path:str=None):
        self.img = WSIImage(img_path)
        self.xml = XMLLabel(xml_path)
        self.mask = None
        if mask_path is not None:
            self.mask = WSIImage(mask_path)

    # def getPatchWithAnnotations(self, size:(int, int)):
    #     annotations = self.xml.annotation_list.getAnnotations()
    #     for idx, annotation in enumerate(annotations):
    #         x, y, width, height = getPositionAndSize(annotation)
    #         level_0_width, level_0_height = img.getLevelDimensions(0)
    #         level_1_width, level_1_height = img.getLevelDimensions(1)
    #         # x *= level_1_width/level_0_width
    #         # y *= level_1_height/level_0_height
    #         width *= level_1_width/level_0_width
    #         height *= level_1_height/level_0_height
    #         x, y, width, height = int(x), int(y), int(width), int(height)
    #         patch_img = img.getUInt16Patch(x, y, width, height, 1)
    #         patch_img = np.array(patch_img, dtype=np.int8)
    #         patch_img = Image.fromarray(patch_img, mode='RGB')
    #         patch_img.save(os.path.join(
    #             path, 'patch', 'imgs', os.path.splitext(os.path.basename(img_name))[0]+'-{}.png'.format(idx)))
    #         del patch_img
    #         patch_mask = mask.getUInt16Patch(x, y, width, height, 1)
    #         patch_mask = np.array(patch_mask, dtype=np.int8)
    #         patch_mask = Image.fromarray(patch_mask[:, :, 0], mode='L')
    #         patch_mask.save(os.path.join(
    #             path, 'patch', 'masks', os.path.splitext(os.path.basename(img_name))[0]+'-{}.png'.format(idx)))
    #         del patch_mask


if __name__ == '__main__':
    # img = Mask('F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42_mask.tiff')
    item = Item('F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.ndpi', 'F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.xml', 'F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42_mask.tiff')
    print(item)
    # img = NDPIImage(
    #     'F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.ndpi')
    # # # print(img.getSpacing())
    # xml = XMLLabel(
    #     'F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.xml')
    # label_map = {'Annotation Group 0': 255}
    # conversion_order = ['Annotation Group 0']
    # print(img.getDimensions())
    # mask = xml.toMask('F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42_mask.tiff', img.getDimensions(), img.getSpacing(), label_map, conversion_order)
    # # # img.WritetoImage('F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.tiff')
    # print(mask)
