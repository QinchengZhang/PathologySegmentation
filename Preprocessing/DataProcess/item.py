# -*- coding: utf-8 -*-
'''
Description: 
Autor: TJUZQC
Date: 2020-09-28 11:37:08
LastEditors: TJUZQC
LastEditTime: 2020-11-20 19:24:50
'''
import glob
import os
import threading
from multiprocessing import cpu_count

import multiresolutionimageinterface as mir
import numpy as np
from libtiff import TIFF
from PIL import Image as I
from tqdm import tqdm

IMAGEEXT = ['.png', '.jpg', '.tif', '.tiff', '.gif', '.bmp']


class WSIImage(object):
    def __init__(self, path: str):
        reader = mir.MultiResolutionImageReader()
        self.img = reader.open(path)
        if self.img is None:
            raise('file not found!')
        self.name = os.path.basename(path)

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

    def WritetoImage(self, output_path: str):
        writer = mir.MultiResolutionImageWriter()
        writer.writeImageInformation(
            self.getDimensions()[0], self.getDimensions()[1])
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
        self.size = (self.img.shape[0], self.img.shape[1])
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


class Mask(WSIImage):
    pass


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
        annotation_mask.convert(self.annotation_list, output_path,
                                dimensions, spacing, label_map, conversion_order)
        return Mask(output_path)


class Item(object):
    def __init__(self, img_path: str, xml_path: str, mask_path: str = None):
        self.img = WSIImage(img_path)
        self.xml = XMLLabel(xml_path)
        self.mask = None
        if mask_path is not None:
            self.mask = Mask(mask_path)

    def getMask(self, label_map: dict, conversion_order: list, output_path: str = None):
        if output_path is None:
            output_path = 'temp.tiff'
        self.mask = self.xml.toMask(output_path, self.img.getDimensions(
        ), self.img.getSpacing(), label_map, conversion_order)
        return self.mask

        # 从xml标注中得到一个Annotation的边界信息，所得边界比真实边界大'margin'px
    def __getPositionAndSize(self, annotation, margin):
        X_min = None
        Y_min = None
        X_max = None
        Y_max = None
        for coordinate in annotation.getCoordinates():
            if X_max is None or coordinate.getX() > X_max:
                X_max = coordinate.getX()
            if X_min is None or coordinate.getX() < X_min:
                X_min = coordinate.getX()
            if Y_min is None or coordinate.getY() < Y_min:
                Y_min = coordinate.getY()
            if Y_max is None or coordinate.getY() > Y_max:
                Y_max = coordinate.getY()
        return int(X_min)-margin, int(Y_min)-margin, int(X_max - X_min)+margin*2, int(Y_max - Y_min) + margin*2

    # def __getPatchWithAnnotations(self, annotations, size, margin, start, end):
    #     retval = []
    #     if end is None:
    #         end = len(annotations)+1
    #     with tqdm(annotations[start:end]) as pbar:
    #         for idx, annotation in enumerate(pbar):
    #             pbar.set_description(
    #                 f'Processing annotation {idx+1}/{len(annotations)}')
    #             x, y, width, height = self.__getPositionAndSize(
    #                 annotation, margin)
    #             level_0_width, level_0_height = self.img.getLevelDimensions(0)
    #             level_1_width, level_1_height = self.img.getLevelDimensions(1)
    #             # x *= level_1_width/level_0_width
    #             # y *= level_1_height/level_0_height
    #             # width *= level_1_width/level_0_width
    #             # height *= level_1_height/level_0_height
    #             x, y, width, height = int(x), int(y), int(width), int(height)
    #             patch_img = self.img.getPatch(x, y, width, height, 0)
    #             patch_img = np.array(patch_img, dtype=np.int8)

    #             patch_mask = self.mask.getPatch(x, y, width, height, 0)
    #             patch_mask = np.array(patch_mask, dtype=np.int8)
    #             if size is not None:
    #                 patch_img = I.fromarray(patch_img, mode="RGB").resize(size)
    #                 patch_mask = I.fromarray(
    #                     patch_mask[:, :, 0], mode="L").resize(size)
    #             else:
    #                 patch_img = I.fromarray(patch_img, mode="RGB")
    #                 patch_mask = I.fromarray(
    #                     patch_mask[:, :, 0], mode="L")
    #             retval.append({'img': np.array(patch_img),
    #                            'mask': np.array(patch_mask)})
    #     return retval

    def getPatchWithAnnotations(self, size: (int, int) = None, border: int = 100):
        annotations = self.xml.annotation_list.getAnnotations()
        retval = []
        with tqdm(annotations) as pbar:
            for idx, annotation in enumerate(pbar):
                pbar.set_description(
                    f'Processing annotation {idx+1}/{len(annotations)}')
                x, y, width, height = self.__getPositionAndSize(
                    annotation, border)
                level_0_width, level_0_height = self.img.getLevelDimensions(0)
                level_1_width, level_1_height = self.img.getLevelDimensions(1)
                # x *= level_1_width/level_0_width
                # y *= level_1_height/level_0_height
                # width *= level_1_width/level_0_width
                # height *= level_1_height/level_0_height
                x, y, width, height = int(x), int(y), int(width), int(height)
                patch_img = self.img.getPatch(x, y, width, height, 0)
                patch_img = np.array(patch_img, dtype=np.int8)

                patch_mask = self.mask.getPatch(x, y, width, height, 0)
                patch_mask = np.array(patch_mask, dtype=np.int8)
                if size is not None:
                    patch_img = I.fromarray(patch_img, mode="RGB").resize(size)
                    patch_mask = I.fromarray(
                        patch_mask[:, :, 0], mode="L").resize(size)
                else:
                    patch_img = I.fromarray(patch_img, mode="RGB")
                    patch_mask = I.fromarray(
                        patch_mask[:, :, 0], mode="L")
                retval.append({'img': np.array(patch_img),
                               'mask': np.array(patch_mask)})
        return retval

    def savePatchWithAnnotations(self, output_path: str, size: (int, int) = None, border: int = 100):
        patchs = self.getPatchWithAnnotations(size, border)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        assert os.path.isdir(
            output_path), 'argument output_path must be a path but a file path was given'
        if not os.path.exists(os.path.join(output_path, 'patch_images')):
            os.mkdir(os.path.join(output_path, 'patch_images'))
        if not os.path.exists(os.path.join(output_path, 'patch_masks')):
            os.mkdir(os.path.join(output_path, 'patch_masks'))
        with tqdm(patchs) as pbar:
            for idx, item in enumerate(pbar):
                pbar.set_description(
                    f'saving annotation and WSI image {idx+1}/{len(patchs)}')
                I.fromarray(item['img']).save(os.path.join(output_path, 'patch_images', self.img.name.replace(
                    os.path.splitext(self.img.name)[-1], f'_{idx}.png')))
                I.fromarray(item['mask']).save(os.path.join(output_path, 'patch_masks', self.mask.name.replace(
                    os.path.splitext(self.mask.name)[-1], f'_{idx}.png')))


if __name__ == '__main__':
    # img = Mask('F:\DATASET\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42_mask.tiff')
    item = Item('G:\\TJUZQC\\DataSet\\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.ndpi',
                'G:\\TJUZQC\\DataSet\\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42.xml', 'G:\\TJUZQC\\DataSet\\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\2020-01-20 10.39.42_mask.tiff')
    print(item.savePatchWithAnnotations(
        'G:\\TJUZQC\\DataSet\\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42\\patchs', (512, 512), 200))
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
