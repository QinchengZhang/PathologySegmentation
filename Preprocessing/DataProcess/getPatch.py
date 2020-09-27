# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-08 14:05:31
LastEditors: TJUZQC
LastEditTime: 2020-09-09 16:44:58
Description: None
'''
import multiresolutionimageinterface as mir
import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import threading
from libtiff import TIFF
from PIL import Image

# pathlist = glob.glob('2020-01-20 13.34.13')
pathlist = ['2020-01-20 13.34.13', '2020-01-21 19.55.32', '2020-01-21 20.03.41', '2020-01-21 20.31.45', '2020-01-22 10.44.42'] 
# 2020-01-20 13.34.13
# '2020-01-20 23.05.54', '2020-01-20 23.17.39', '2020-01-21 00.18.11',
img_reader = mir.MultiResolutionImageReader()
mask_reader = mir.MultiResolutionImageReader()


def getPositionAndSize(annotation):
    X_min = 10000000
    Y_min = 10000000
    X_max = 0
    Y_max = 0
    for coordinate in annotation.getCoordinates():
        if coordinate.getX() > X_max:
            X_max = coordinate.getX()
        if coordinate.getX() < X_min:
            X_min = coordinate.getX()
        if coordinate.getY() < Y_min:
            Y_min = coordinate.getY()
        if coordinate.getY() > Y_max:
            Y_max = coordinate.getY()
    return int(X_min)-200, int(Y_min)-200, int(X_max - X_min)+400, int(Y_max - Y_min) + 400


def getPatch(pathlist, start, end):
    print('getting start from {} to {}'.format(start, end))
    pathlist = pathlist[start:end] if end != -1 else pathlist[start:]
    for path in pathlist:
        print(path)
        img_name = glob.glob(os.path.join(path, '*.ndpi'))
        xml_name = glob.glob(os.path.join(path, '*.xml'))
        mask_name = glob.glob(os.path.join(path, '*_mask.tiff'))
        assert len(
            img_name) == 1, 'failed to get image {} : no image or multi image'.format(img_name)
        assert len(
            xml_name) == 1, 'failed to get xml label {} : no xml label or multi xml label'.format(xml_name)
        assert len(mask_name) == 1, 'failed to get mask {} : no mask or multi mask'.format(
            mask_name)
        img_name = img_name[0]
        xml_name = xml_name[0]
        mask_name = mask_name[0]

        img = img_reader.open(img_name)
        mask = mask_reader.open(mask_name)
        annotation_list = mir.AnnotationList()
        xml_repository = mir.XmlRepository(annotation_list)
        xml_repository.setSource(xml_name)
        xml_repository.load()

        # annotation_group = annotation_list.getGroup('Annotation Group 0')
        annotations = annotation_list.getAnnotations()
        del xml_repository
        if not os.path.exists(os.path.join(path, 'patch')):
            os.mkdir(os.path.join(path, 'patch'))
        if not os.path.exists(os.path.join(path, 'patch', 'imgs')):
            os.mkdir(os.path.join(path, 'patch', 'imgs'))
        if not os.path.exists(os.path.join(path, 'patch', 'masks')):
            os.mkdir(os.path.join(path, 'patch', 'masks'))
        for idx, annotation in enumerate(annotations):
            x, y, width, height = getPositionAndSize(annotation)
            level_0_width, level_0_height = img.getLevelDimensions(0)
            level_1_width, level_1_height = img.getLevelDimensions(1)
            # x *= level_1_width/level_0_width
            # y *= level_1_height/level_0_height
            width *= level_1_width/level_0_width
            height *= level_1_height/level_0_height
            x, y, width, height = int(x), int(y), int(width), int(height)
            patch_img = img.getUInt16Patch(x, y, width, height, 1)
            patch_img = np.array(patch_img, dtype=np.int8)
            patch_img = Image.fromarray(patch_img, mode='RGB')
            patch_img.save(os.path.join(
                path, 'patch', 'imgs', os.path.splitext(os.path.basename(img_name))[0]+'-{}.png'.format(idx)))
            del patch_img
            patch_mask = mask.getUInt16Patch(x, y, width, height, 1)
            patch_mask = np.array(patch_mask, dtype=np.int8)
            patch_mask = Image.fromarray(patch_mask[:, :, 0], mode='L')
            patch_mask.save(os.path.join(
                path, 'patch', 'masks', os.path.splitext(os.path.basename(img_name))[0]+'-{}.png'.format(idx)))
            del patch_mask


getPatch(pathlist, 0, -1)
# num = int(len(pathlist)/4)
# threads = []
# t1 = threading.Thread(target=getPatch, args=(pathlist, 1, num))
# threads.append(t1)
# t2 = threading.Thread(target=getPatch, args=(pathlist, num, num*2))
# threads.append(t2)
# t3 = threading.Thread(target=getPatch, args=(pathlist, num*2, num*3))
# threads.append(t3)
# t4 = threading.Thread(target=getPatch, args=(pathlist, num*3, -1))
# threads.append(t4)

# for t in threads:
#     t.setDaemon(True)
#     t.start()
# t.join()
# print('ok')
