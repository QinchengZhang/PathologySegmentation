# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-08 14:05:31
LastEditors: TJUZQC
LastEditTime: 2020-09-27 17:26:24
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



# 从xml标注中得到一个Annotation的边界信息，所得边界比真实边界大200px
def getPositionAndSize(annotation):
    X_min = None
    Y_min = None
    X_max = None
    Y_max = None
    for coordinate in annotation.getCoordinates():
        if coordinate.getX() > X_max or X_max is None:
            X_max = coordinate.getX()
        if coordinate.getX() < X_min or X_min is None:
            X_min = coordinate.getX()
        if coordinate.getY() < Y_min or Y_min is None:
            Y_min = coordinate.getY()
        if coordinate.getY() > Y_max or Y_max is None:
            Y_max = coordinate.getY()
    return int(X_min)-200, int(Y_min)-200, int(X_max - X_min)+400, int(Y_max - Y_min) + 400

# 切patch
def __getPatch(pathlist, start, end):
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

        img_reader = mir.MultiResolutionImageReader()
        mask_reader = mir.MultiResolutionImageReader()

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

def getPatch(pathlist, num_works):
    num = int(len(pathlist)/num_works) if int(len(pathlist)/num_works) > 1 else 1
    threads = []
    for work_idx in range(num_works):
        
        threads.append(threading.Thread(target=__getPatch, args=(pathlist, work_idx*num, (work_idx+1)*num)))
    for t in threads:
        t.setDaemon(True)
        t.start()
    t.join()
    print('All threads is done!')  


if __name__ == '__main__':
    path = 'G:\TJUZQC\DataSet\Beijing-small_cell_lung_cancer-pathology\\2020-01-20 10.39.42'
    pathlist = glob.glob(path)
    getPatch(pathlist, 6)
