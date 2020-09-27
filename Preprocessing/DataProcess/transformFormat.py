# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-08 17:51:09
LastEditors: TJUZQC
LastEditTime: 2020-09-17 16:17:27
Description: None
'''
import multiresolutionimageinterface as mir
import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
import threading


reader = mir.MultiResolutionImageReader()


def makeMask(pathlist, start, end):
    print('processing pathlist from {} to {}'.format(start, end))
    for path in pathlist[start:end]:
        img_name = glob.glob(os.path.join(path, '*.ndpi'))
        mask_name = glob.glob(os.path.join(path, '*.xml'))
        print(img_name)
        assert len(img_name) == 1, f'no image or multi image'
        assert len(mask_name) == 1, f'no mask or multi mask'
        img_name = img_name[0]
        mask_name = mask_name[0]
        img = reader.open(img_name)
        annotation_list = mir.AnnotationList()
        xml_repository = mir.XmlRepository(annotation_list)
        xml_repository.setSource(mask_name)
        xml_repository.load()

        annotation_mask = mir.AnnotationToMask()   # 由标注转换成mask

        label_map = {'Annotation Group 0': 255}
        conversion_order = ['Annotation Group 0']
        output_path = os.path.join(path, os.path.basename(
            img_name).replace('.ndpi', '_mask.tiff'))
        annotation_mask.convert(annotation_list, output_path, img.getDimensions(
        ), img.getSpacing(), label_map, conversion_order)

pathlist = glob.glob('2020-01-22 10.44.42')
# num = int(len(pathlist)/4)
makeMask(pathlist, 0, -1)
# threads = []
# t1 = threading.Thread(target=makeMask, args=(pathlist, 0, num))
# threads.append(t1)
# t2 = threading.Thread(target=makeMask, args=(pathlist, num, num*2))
# threads.append(t2)
# t3 = threading.Thread(target=makeMask, args=(pathlist, num*2, num*3))
# threads.append(t3)
# t4 = threading.Thread(target=makeMask, args=(pathlist, num*3, -1))
# threads.append(t4)


# for t in threads:
#     t.setDaemon(True)
#     t.start()
# t.join()
# print('ok')

