# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-03 09:54:56
LastEditors: TJUZQC
LastEditTime: 2020-09-03 12:29:15
Description: None
'''
import multiresolutionimageinterface as mir
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
reader = mir.MultiResolutionImageReader()
path = 'G:\TJUZQC\DataSet\Beijing-no_small_cell_lung_cancer-pathology'
name = '1467171,H2 - 2020-05-18 16.23.51'
if not os.path.exists(os.path.join(path, name)):
    os.mkdir(os.path.join(path, name))

mr_image = reader.open(os.path.join(path, name+'.ndpi'))
try:
    os.path.getsize(os.path.join(path, name, name + '.tiff'))
    print('mask exists')
except FileNotFoundError:
    annotation_list = mir.AnnotationList()
    
    xml_repository = mir.XmlRepository(annotation_list)
    xml_repository.setSource(path + name + '.xml')
    
    xml_repository.load()
    annotations = annotation_list.getAnnotations()
    print(annotation_list.getAnnotations())

#     annotation_mask = mir.AnnotationToMask()
#     # _0 等就是你自己设置的group，ASAP默认group为Annotation Group *
#     label_map = {'metastases': 255, 'None': 255, '_2': 0}
#     conversion_order = ['metastases', 'None', '_2']
#     output_path = os.path.join(path, name, name + '.tiff')
#     annotation_mask.convert(annotation_list, output_path, mr_image.getDimensions(
#     ), mr_image.getSpacing(), label_map, conversion_order)
#     del annotation_list, annotation_mask, xml_repository
#     print('mask generated')
# mr_mask = reader.open(os.path.join(path, name, name + '.tiff'))
# width, height = mr_image.getDimensions()
# min_width, min_height = mr_image.getLevelDimensions(
#     mr_image.getNumberOfLevels()-1)

# with tqdm(range(0, height, min_height)) as pbar:
#     for idx1, j in enumerate(pbar):
#         pbar.set_description("Processing {}".format(idx1))
#         for idx2, i in enumerate(range(0, width, min_width)):
#             patch = mr_image.getUInt16Patch(i, j, min_width, min_height, 0)
#             patch_mask = mr_mask.getUInt16Patch(i, j, min_width, min_height, 0)
#             patch = np.array(patch, dtype='uint8')
#             patch_mask = np.array(patch_mask, dtype='uint8')
#             print(patch_mask.max())
#             patch_mask[patch_mask < 1] = 0
#             patch_mask[patch_mask >= 1] = 255
#             if patch_mask.max() >= 1:
#                 # patch.resize(512,512)
#                 # patch_mask.resize(512,512)
#                 patch_img = Image.fromarray(patch)
#                 patch_img = patch_img.resize((512, 512))
#                 patch_mask = Image.fromarray(patch_mask[:, :, 0], mode='L')
#                 patch_mask = patch_mask.resize((512, 512))
#                 patch_img.save(os.path.join(
#                     path, name, '{}-{}.png'.format(idx1, idx2)))
#                 patch_mask.save(os.path.join(
#                     path, name, 'mask-{}-{}.png'.format(idx1, idx2)))
