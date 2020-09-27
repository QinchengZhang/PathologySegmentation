# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-02 09:44:35
LastEditors: TJUZQC
LastEditTime: 2020-09-02 14:58:05
Description: None
'''
import openslide
import os
from PIL import Image
import glob
from matplotlib import pyplot as plt
import shutil
from libtiff import TIFF

print(os.getcwd())
data_path = 'G:\\TJUZQC\\DataSet\\Beijing-no_small_cell_lung_cancer-pathology'
for idx, name in enumerate(glob.glob(data_path + '\\*.ndpi')):
    name, ext_name = os.path.splitext(name)
    try:
        os.path.getsize(name + '.xml')
        print(name + ' found')
        os.mkdir(os.path.join(data_path, 'data-{}'.format(idx)))
        new_img_path = os.path.join(
            data_path, 'data-{}'.format(idx), '{}.ndpi'.format(idx))
        new_xml_path = os.path.join(
            data_path, 'data-{}'.format(idx), '{}.xml'.format(idx))
        new_mask_path = os.path.join(
            data_path, 'data-{}'.format(idx), '{}.tif'.format(idx))
        shutil.move(name + '.ndpi', new_img_path)
        shutil.move(name + '.xml', new_xml_path)
        shutil.move(name + '.tif', new_mask_path)
        print('OpenSlide open:' + new_img_path)
        img = openslide.OpenSlide(new_img_path)
        mask = TIFF.open(new_mask_path)
        mask = mask.read_image()
        mask[mask >= 1] = 255
        width, height = img.dimensions
        min_width, min_height = img.level_dimensions[7]
        for j in range(0, height, min_height):
            for idx2, i in enumerate(range(0, width, min_width)):
                patch = img.read_region((i, j), 0, (min_width, min_height))
                patch = patch.resize((512, 512))

                patch_mask = mask[j:j+min_height, i:i+min_width]
                if patch_mask.max() >= 1 or idx2 % 10 == 0:
                    patch_mask = Image.fromarray(patch_mask)
                    patch_mask = patch_mask.resize((512, 512))
                    patch.save(os.path.join(
                        data_path, 'data-{}'.format(idx), '{}-({},{}).png'.format(idx, i, j)))
                    patch_mask.save(os.path.join(
                        data_path, 'data-{}'.format(idx), '{}-({},{}).tif'.format(idx, i, j)))
    except FileNotFoundError as e:
        print(name, 'not found')
# name = glob.glob(data_path + '\\*.ndpi')[1]
# img = openslide.OpenSlide(name)
# img = img.read_region((0,0), 0, dimensions[-1])
# plt.imshow(img)
# plt.show()
