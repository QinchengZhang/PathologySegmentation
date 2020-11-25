# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-05 12:29:09
LastEditors: TJUZQC
LastEditTime: 2020-11-20 19:25:22
Description: None
'''
import cv2
import kornia
import kornia.augmentation as K
import torch
from matplotlib import pyplot as plt
from torch import nn


class Norm_MinMax(nn.Module):
    def __init__(self) -> None:
        super(Norm_MinMax, self).__init__()

    def forward(self, input):
        min = float(input.min())
        max = float(input.max())
        return min, max, K.Normalize(min, max - min)(input)


class Denorm_MinMax(nn.Module):
    def __init__(self, min, max) -> None:
        super(Denorm_MinMax, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return torch.tensor(K.Denormalize(min_img, max_img - min_img)(input), dtype=torch.uint8)


if __name__ == '__main__':
    img1 = cv2.imread(
        'G:\TJUZQC\code\python\PathologySegmentation\Training\data\WSI\imgs\\1.png')
    mask1 = cv2.imread(
        'G:\TJUZQC\code\python\PathologySegmentation\Training\data\WSI\masks\\1.png')

    tensor_img1 = kornia.image_to_tensor(img1)
    tensor_mask1 = kornia.image_to_tensor(mask1)
    # print(tensor_img1.max())
    norm = Norm_MinMax()
    min_img, max_img, tensor_img2 = norm(tensor_img1)
    min_mask, max_mask, tensor_mask2 = norm(tensor_mask1)
    aff = K.RandomAffine(45, padding_mode=2, return_transform=True)
    # print(tensor2.max())
    tensor_img2, transform = aff(tensor_img2)
    tensor_mask2, transform = aff((tensor_mask2, transform))
    denorm = Denorm_MinMax(min_img, max_img)
    tensor_img2 = denorm(tensor_img2)
    denorm = Denorm_MinMax(min_mask, max_mask)
    tensor_mask2 = denorm(tensor_mask2)
    # print(tensor_img2.max())
    img2 = kornia.tensor_to_image(tensor_img2)
    mask2 = kornia.tensor_to_image(tensor_mask2)
    plt.subplot(141)
    plt.imshow(img1)
    plt.subplot(142)
    plt.imshow(mask1)
    plt.subplot(143)
    plt.imshow(img2)
    plt.subplot(144)
    plt.imshow(mask2)
    plt.show()
