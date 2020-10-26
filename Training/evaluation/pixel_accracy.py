# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-26 13:37:34
LastEditors: TJUZQC
LastEditTime: 2020-10-26 13:40:33
Description: None
'''
import torch
from torch.autograd import Function


class PixelAccuracy(Function):
    """Pixel accuracy for individual examples"""

    def forward(self, input, target):
        assert input.shape == target.shape, f'the size of input and target must be same'
        tmp = input == target
        acc = sum(tmp).float() / input.nelement()
        # acc = np.sum(input[target == input])/np.sum(input)
        return acc


def pixel_accuracy(input, target):
    """Pixel accuracy for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + PixelAccuracy().forward(c[0], c[1])

    return s / (i + 1)
