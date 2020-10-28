# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-27 13:46:06
LastEditors: TJUZQC
LastEditTime: 2020-10-27 13:48:02
Description: None
'''
import torch
from torch.autograd import Function

class OverallAccuracy(Function):
    """Overall accuracy for individual examples"""

    def forward(self, input, target):
        assert input.shape == target.shape, f'the size of input and target must be same'
        true = input == target
        false = input != target
        acc = torch.sum(true).float() / torch.sum(false).float()
        return acc


def overall_accuracy(input, target):
    """Overall accuracy for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + OverallAccuracy().forward(c[0], c[1])

    return s / (i + 1)
