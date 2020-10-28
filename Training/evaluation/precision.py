# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-27 13:54:35
LastEditors: TJUZQC
LastEditTime: 2020-10-27 14:10:45
Description: None
'''
import torch
from torch.autograd import Function

class Precision(Function):
    """Precision for individual examples"""

    def forward(self, input, target):
        assert input.shape == target.shape, f'the size of input and target must be same'
        TP = (input == 1) & (target == 1)
        FP = (input == 1) & (target == 0)
        acc = torch.sum(TP).float() / (torch.sum(TP).float() + torch.sum(FP).float())
        # acc = np.sum(input[target == input])/np.sum(input)
        return acc


def precision(input, target):
    """Precision for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + Precision().forward(c[0], c[1])

    return s / (i + 1)
