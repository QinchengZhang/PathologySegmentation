# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-27 13:48:29
LastEditors: TJUZQC
LastEditTime: 2020-10-27 14:11:23
Description: None
'''
import torch
from torch.autograd import Function

class Recall(Function):
    """Recall for individual examples"""

    def forward(self, input, target):
        assert input.shape == target.shape, f'the size of input and target must be same'
        TP = (input == 1) & (target == 1)
        FN = (input == 0) & (target == 1)
        acc = torch.sum(TP).float() / (torch.sum(TP).float() + torch.sum(FN).float())
        # acc = np.sum(input[target == input])/np.sum(input)
        return acc


def recall(input, target):
    """recall for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + Recall().forward(c[0], c[1])

    return s / (i + 1)
