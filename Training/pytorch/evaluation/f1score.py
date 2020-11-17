# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-27 13:52:22
LastEditors: TJUZQC
LastEditTime: 2020-10-27 14:12:09
Description: None
'''
import torch
from torch.autograd import Function
from .precision import Precision
from .recall import Recall


class F1Score(Function):
    """F1Score for individual examples"""

    def forward(self, input, target):
        assert input.shape == target.shape, f'the size of input and target must be same'
        precision = Precision().forward(input, target)
        recall = Recall().forward(input, target)
        acc = 2*(precision*recall/(precision + recall))
        return acc


def f1score(input, target):
    """F1Score accuracy for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + F1Score().forward(c[0], c[1])

    return s / (i + 1)
