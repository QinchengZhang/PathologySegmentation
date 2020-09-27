# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-15 17:20:20
LastEditors: TJUZQC
LastEditTime: 2020-09-16 10:00:56
Description: None
'''
import numpy as np
import torch

def DiceCoeff(input, target):
    assert input.shape == target.shape, f'the size of input and target must be same'
    input = imagetoTensor(input)
    target = imagetoTensor(target)
    eps = 0.0001
    target = target.view(-1)
    input = input.view(-1)
    intersection = (target * input).sum()
    union = target.sum() + input.sum()
    t = (2. * intersection + eps) / (union + eps)
    return t

def PixelAccuracy(input, target):
    assert input.shape == target.shape, f'the size of input and target must be same'
    input = imagetoTensor(input)
    target = imagetoTensor(target)
    tmp = input == target
    acc = torch.sum(tmp).float() / input.nelement()
    # acc = np.sum(input[target == input])/np.sum(input)
    return acc

def imagetoTensor(input):
    return torch.from_numpy(input).floor_divide(255)