# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-15 17:20:20
LastEditors: TJUZQC
LastEditTime: 2020-11-20 19:24:15
Description: None
'''
import torch
from torch import from_numpy
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    input, target = from_numpy(input).squeeze(0), from_numpy(target).squeeze(0)
    if input.max() == 255.:
        input = input.div(255.)
    if target.max() == 255.:
        target = target.div(255.)

    return DiceCoeff().forward(input, target)


class PixelAccuracy(Function):
    """Pixel accuracy for individual examples"""

    def forward(self, input, target):
        assert input.shape == target.shape, f'the size of input and target must be same'
        tmp = input == target
        acc = torch.sum(tmp).float() / input.nelement()
        # acc = np.sum(input[target == input])/np.sum(input)
        return acc


def pixel_accuracy(input, target):
    """Pixel accuracy for batches"""
    input, target = from_numpy(input).squeeze(0), from_numpy(target).squeeze(0)
    if input.max() == 255.:
        input = input.div(255.)
    if target.max() == 255.:
        target = target.div(255.)

    return PixelAccuracy().forward(input, target)


def Mask2Tensor(input):
    return from_numpy(input).div(255)
