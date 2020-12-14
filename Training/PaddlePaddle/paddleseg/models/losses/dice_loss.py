# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-12-03 14:31:32
LastEditors: TJUZQC
LastEditTime: 2020-12-09 15:30:47
Description: None
'''
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class DiceLoss(nn.Layer):
    """
    Implements the dice loss function.
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.EPS = 1e-5

    def forward(self, input, label):
        """
        Forward computation.

        Args:
            input (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, 1), where C is number of classes, and if shape is more than 2D, this
                is (N, 1, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        assert input.shape[1] == 1, f'The channel of logit except 1 but got {input.shape[1]}'
        input = paddle.squeeze(input, 1)
        assert len(label.shape) == len(input.shape), 'The shape of logit and label must be same'
        if len(label.shape) != len(input.shape):
            label = paddle.squeeze(label, 1)

        out = self._compute(input, label.astype(input.dtype), self.EPS)
        label.stop_gradient = True
        return out

    def _compute(self, input, label, eps=1e-5):
        inse = paddle.sum(paddle.multiply(input, label))
        dice_denominator = paddle.sum(input) + paddle.sum(label)
        dice_score = 1 - inse * 2 / (dice_denominator + eps)
        return paddle.mean(dice_score)
        
