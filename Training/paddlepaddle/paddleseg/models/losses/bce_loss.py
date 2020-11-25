# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-25 13:40:28
LastEditors: TJUZQC
LastEditTime: 2020-11-25 14:03:19
Description: None
'''
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class BCELoss(nn.Layer):
    """
    Implements the binary cross entropy loss function.

    Args:
        weight (Tensor): Manually specify the weight of the binary cross entropy of each batch. 
            If specified, the dimension must be the dimension of a batch of data. The data type is float32, float64.
        reduction (str): Specify the calculation method applied to the output result, the optional
            values ​​are:'none','mean','sum'. The default is'mean', which calculates the mean of BCELoss;
            when set to'sum', the sum of BCELoss is calculated; when set to'none', the original loss is returned.

    """

    def __init__(self, weight=None, reduction='mean'):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in bce_loss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction)

        super(BCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, label):
        out = paddle.nn.functional.binary_cross_entropy(
            input, label, self.weight, self.reduction)
        return out
