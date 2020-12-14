# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-25 13:40:28
LastEditors: TJUZQC
LastEditTime: 2020-12-03 14:30:57
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
        """
        Forward computation.

        Args:
            input (Tensor): input tensor, the data type is float32, float64. Shape is
                (N, 1), where C is number of classes, and if shape is more than 2D, this
                is (N, 1, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        assert input.shape[1] == 1, f'The channel of input except 1 but fot {input.shape[1]}'
        input = paddle.squeeze(input, 1)
        assert len(label.shape) == len(input.shape), 'The shape of input and label must be same'
        if len(label.shape) != len(input.shape):
            label = paddle.squeeze(label, 1)
        out = F.binary_cross_entropy(
            input, label, self.weight, self.reduction)
        return out
