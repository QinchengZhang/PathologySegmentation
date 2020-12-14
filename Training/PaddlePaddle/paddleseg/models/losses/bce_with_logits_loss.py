# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-25 13:40:58
LastEditors: TJUZQC
LastEditTime: 2020-12-09 14:04:27
Description: None
'''
import paddle
from paddle import nn
import paddle.nn.functional as F

from paddleseg.cvlibs import manager


@manager.LOSSES.add_component
class BCEWithLogitsLoss(nn.Layer):
    """
    Implements the binary cross entropy loss with logits function.

    Args:
        weight (Tensor): Manually specify the weight of the binary cross entropy of each batch. 
            If specified, the dimension must be the dimension of a batch of data. The data type is float32, float64.
        reduction (str): Specify the calculation method applied to the output result, the optional
            values ​​are:'none','mean','sum'. The default is'mean', which calculates the mean of BCELoss;
            when set to'sum', the sum of BCELoss is calculated; when set to'none', the original loss is returned.
        pos_weight (Tensor): Manually specify the weight of the positive class, which must be a vector
            of the same length as the number of classes. The data type is float32, float64.

    """

    def __init__(self,
                 weight=None,
                 reduction='mean',
                 pos_weight=None):
        if reduction not in ['sum', 'mean', 'none']:
            raise ValueError(
                "The value of 'reduction' in BCEWithLogitsLoss should be 'sum', 'mean' or 'none', but "
                "received %s, which is not allowed." % reduction)

        super(BCEWithLogitsLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, logit, label):
        """
        Forward computation.

        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, 1), where C is number of classes, and if shape is more than 2D, this
                is (N, 1, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        assert logit.shape[1] == 1, f'The channel of logit except 1 but fot {logit.shape[1]}'
        logit = paddle.squeeze(logit, 1)
        assert len(label.shape) == len(logit.shape), 'The shape of logit and label must be same'
        if len(label.shape) != len(logit.shape):
            label = paddle.squeeze(label, 1)

        out = F.binary_cross_entropy_with_logits(
            logit, label.astype(logit.dtype), self.weight, self.reduction, self.pos_weight)
        label.stop_gradient = True
        return out
