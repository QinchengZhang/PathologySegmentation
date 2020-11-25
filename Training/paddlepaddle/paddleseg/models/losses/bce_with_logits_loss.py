# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-25 13:40:58
LastEditors: TJUZQC
LastEditTime: 2020-11-25 14:03:04
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
        out = F.binary_cross_entropy_with_logits(
            logit, label, self.weight, self.reduction, self.pos_weight)
        return out
