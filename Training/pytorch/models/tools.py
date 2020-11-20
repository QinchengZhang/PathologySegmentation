# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-25 13:47:32
LastEditors: TJUZQC
LastEditTime: 2020-11-20 13:47:16
Description: None
'''
from torch.nn import init
from . import *

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def ChooseModel(model_name:str):
    switch = {'unet': U_Net,
              'r2unet': R2U_Net,
              'attunet': AttU_Net,
              'r2attunet': R2AttU_Net,
              'hsunet': HSU_Net,
              'fcn8s': FCN8s,
              'fcn16s': FCN16s,
              'fcn32s': FCN32s,
              'fcn1s': FCN1s,
              }
    return switch.get(model_name, None)
