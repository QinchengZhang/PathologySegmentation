# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-26 10:26:51
LastEditors: TJUZQC
LastEditTime: 2020-10-26 10:27:04
Description: None
'''
from models import *
import torch

x = torch.ones(4, 3, 128, 128)
x = x.to('cuda')
hsunet = HSU_Net(3, 1, 5).to('cuda')
y = hsunet(x)
y.shape