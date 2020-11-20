# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-11 14:53:39
LastEditors: TJUZQC
LastEditTime: 2020-11-20 13:38:13
Description: None
'''
import torch
from torch import onnx
from models import *
from PIL import Image
import torchvision

dummy_input = torch.ones((1,3,512,512))
print(dummy_input.shape)
model = HSBottleNeck(3, 16)
# model.load_state_dict(torch.load('G:/TJUZQC/code/python/PathologySegmentation/Training/pytorch/checkpoints_WSI/hsunet_best_in_20_11_20.pth.pth', map_location='cpu'))

# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "hsbottleneck.onnx", opset_version=11, verbose=True, input_names=input_names, output_names=output_names) #