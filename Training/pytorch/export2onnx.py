# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-11 14:53:39
LastEditors: TJUZQC
LastEditTime: 2020-11-11 15:41:12
Description: None
'''
import torch
from torch import onnx
from models import *
from PIL import Image
import torchvision

img = torchvision.io.read_image('G:/TJUZQC/code/python/PathologySegmentation/Training/data/WSI/imgs/1.png')
size = (1, *img.shape)
dummy_input = img.reshape(size).to(device='cuda', dtype=torch.float32)
print(dummy_input.dtype)
model = HSU_Net(n_channels=3,n_classes=1, split=5, bilinear=True).cuda()
model.load_state_dict(torch.load('G:\TJUZQC\code\python\PathologySegmentation\Training\checkpoints_WSI\hsunet_best_in_20_10_26.pth', map_location='cuda'))

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

torch.onnx.export(model, dummy_input, "hsunet.onnx", verbose=True, input_names=input_names, output_names=output_names)