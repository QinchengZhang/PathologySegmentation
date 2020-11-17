# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-17 12:31:18
LastEditors: TJUZQC
LastEditTime: 2020-11-17 13:39:25
Description: None
'''
import paddle
from .modules import *


class HSU_Net(paddle.nn.Layer):
    def __init__(self,n_channels=3,n_classes=1, split=5, bilinear=True):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        super(HSU_Net,self).__init__()
        
        self.Maxpool = paddle.nn.MaxPool2D(kernel_size=2,stride=2)

        # self.Conv1 = conv_block(ch_in=n_channels,ch_out=64)
        self.Conv1 = HSBottleNeck(n_channels, 64, split)
        # self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv2 = HSBottleNeck(64, 128, split)
        # self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv3 = HSBottleNeck(128, 256, split)
        # self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv4 = HSBottleNeck(256, 512, split)
        # self.Conv5 = conv_block(ch_in=512,ch_out=1024)
        self.Conv5 = HSBottleNeck(512, 1024, split)

        self.Up5 = up_conv(ch_in=1024,ch_out=512, bilinear=bilinear)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up_conv5 = HSBottleNeck(1024, 512, split)

        self.Up4 = up_conv(ch_in=512,ch_out=256, bilinear=bilinear)
        # self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up_conv4 = HSBottleNeck(512, 256, split)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128, bilinear=bilinear)
        # self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up_conv3 = HSBottleNeck(256, 128, split)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64, bilinear=bilinear)
        # self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Up_conv2 = HSBottleNeck(128, 64, split)

        self.Conv_1x1 = paddle.nn.Conv2D(64,n_classes,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = paddle.concat((x4,d5),axis=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = paddle.concat((x3,d4),axis=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = paddle.concat((x2,d3),axis=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = paddle.concat((x1,d2),axis=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

if __name__ == '__main__':
    paddle.set_device('cpu')
    x = paddle.ones((1, 3, 1024, 1024))
    model = HSU_Net(3, 1)
    y = model(x)
    print(y.shape, x.shape)