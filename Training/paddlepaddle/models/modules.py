# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-17 11:58:53
LastEditors: TJUZQC
LastEditTime: 2020-11-17 12:39:31
Description: None
'''
import paddle

class conv_block(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            paddle.nn.BatchNorm2D(ch_out),
            paddle.nn.ReLU6(),
            paddle.nn.Conv2D(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            paddle.nn.BatchNorm2D(ch_out),
            paddle.nn.ReLU6(),
        )

    def forward(self, x):
        return self.conv(x)

class up_conv(paddle.nn.Layer):
    """Upscaling then double conv"""
    def __init__(self,ch_in,ch_out, bilinear=True):
        super(up_conv,self).__init__()
        self.up = paddle.nn.Sequential(
            paddle.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else paddle.nn.ConvTranspose2d(ch_in // 2, ch_out // 2, kernel_size=2, stride=2),
            paddle.nn.Conv2D(ch_in,ch_out,kernel_size=3,stride=1,padding=1),
		    paddle.nn.BatchNorm2D(ch_out),
			paddle.nn.ReLU6()
        )

    def forward(self,x):
        x = self.up(x)
        return x

class HSBlock(paddle.nn.Layer):
    def __init__(self, w:int, split:int, stride:int=1) -> None:
        super(HSBlock, self).__init__()
        self.split_list = []
        self.last_split = None
        self.w = w
        self.split = split
        self.stride = stride

    def forward(self, x):
        self.split_list = []
        self.last_split = None
        channels = x.shape[1]
        assert channels == self.w*self.split, f'input channels({channels}) is not equal to w({self.w})*split({self.split})'
        self.split_list.append(x[:, 0:self.w, :, :])
        for s in range(1, self.split):
            if self.last_split is None:
                x1, x2 = self._split(x[:, s*self.w:(s+1)*self.w, :, :])
                self.split_list.append(x1)
                self.last_split = x2
            else:
                temp = paddle.concat([self.last_split, x[:, s*self.w:(s+1)*self.w, :, :]], axis=1)
                ops = paddle.nn.Sequential(
                    paddle.nn.Conv2D(temp.shape[1], temp.shape[1], kernel_size=3, padding=1, stride=self.stride),
                    paddle.nn.BatchNorm2D(temp.shape[1]),
                    paddle.nn.ReLU()
                )

                temp = ops(temp)
                x1, x2 = self._split(temp)
                del temp
                self.split_list.append(x1)
                self.last_split = x2
        self.split_list.append(self.last_split)
        return paddle.concat(self.split_list, axis=1)

    def _split(self, x):
        channels = int(x.shape[1]/2)
        return x[:, 0:channels, :, :], x[:, channels:, :, :]

class HSBottleNeck(paddle.nn.Layer):
    def __init__(self, in_channels:int, out_channels:int, split:int, stride:int=1) -> None:
        super(HSBottleNeck, self).__init__()
        self.w = max(2**(split-2), 1)
        self.residual_function = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels, self.w*split, kernel_size=1, stride=stride),
            paddle.nn.BatchNorm2D(self.w*split),
            paddle.nn.ReLU(),
            HSBlock(self.w, split, stride),
            paddle.nn.BatchNorm2D(self.w*split),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(self.w*split, out_channels, kernel_size=1, stride=stride),
            paddle.nn.BatchNorm2D(out_channels)
        )
        self.shortcut = paddle.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = paddle.nn.Sequential(
                paddle.nn.Conv2D(in_channels, out_channels, stride=stride, kernel_size=1),
                paddle.nn.BatchNorm2D(out_channels)
            )

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        return paddle.nn.ReLU()(residual + shortcut)