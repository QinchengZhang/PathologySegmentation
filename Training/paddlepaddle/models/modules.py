# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-17 11:58:53
LastEditors: TJUZQC
LastEditTime: 2020-11-18 13:14:19
Description: None
'''
import paddle


class conv_block(paddle.nn.Layer):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(ch_in, ch_out, kernel_size=3,
                             stride=1, padding=1),
            paddle.nn.BatchNorm2D(ch_out),
            paddle.nn.ReLU6(),
            paddle.nn.Conv2D(ch_out, ch_out, kernel_size=3,
                             stride=1, padding=1),
            paddle.nn.BatchNorm2D(ch_out),
            paddle.nn.ReLU6(),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(paddle.nn.Layer):
    """Upscaling then double conv"""

    def __init__(self, ch_in, ch_out, bilinear=True):
        super(up_conv, self).__init__()
        self.up = paddle.nn.Sequential(
            paddle.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else paddle.nn.ConvTranspose2d(
                ch_in // 2, ch_out // 2, kernel_size=2, stride=2),
            paddle.nn.Conv2D(ch_in, ch_out, kernel_size=3,
                             stride=1, padding=1),
            paddle.nn.BatchNorm2D(ch_out),
            paddle.nn.ReLU6()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class HSBlock(paddle.nn.Layer):
    def __init__(self, w: int, split: int, stride: int = 1) -> None:
        super(HSBlock, self).__init__()
        self.w = w
        self.channels = w*split
        self.split = split
        self.stride = stride
        self.ops_list = []
        for s in range(1, self.split):
            hc = int((2**(s)-1)/2**(s-1)*self.w)
            self.ops_list.append(paddle.nn.Sequential(
                paddle.nn.Conv2D(hc, hc, kernel_size=3, padding=1, stride=self.stride),
                paddle.nn.BatchNorm2D(hc),
                paddle.nn.ReLU(),
                ))

    def forward(self, x):
        split_list = []
        last_split = None
        split_list.append(x[:, 0:self.w, :, :])
        for s in range(1, self.split):
            if last_split is None:
                temp = self.ops_list[s-1](x[:, s*self.w:(s+1)*self.w, :, :])
                x1, x2 = self._split(temp)
                split_list.append(x1)
                last_split = x2
            else:
                temp = paddle.concat(
                    [last_split, x[:, s*self.w:(s+1)*self.w, :, :]], axis=1)
                temp = self.ops_list[s-1](temp)
                x1, x2 = self._split(temp)
                split_list.append(x1)
                last_split = x2
        split_list.append(last_split)
        del last_split
        return paddle.concat(split_list, axis=1)

    def _split(self, x):
        channels = int(x.shape[1]/2)
        return x[:, 0:channels, :, :], x[:, channels:, :, :]


class HSBottleNeck(paddle.nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, split: int, stride: int = 1) -> None:
        super(HSBottleNeck, self).__init__()
        self.w = max(2**(split-2), 1)
        self.residual_function = paddle.nn.Sequential(
            paddle.nn.Conv2D(in_channels, self.w*split,
                             kernel_size=1, stride=stride),
            paddle.nn.BatchNorm2D(self.w*split),
            paddle.nn.ReLU(),
            HSBlock(self.w, split, stride),
            paddle.nn.BatchNorm2D(self.w*split),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(self.w*split, out_channels,
                             kernel_size=1, stride=stride),
            paddle.nn.BatchNorm2D(out_channels)
        )
        self.shortcut = paddle.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = paddle.nn.Sequential(
                paddle.nn.Conv2D(in_channels, out_channels,
                                 stride=stride, kernel_size=1),
                paddle.nn.BatchNorm2D(out_channels)
            )

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        return paddle.nn.ReLU()(residual + shortcut)
