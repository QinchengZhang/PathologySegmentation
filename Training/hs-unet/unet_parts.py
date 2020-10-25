# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-23 12:34:09
LastEditors: TJUZQC
LastEditTime: 2020-10-23 14:57:06
Description: None
'''
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class HSBlock(nn.Module):
    def __init__(self, w:int, split:int, stride:int=1) -> None:
        super(HSBlock, self).__init__()
        self.split_list = []
        self.last_split = None
        self.w = w
        self.split = split
        self.stride = stride

    def forward(self, x):
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
                temp = torch.cat([self.last_split, x[:, s*self.w:(s+1)*self.w, :, :]], dim=1)
                ops = nn.Sequential(
                    nn.Conv2d(temp.shape[1], temp.shape[1], kernel_size=3, padding=1, stride=self.stride),
                    nn.BatchNorm2d(temp.shape[1]),
                    nn.ReLU(inplace=True)
                )
                temp = ops(temp)
                x1, x2 = self._split(temp)
                del temp
                self.split_list.append(x1)
                self.last_split = x2
        self.split_list.append(self.last_split)
        return torch.cat(self.split_list, dim=1)

    def _split(self, x):
        channels = int(x.shape[1]/2)
        return x[:, 0:channels, :, :], x[:, channels:, :, :]

class HSBottleNeck(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, split:int, stride:int=1) -> None:
        super(HSBottleNeck, self).__init__()
        self.w = max(2**(split-2), 1)
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, self.w*split, kernel_size=1, stride=stride),
            nn.BatchNorm2d(self.w*split),
            nn.ReLU(inplace=True),
            HSBlock(self.w, split, stride),
            nn.BatchNorm2d(self.w*split),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.w*split, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))