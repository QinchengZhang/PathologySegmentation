# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-23 12:34:09
LastEditors: TJUZQC
LastEditTime: 2020-10-25 13:31:27
Description: None
'''
""" Parts of the U-Net model """

import torch
import torch.nn as nn

class conv_block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """Upscaling then double conv"""
    def __init__(self,ch_in,ch_out, bilinear=True):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(ch_in // 2, ch_out // 2, kernel_size=2, stride=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
        
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