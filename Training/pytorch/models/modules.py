# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-25 13:08:10
LastEditors: TJUZQC
LastEditTime: 2020-11-19 13:55:00
Description: None
'''
import torch
import torch.nn as nn


class conv_block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, ch_in, ch_out, bilinear=True):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(
                ch_in // 2, ch_out // 2, kernel_size=2, stride=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv(x)
        for i in range(self.t):
            x1 = self.conv(x+x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(
            ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class HSBlock_NEW(nn.Module):
    def __init__(self, w: int, split: int, stride: int = 1) -> None:
        super(HSBlock_NEW, self).__init__()
        self.w = w
        self.split = split
        self.stride = stride

    def forward(self, x):
        split_list = []
        last_split = None
        channels = x.shape[1]
        assert channels == self.w * \
            self.split, f'input channels({channels}) is not equal to w({self.w})*split({self.split})'
        retfeature = x[:, 0:self.w, :, :]
        # split_list.append(x[:, 0:self.w, :, :])
        for s in range(1, self.split):
            hc = int((2**(s)-1)/2**(s-1)*self.w)
            ops = nn.Sequential(
                nn.Conv2d(hc, hc, kernel_size=3,
                          padding=1, stride=self.stride),
                nn.BatchNorm2d(hc),
                nn.ReLU(inplace=True)
            )
            if x.is_cuda:
                ops = ops.to('cuda')
            temp = torch.cat([last_split, x[:, s*self.w:(s+1)*self.w, :, :]],
                             dim=1) if last_split is not None else x[:, s*self.w:(s+1)*self.w, :, :]
            temp = ops(temp)
            x1, x2 = self._split(temp)
            del temp
            retfeature = torch.cat([retfeature, x1], dim=1)
            # split_list.append(x1)
            last_split = x2
        retfeature = torch.cat([retfeature, last_split], dim=1)
        # split_list.append(last_split)
        return retfeature
        # return torch.cat(split_list, dim=1)

    def _split(self, x):
        channels = int(x.shape[1]/2)
        return x[:, 0:channels, :, :], x[:, channels:, :, :]


class HSBlock(nn.Module):
    def __init__(self, w: int, split: int, stride: int = 1) -> None:
        super(HSBlock, self).__init__()
        self.w = w
        self.split = split
        self.channel = w*split
        self.stride = stride
        self.conv_list = []
        for s in range(1, split):
            hc = int((2**(s)-1)/2**(s-1)*self.w)
            self.conv_list.append(nn.Sequential(
                nn.Conv2d(hc, hc, kernel_size=3,
                          padding=1, stride=self.stride),
                nn.BatchNorm2d(hc),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        split_list = []
        last_split = None
        channels = x.shape[1]
        assert channels == self.w * \
            self.split, f'input channels({channels}) is not equal to w({self.w})*split({self.split})'
        retfeature = x[:, 0:self.w, :, :]
        # split_list.append(x[:, 0:self.w, :, :])
        for s in range(1, self.split):
            # if x.is_cuda:
            #     self. = ops.to('cuda')
            temp = torch.cat([last_split, x[:, s*self.w:(s+1)*self.w, :, :]],
                             dim=1) if last_split is not None else x[:, s*self.w:(s+1)*self.w, :, :]
            temp = self.conv_list[s-1](temp)
            x1, x2 = self._split(temp)
            del temp
            retfeature = torch.cat([retfeature, x1], dim=1)
            # split_list.append(x1)
            last_split = x2
        retfeature = torch.cat([retfeature, last_split], dim=1)
        # split_list.append(last_split)
        return retfeature
        # return torch.cat(split_list, dim=1)

    def _split(self, x):
        channels = int(x.shape[1]/2)
        return x[:, 0:channels, :, :], x[:, channels:, :, :]

class HSBlock5(nn.Module):
    def __init__(self, w: int, stride: int = 1) -> None:
        super(HSBlock5, self).__init__()
        self.w = w
        self.channel = w*5
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(w, w, kernel_size=3, padding=1, stride=self.stride),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(1.5*w), int(1.5*w), kernel_size=3, padding=1, stride=self.stride),
            nn.BatchNorm2d(int(1.5*w)),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(1.75*w), int(1.75*w), kernel_size=3, padding=1, stride=self.stride),
            nn.BatchNorm2d(int(1.75*w)),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(int(1.875*w), int(1.875*w), kernel_size=3, padding=1, stride=self.stride),
            nn.BatchNorm2d(int(1.875*w)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        retfeature = x[:, 0:self.w, :, :]
        temp = self.conv1(x[:, self.w:self.w*2, :, :])
        x1, x2 = self._split(temp)
        retfeature = torch.cat([retfeature, x1], dim=1)
        temp = self.conv2(torch.cat([x2, x[:, self.w*2:self.w*3, :, :]], dim=1))
        x1, x2 = self._split(temp)
        retfeature = torch.cat([retfeature, x1], dim=1)
        temp = self.conv3(torch.cat([x2, x[:, self.w*3:self.w*4, :, :]], dim=1))
        x1, x2 = self._split(temp)
        retfeature = torch.cat([retfeature, x1], dim=1)
        temp = self.conv4(torch.cat([x2, x[:, self.w*4:self.w*5, :, :]], dim=1))
        x1, x2 = self._split(temp)
        retfeature = torch.cat([retfeature, x1], dim=1)
        return retfeature
        # return torch.cat(split_list, dim=1)

    def _split(self, x):
        channels = int(x.shape[1]/2)
        return x[:, 0:channels, :, :], x[:, channels:, :, :]


class HSBottleNeck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, split: int=5, stride: int = 1) -> None:
        super(HSBottleNeck, self).__init__()
        self.w = max(2**(split-2), 1)
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, self.w*split, kernel_size=1, stride=stride),
            nn.BatchNorm2d(self.w*split),
            nn.ReLU(inplace=True),
            HSBlock5(self.w, stride),
            nn.BatchNorm2d(self.w*split),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.w*split, out_channels,
                      kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=stride,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        return nn.ReLU(inplace=True)(residual + shortcut)


class HSBottleNeck_NEW(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, split: int, stride: int = 1) -> None:
        super(HSBottleNeck_NEW, self).__init__()
        self.w = max(2**(split-2), 1)
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, self.w*split, kernel_size=1, stride=stride),
            nn.BatchNorm2d(self.w*split),
            nn.ReLU(inplace=True),
            HSBlock_NEW(self.w, split, stride),
            nn.BatchNorm2d(self.w*split),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.w*split, out_channels,
                      kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, stride=stride,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        return nn.ReLU(inplace=True)(residual + shortcut)
