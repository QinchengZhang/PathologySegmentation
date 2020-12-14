# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-12-11 11:52:47
LastEditors: TJUZQC
LastEditTime: 2020-12-14 17:10:30
Description: None
'''
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers


@manager.MODELS.add_component
class HS_Att_UNet(nn.Layer):
    # TODO
    """
    The UNet implementation based on PaddlePaddle.

    The original article refers to
    Olaf Ronneberger, et, al. "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597).

    Args:
        num_classes (int): The unique number of target classes.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        use_deconv (bool, optional): A bool value indicates whether using deconvolution in upsampling.
            If False, use resize_bilinear. Default: False.
        pretrained (str, optional): The path or url of pretrained model for fine tuning. Default: None.
    """

    def __init__(self,
                 num_classes,
                 split=5,
                 align_corners=False,
                 use_deconv=False,
                 pretrained=None):
        super(HS_Att_UNet, self).__init__()

        self.encode = Encoder(split=split)
        self.decode = Decoder(
            align_corners, use_deconv=use_deconv, split=split)
        self.cls = self.conv = nn.Conv2D(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=1)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        logit_list = []
        x, short_cuts = self.encode(x)
        x = self.decode(x, short_cuts)
        logit = self.cls(x)
        logit_list.append(logit)
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)


class Encoder(nn.Layer):
    def __init__(self, split: int = 5):
        super().__init__()

        self.double_conv = nn.Sequential(
            layers.HSBottleNeck(3, 64, split), layers.HSBottleNeck(64, 64, split))
        down_channels = [[64, 128], [128, 256], [256, 512], [512, 512]]
        self.down_sample_list = nn.LayerList([
            self.down_sampling(channel[0], channel[1], split)
            for channel in down_channels
        ])

    def down_sampling(self, in_channels, out_channels, split=5):
        modules = []
        modules.append(nn.MaxPool2D(kernel_size=2, stride=2))
        modules.append(layers.HSBottleNeck(in_channels, out_channels, split))
        modules.append(layers.HSBottleNeck(out_channels, out_channels, split))
        return nn.Sequential(*modules)

    def forward(self, x):
        short_cuts = []
        x = self.double_conv(x)
        for down_sample in self.down_sample_list:
            short_cuts.append(x)
            x = down_sample(x)
        return x, short_cuts


class Decoder(nn.Layer):
    def __init__(self, align_corners, use_deconv=False, split=5):
        super().__init__()

        up_channels = [[512, 256], [256, 128], [128, 64], [64, 64]]
        self.up_sample_list = nn.LayerList([
            UpSampling(channel[0], channel[1],
                       align_corners, use_deconv, split)
            for channel in up_channels
        ])

    def forward(self, x, short_cuts):
        low_f = None
        for i in range(len(short_cuts)):
            x, low_f = self.up_sample_list[i](x, short_cuts[-(i + 1)], low_f)
        return x


class UpSampling(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 align_corners,
                 use_deconv=False,
                 split=5):
        super().__init__()

        self.align_corners = align_corners

        self.use_deconv = use_deconv
        if self.use_deconv:
            self.deconv = nn.Conv2DTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0)
            in_channels = in_channels + out_channels
            self.attention_gate = layers.AttentionBlock(out_channels , in_channels, out_channels//2)

        else:
            self.attention_gate = layers.AttentionBlock(in_channels , in_channels, out_channels//2)
            self.conv = layers.ConvBN(in_channels*2, in_channels, kernel_size=1)
            in_channels *= 2
            
        self.double_conv = nn.Sequential(
            # layers.ConvBNReLU(in_channels, out_channels, 3),
            # layers.ConvBNReLU(out_channels, out_channels, 3))
            layers.HSBottleNeck(in_channels, out_channels, split),
            layers.HSBottleNeck(out_channels, out_channels, split))

    def forward(self, x, short_cut, low_f=None):
        if self.use_deconv:
            x = self.deconv(x)
            if low_f is not None:
                low_f = self.deconv(low_f)
        else:
            x = F.interpolate(
                x,
                short_cut.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            if low_f is not None:
                low_f = F.interpolate(
                        low_f,
                        short_cut.shape[2:],
                        mode='bilinear',
                        align_corners=self.align_corners)
                low_f = self.conv(low_f)
                
        # print(x.shape, short_cut.shape, low_f.shape if low_f is not None else '')
        f = self.attention_gate(
            x, short_cut)*low_f if low_f is not None else self.attention_gate(x, short_cut)
        x = paddle.concat([x, f], axis=1)
        x = self.double_conv(x)
        # print(f.shape)
        return x, f
