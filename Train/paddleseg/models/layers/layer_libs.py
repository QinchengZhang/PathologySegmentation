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


def SyncBatchNorm(*args, **kwargs):
    """In cpu environment nn.SyncBatchNorm does not have kernel so use nn.BatchNorm instead"""
    if paddle.get_device() == 'cpu':
        return nn.BatchNorm(*args, **kwargs)
    else:
        return nn.SyncBatchNorm(*args, **kwargs)


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()

        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)

        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class ConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self._conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._batch_norm = SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self._conv(x)
        x = self._batch_norm(x)
        return x


class ConvReLUPool(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = F.pool2d(x, pool_size=2, pool_type="max", pool_stride=2)
        return x


class SeparableConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)
        self.piontwise_conv = ConvBNReLU(
            in_channels, out_channels, kernel_size=1, groups=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class DepthwiseConvBN(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding='same',
                 **kwargs):
        super().__init__()
        self.depthwise_conv = ConvBN(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            **kwargs)

    def forward(self, x):
        x = self.depthwise_conv(x)
        return x


class AuxLayer(nn.Layer):
    """
    The auxiliary layer implementation for auxiliary loss.

    Args:
        in_channels (int): The number of input channels.
        inter_channels (int): The intermediate channels.
        out_channels (int): The number of output channels, and usually it is num_classes.
        dropout_prob (float, optional): The drop rate. Default: 0.1.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 dropout_prob=0.1):
        super().__init__()

        self.conv_bn_relu = ConvBNReLU(
            in_channels=in_channels,
            out_channels=inter_channels,
            kernel_size=3,
            padding=1)

        self.dropout = nn.Dropout(p=dropout_prob)

        self.conv = nn.Conv2D(
            in_channels=inter_channels,
            out_channels=out_channels,
            kernel_size=1)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.dropout(x)
        x = self.conv(x)
        return x

class HSBlock(nn.Layer):
    def __init__(self, w: int, split: int, stride: int = 1) -> None:
        super(HSBlock, self).__init__()
        self.w = w
        self.channels = w*split
        self.split = split
        self.stride = stride
        self.ops_list = nn.LayerList()
        for s in range(1, self.split):
            hc = int((2**(s)-1)/2**(s-1)*self.w)
            self.ops_list.append(nn.Sequential(
                nn.Conv2D(hc, hc, kernel_size=3, padding=1, stride=self.stride),
                # nn.BatchNorm2D(hc),
                # nn.ReLU(),
                ))

    def forward(self, x):
        last_split = None
        channels = x.shape[1]
        assert channels == self.w * \
            self.split, f'input channels({channels}) is not equal to w({self.w})*split({self.split})'
        retfeature = x[:, 0:self.w, :, :]
        for s in range(1, self.split):
            temp = x[:, s*self.w:(s+1)*self.w, :, :] if last_split is None else paddle.concat([last_split, x[:, s*self.w:(s+1)*self.w, :, :]], axis=1)
            temp = self.ops_list[s-1](temp)
            x1, x2 = self._split(temp)
            retfeature = paddle.concat([retfeature, x1], axis=1)
            last_split = x2
        retfeature = paddle.concat([retfeature, last_split], axis=1)
        del last_split
        return retfeature

    def _split(self, x):
        channels = int(x.shape[1]/2)
        return x[:, 0:channels, :, :], x[:, channels:, :, :]

class HSBlockBNReLU(nn.Layer):
    def __init__(self, w: int, split: int, stride: int = 1) -> None:
        super(HSBlockBNReLU, self).__init__()
        self._hsblock = HSBlock(w, split, stride)
        self._batch_norm = SyncBatchNorm(split*w)

    def forward(self, x):
        x = self._hsblock(x)
        x = self._batch_norm(x)
        x = F.relu(x)
        return x


class HSBottleNeck(nn.Layer):
    def __init__(self, in_channels: int, out_channels: int, split: int=5, stride: int = 1) -> None:
        super(HSBottleNeck, self).__init__()
        self.w = max(2**(split-2), 1)
        self.residual_function = nn.Sequential(
            ConvBNReLU(in_channels, self.w*split, kernel_size=1, stride=stride),
            HSBlockBNReLU(self.w, split, stride),
            ConvBN(self.w*split, out_channels,
                             kernel_size=1, stride=stride),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBN(in_channels, out_channels, stride=stride, kernel_size=1)

    def forward(self, x):
        residual = self.residual_function(x)
        shortcut = self.shortcut(x)
        return F.relu(residual + shortcut)

class AttentionBlock(nn.Layer):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            ConvBN(F_g, F_int, kernel_size=1, padding=0)
        )

        self.W_x = nn.Sequential(
            ConvBN(F_l, F_int, kernel_size=1, padding=0)
        )

        self.psi = nn.Sequential(
            ConvBN(F_int, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi