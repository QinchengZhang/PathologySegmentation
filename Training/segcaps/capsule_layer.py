# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-23 16:17:18
LastEditors: TJUZQC
LastEditTime: 2020-09-24 11:36:33
Description: None
'''
import torch
from torch import nn
from torch.autograd import Variable 
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# 胶囊化层
class Capsulation2D(nn.Module):
    def __init__(self):
        super(Capsulation2D, self).__init__()
        
    def forward(self, x):
        # x.shape = (batch_size, channels, height, weight)
        assert len(x.size()) == 4, f"while x.size() = {x.size()}"
        # output.shape = (batch_size, out_channels, out_dim_capsule, height, weight)
        return torch.unsqueeze(x, dim=2)

# 反胶囊化层
class DeCapsulation2D(nn.Module):
    def __init__(self):
        super(DeCapsulation2D, self).__init__()
        
    def forward(self, x):
        # x.shape = (batch_size, channels, dim_capsule, height, weight)
        assert len(x.size()) == 5, f"while x.size() = {x.size()}"
        # output.shape = (batch_size, out_channels, height, weight)
        return x.view(x.size(0), -1, *x.size()[-2:])

# 平化层
class CapFlatten(nn.Module):
    def __init__(self):
        super(CapFlatten, self).__init__()
        
    def forward(self, x):
        # x.shape = (batch_size, channels, dim_capsule, height, weight)
        assert len(x.size()) == 5, f"while x.size() = {x.size()}"
        # output.shape = (batch_size, channels * height * weight, dim_capsule) which is (batch_size, num_capsules, dim_capsule)
        bs, c, dc, h, w = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x.view(bs, c * h * w, dc)

# 反平化层
class DeCapFlatten(nn.Module):
    def __init__(self, channels, height, weight):
        super(DeCapFlatten, self).__init__()
        self.channels = channels
        self.height = height
        self.weight = weight
        
    def forward(self, x):
        # x.shape = (batch_size, channels * height * weight, dim_capsule) which is (batch_size, num_capsules, dim_capsule)
        assert len(x.size()) == 3, f"while x.size() = {x.size()}"
        
        # output.shape = (batch_size, channels, dim_capsule, height, weight)
        bs, num_capsule, dim_capsule = x.size()
        assert self.channels * self.height * self.weight == num_capsule, f"while {self.channels} * {self.height} * {self.weight} != {num_capsule}"
        x = x.view(bs, self.channels, self.height, self.weight, dim_capsule)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        return x

# 标量化层
class CapScalarization(nn.Module):
    def __init__(self, dim=-1):
        """
        :param dim: vector norm will be applied to this dimension
        """
        super(CapScalarization, self).__init__()
        self.dim = dim
        
    def forward(self, x):
        # x.shape = (batch_size, num_capsules, dim_capsule)
        assert len(x.size()) == 3, f"while x.size() = {x.size()}"
        # output.shape = (batch_size, num_capsules)
        return torch.norm(x, p=2, dim=self.dim)

# 胶囊2D卷积层
class CapConv2d(nn.Module):
    def __init__(self, in_channels, in_dim, out_channels, out_dim, kernel_size, **kargs):
        super(CapConv2dV1, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels*in_dim, out_dim*out_channels, kernel_size=kernel_size, **kargs)
        nn.init.xavier_normal_(self.conv1.weight.data, gain=1)
        
        self.bn1 = nn.BatchNorm2d(out_dim*out_channels)
        nn.init.normal_(self.bn1.weight.data, mean=0.0, std=1.0)
        
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.out_channels = out_channels
        self.out_dim = out_dim
        
    def forward(self, x):
        # x.shape = (batch_size, channels, dim_capsule, height, weight)
        assert len(x.size()) == 5, f"while x.size() = {x.size()}"
        x = self.conv1(x.view(x.size(0), self.in_channels*self.in_dim, *x.size()[-2:]))
        x = F.relu(self.bn1(x))
        # output.shape = (batch_size, out_channels, out_dim_capsule, out_height, out_weight)
        return x.view(x.size(0), self.out_channels, self.out_dim, *x.size()[-2:])  # bs = x.size(0) | h, w = x.size()[-2:]

# 数字胶囊（路由输出层）
class CapsuleLayer(nn.Module):
    def __init__(self, in_num_capsule, in_dim_capsule, out_num_capsule, out_dim_capsule, routings=3):
        super(CapsuleLayer, self).__init__()
        self.in_num_capsule = in_num_capsule
        self.in_dim_capsule = in_dim_capsule
        self.out_num_capsule = out_num_capsule
        self.out_dim_capsule = out_dim_capsule
        self.routings = routings
        
        self.squash = CapTool().squash
        
        self.W = torch.nn.Parameter(torch.Tensor(out_num_capsule, in_num_capsule, out_dim_capsule, in_dim_capsule))
        torch.nn.init.xavier_uniform_(self.W, gain=1)
        
    def forward(self, x):
        # x.shape        = [batch, input_num_capsule, input_dim_capsule]
        # u_expend.shape = [batch, 1, input_num_capsule, input_dim_capsule, 1]
        # u_tiled.shape  = [batch, num_capsule, input_num_capsule, input_dim_capsule, 1]
        # W.shape        = [       num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # matmul -> [dim_capsule, input_dim_capsule] x [input_dim_capsule, 1] = [dim_capsule, 1]
        # u_hat.shape    = [batch, num_capsule, input_num_capsule, dim_capsule, 1]
        u_expend = x[:, None, :, :, None]
        u_tiled = u_expend.expand(-1, self.out_num_capsule, self.in_num_capsule, self.in_dim_capsule, 1)
        u_hat = torch.matmul(self.W, u_tiled)
        # u_hat.shape    = [batch, num_capsule, input_num_capsule, dim_capsule]
        u_hat = torch.squeeze(u_hat)
        
        u_hat_stoped = u_hat.detach()
        
        b = Variable(torch.zeros(x.size(0), self.out_num_capsule, self.in_num_capsule))
        if x.is_cuda:
            b = b.cuda()
        for i in range(self.routings):
            # c.shape = b.shape = [batch, num_capsule, input_num_capsule]
            c = F.softmax(b, dim=1)
            # u_hat.shape    = [batch, num_capsule, input_num_capsule, dim_capsule]
            # c_expend.shape = [batch, num_capsule, input_num_capsule, 1]
            # s.shape        = [batch, num_capsule, 1, dim_capsule]
            s = torch.sum(c[:, :, :, None] * u_hat if i == self.routings - 1 else u_hat_stoped, dim=-2, keepdim=True)
            # v.shape = s.shape
            v = self.squash(s)
            if i == self.routings - 1:
                # v.shape            = [batch, num_capsule, 1, dim_capsule]
                # u_hat_stoped.shape = [batch, num_capsule, input_num_capsule, dim_capsule]
                # => b.shape = [batch, num_capsule, input_num_capsule]
                b += torch.sum(v * u_hat_stoped, dim=-1)
        # v.shape = [batch, num_capsule, 1, dim_capsule]
        return torch.squeeze(v, dim=-2)

# 掩码层
class CapReconMask(nn.Module):
    def __init__(self):
        super(CapReconMask, self).__init__()
        self.one_hot = CapTool().one_hot

    def forward(self, x, target=None, random=False):
        """
        :params target: 0~num_classes, LongTensor
        :params random: if `target` is None, should random generate a mask random? If false, `target` wiil be set to max $L2(x)$
        :return: shape = (batch, dim_capsules)
        """
        # x.shape = (batch, num_classes, dim_capsules) | (batch, num_capsules, dim_capsules)
        # target is a LongTensor, shape = (batch, )
        if target is None:
            target = torch.randint(0, x.size(1)-1, (x.size(0), 1)) if random else torch.norm(x, dim=2).max(dim=1).indices.squeeze()
            
        assert len(target.size()) == 1 and len(x.size()) == 3, \
            f"while recive with len of {len(target.size())} and {len(x.size())}"
        assert isinstance(target, torch.LongTensor) or isinstance(target, torch.cuda.LongTensor), f"while is `{target.type()}`"
        
        ont_hot_target = self.one_hot(target, num_dim=x.size(1))  # shape = (batch, num_classes)
        masked = (x * ont_hot_target[:, :, None]) # (batch, num_classes, dim_capsules)
        return masked.sum(dim=-2), target # masked.shape = (batch, dim_capsules)

# 