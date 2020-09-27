# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-24 11:29:44
LastEditors: TJUZQC
LastEditTime: 2020-09-24 15:45:32
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


def one_hot(y, num_dim=10):
    """
    One Hot Encoding, similar to `torch.eye(num_dim).index_select(dim=0, index=y)`
    :param y: N-dim tenser
    :param num_dim: do one-hot labeling from `0` to `num_dim-1`
    :return: shape = (batch_size, num_dim)
    """
    one_hot_y = torch.zeros(y.size(0), num_dim)
    if y.is_cuda:
        one_hot_y = one_hot_y.cuda()
    return one_hot_y.scatter_(1, y.view(-1, 1), 1.)


def margin_loss(input, target, num_classes=10, m_plus=None, m_minus=None, m_lambda=0.5):
    """
    The non-linear activation used in Capsule.
    It drives the length of a large vector to near 1 and small vector to 0

    input.shape = (batch_size, num_classes)
    target.shape = (batch_size, ), type of `LongTensor`, True-Label of classifications

    :param input: Predict-ablility of classifications
    :param target: True-Label of classifications
    :param num_classes: 10
    :param m_plus: 0.9
    :param m_minus: 0.1
    :param m_lambda: 0.5
    :return: shape = (1, )
    """
    assert len(input.size()) == 2 and len(target.size()) == 1

    if m_plus is None or m_minus is None:
        m_minus = 1. / num_classes
        m_plus = 1. - m_minus
    y = one_hot(target, num_dim=num_classes)

    pos = y * torch.clamp(m_plus - input, min=0.) ** 2
    neg = (1.-y) * torch.clamp(input - m_minus, min=0.) ** 2
    loss = pos + m_lambda * neg

    return torch.mean(torch.sum(loss, dim=1))


def squash(s, dim=-1, constant=1, epsilon=1e-8):
    """
    It drives the length of a large vector to near 1 and small vector to 0
    :params s: N-dim tenser
    :params dim: the dimension to squash
    :params constant: (0, 1]
    :return: The same shape like `s`
    """
    norm_2 = torch.norm(s, p=2, dim=dim, keepdim=True)
    scale = norm_2**2 / (constant + norm_2**2) / (norm_2 + epsilon)
    return scale * s


def acc_eval(model, test_loader, loss_fn, y_pred_dim=0):
    with torch.no_grad():
        model.eval()
        test_loss, test_acc, test_size = 0., 0, len(test_loader.dataset)
        for x, y in test_loader:
            batch = x.size(0)
            x, y = Variable(x.cuda()), Variable(y.cuda())

            y_pred = model(x)
            loss = loss_fn(y_pred, y) if y_pred_dim == 0 else loss_fn(
                y_pred[y_pred_dim], y)

            test_loss += loss * batch
            test_acc += y_pred.data.max(1)[1].eq(
                one_hot(y).data.max(1)[1]).sum()
    return test_loss/test_size, test_acc.float()/test_size


def model_summary(model, show_layer_detail=True):
    import re
    sum_params = 0
    FORMAT, pat = "{0:<50s} | {1:<50s} | {2:<10s}", r'\B(?=(\d{3})+(?!\d))'
    print(FORMAT.format("Layer Weight Name", "Weight Shape", "Params"))
    print("-"*(50*2+20+6))
    for name, i in model.named_parameters():
        if len(i.size()) <= 1:
            pass
        if show_layer_detail:
            print(FORMAT.format(name, str(i.size()),
                                re.sub(pat, ',', str(i.numel()))))
        sum_params += i.numel()
    print("-"*(50*2+20+6), "\nTotal Params:",
          re.sub(pat, ',', str(sum_params)))


def draw_recon_pics(images, pics_per_line=10, labels=None, label_text_x=16, label_text_y=36):
    if images.is_cuda:
        images = images.cpu()
    rows = images.size(0) // pics_per_line + 1
    plt.figure(figsize=(16, 9))
    for ind in range(images.size(0)):
        plt.subplot(rows, pics_per_line, ind+1)
        if labels is not None:
            plt.text(label_text_x, label_text_y, labels[ind].item())
        plt.xticks([]), plt.yticks([])
        pic = images[ind].permute(1, 2, 0).detach().numpy()
        pic -= pic.min()
        pic = pic / pic.max() * 255
        plt.imshow(pic.astype(int))
    plt.show()


class CapTool():
    def __init__(self):
        pass

    @staticmethod
    def one_hot(cls, y, num_dim=10):
        """
        One Hot Encoding, similar to `torch.eye(num_dim).index_select(dim=0, index=y)`
        :param y: N-dim tenser
        :param num_dim: do one-hot labeling from `0` to `num_dim-1`
        :return: shape = (batch_size, num_dim)
        """
        one_hot_y = torch.zeros(y.size(0), num_dim)
        if y.is_cuda:
            one_hot_y = one_hot_y.cuda()
        return one_hot_y.scatter_(1, y.view(-1, 1), 1.)

    @staticmethod
    def margin_loss(cls, input, target, num_classes=10, m_plus=None, m_minus=None, m_lambda=0.5):
        """
        The non-linear activation used in Capsule. 
        It drives the length of a large vector to near 1 and small vector to 0

        input.shape = (batch_size, num_classes)
        target.shape = (batch_size, ), type of `LongTensor`, True-Label of classifications

        :param input: Predict-ablility of classifications
        :param target: True-Label of classifications
        :param num_classes: 10
        :param m_plus: 0.9
        :param m_minus: 0.1
        :param m_lambda: 0.5
        :return: shape = (1, )
        """
        assert len(input.size()) == 2 and len(target.size()) == 1

        if m_plus is None or m_minus is None:
            m_minus = 1. / num_classes
            m_plus = 1. - m_minus
        y = cls.one_hot(target, num_dim=num_classes)

        pos = y * torch.clamp(m_plus - input, min=0.) ** 2
        neg = (1.-y) * torch.clamp(input - m_minus, min=0.) ** 2
        loss = pos + m_lambda * neg

        return torch.mean(torch.sum(loss, dim=1))

    @staticmethod
    def squash(cls, s, dim=-1, constant=1, epsilon=1e-8):
        """
        It drives the length of a large vector to near 1 and small vector to 0
        :params s: N-dim tenser
        :params dim: the dimension to squash
        :params constant: (0, 1]
        :return: The same shape like `s`
        """
        norm_2 = torch.norm(s, p=2, dim=dim, keepdim=True)
        scale = norm_2**2 / (constant + norm_2**2) / (norm_2 + epsilon)
        return scale * s

    @staticmethod
    def acc_eval(cls, model, test_loader, loss_fn, y_pred_dim=0):
        with torch.no_grad():
            model.eval()
            test_loss, test_acc, test_size = 0., 0, len(test_loader.dataset)
            for x, y in test_loader:
                batch = x.size(0)
                x, y = Variable(x.cuda()), Variable(y.cuda())

                y_pred = model(x)
                loss = loss_fn(y_pred, y) if y_pred_dim == 0 else loss_fn(
                    y_pred[y_pred_dim], y)

                test_loss += loss * batch
                test_acc += y_pred.data.max(1)[1].eq(
                    cls.one_hot(y).data.max(1)[1]).sum()
        return test_loss/test_size, test_acc.float()/test_size

    @staticmethod
    def model_summary(cls, model, show_layer_detail=True):
        import re
        sum_params = 0
        FORMAT, pat = "{0:<50s} | {1:<50s} | {2:<10s}", r'\B(?=(\d{3})+(?!\d))'
        print(FORMAT.format("Layer Weight Name", "Weight Shape", "Params"))
        print("-"*(50*2+20+6))
        for name, i in model.named_parameters():
            if len(i.size()) <= 1:
                pass
            if show_layer_detail:
                print(FORMAT.format(name, str(i.size()),
                                    re.sub(pat, ',', str(i.numel()))))
            sum_params += i.numel()
        print("-"*(50*2+20+6), "\nTotal Params:",
              re.sub(pat, ',', str(sum_params)))

    @staticmethod
    def draw_recon_pics(cls, images, pics_per_line=10, labels=None, label_text_x=16, label_text_y=36):
        if images.is_cuda:
            images = images.cpu()
        rows = images.size(0) // pics_per_line + 1
        plt.figure(figsize=(16, 9))
        for idx in range(images.size(0)):
            plt.subplot(rows, pics_per_line, idx+1)
            if labels is not None:
                plt.text(label_text_x, label_text_y, labels[idx].item())
            plt.xticks([]), plt.yticks([])
            pic = images[idx].permute(1, 2, 0).detach().numpy()
            pic -= pic.min()
            pic = pic / pic.max() * 255
            plt.imshow(pic.astype(int))
        plt.show()
