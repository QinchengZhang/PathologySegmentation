# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-17 15:51:56
LastEditors: TJUZQC
LastEditTime: 2020-10-26 14:44:11
Description: None
'''
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dice_loss import dice_coeff
from .pixel_accuracy import pixel_accuracy


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient and pixel accuracy"""
    net.eval()
    tot = 0
    acc = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs = batch['image']
            true_masks = batch['mask']

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            mask_pred = net(imgs)

            for true_mask, pred in zip(true_masks, mask_pred):
                pred = (pred > 0.5).float()
                if net.n_classes > 1:
                    tot += F.cross_entropy(pred.unsqueeze(dim=0),
                                           true_mask.unsqueeze(dim=0)).item()
                else:
                    acc += pixel_accuracy(pred,
                                          true_mask.squeeze(dim=1)).item()
                    tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])
    # print(acc/ n_val, tot / n_val)
    return tot / n_val, acc/ n_val
