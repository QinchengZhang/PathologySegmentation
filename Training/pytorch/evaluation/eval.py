# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-09-17 15:51:56
LastEditors: TJUZQC
LastEditTime: 2020-10-28 12:24:06
Description: None
'''
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dice_loss import dice_coeff
from .pixel_accuracy import pixel_accuracy
from .overall_accuracy import overall_accuracy
from .precision import precision
from .recall import recall
from .f1score import f1score


def eval_net(net, loader, device, n_val):
    """Evaluation without the densecrf with the dice coefficient, pixel accuracy, overall accuracy, precision, recall, f1score"""
    net.eval()
    tot = 0
    PA = 0
    OA = 0
    pre = 0
    recal = 0
    f1s = 0

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
                    PA += pixel_accuracy(pred,
                                         true_mask.squeeze(dim=1)).item()
                    tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                    OA += overall_accuracy(pred,
                                           true_mask.squeeze(dim=1)).item()
                    pre += precision(pred,
                                     true_mask.squeeze(dim=1)).item()
                    recal += recall(pred,
                                    true_mask.squeeze(dim=1)).item()
                    f1s += f1score(pred,
                                   true_mask.squeeze(dim=1)).item()
            pbar.update(imgs.shape[0])
    # print(acc/ n_val, tot / n_val)
    return tot / n_val, PA / n_val, OA/n_val, pre/n_val, recal/n_val, f1s/n_val
