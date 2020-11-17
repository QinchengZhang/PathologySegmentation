import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from evaluation import *
import torch.nn.functional as F
import sys
# from unet import U_Net, R2U_Net, AttU_Net, R2AttU_Net, init_weights
from models import U_Net, R2AttU_Net, R2U_Net, AttU_Net, HSU_Net, FCN1s, FCN8s, FCN16s, FCN32s, init_weights

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import yaml

conf = yaml.load(open(os.path.join(
    sys.path[0], 'config', 'config.yaml')), Loader=yaml.FullLoader)
dir_img = conf['DATASET']['IMGS_DIR']
dir_mask = conf['DATASET']['MASKS_DIR']

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

def evaluate_net(net,
                 device,
                 batch_size=16,
                 img_scale=0.5,
                 classes=2):

    dataset = BasicDataset(dir_img, dir_mask, img_scale,
                           train=True, classes=classes)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)
    n_val = len(dataset)
    writer = SummaryWriter(
        comment=f'BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Batch size:      {batch_size}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    optimizers = {
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sparseadam': optim.SparseAdam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'lbfgs': optim.LBFGS,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'sgd': optim.SGD,
    }

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    net.eval()

    epoch_loss = 0
    tot = 0
    PA = 0
    OA = 0
    pre = 0
    recal = 0
    f1s = 0
    params = None
    for batch in tqdm(val_loader):
        imgs = batch['image']
        true_masks = batch['mask']
        assert imgs.shape[1] == net.n_channels, \
            f'Network has been defined with {net.n_channels} input channels, ' \
            f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
            'the images are loaded correctly.'

        imgs = imgs.to(device=device, dtype=torch.float32)
        if params is None:
            params = count_param(net)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)
        if net.n_classes > 1:
            b, c, w, h = true_masks.shape
            true_masks = true_masks.view(b, w, h)
        masks_pred = net(imgs)
        loss = criterion(masks_pred, true_masks)
        epoch_loss += loss.item()
        writer.add_scalar('Loss/train', loss.item(), global_step)
        for true_mask, pred in zip(true_masks, masks_pred):
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
    epoch_loss /= n_val
    tot /= n_val
    PA /= n_val
    OA /= n_val
    pre /= n_val
    recal /= n_val
    f1s /= n_val

    if net.n_classes > 1:
        logging.info(f'Validation loss:{epoch_loss}')
        logging.info(
            'Validation cross entropy: {}'.format(tot))
        logging.info(f'Params in this model is: {params}')
        writer.add_scalar('Loss/test', tot, global_step)

    else:
        logging.info(f'Validation loss:{epoch_loss}')
        logging.info(
            'Validation Dice Coeff: {}'.format(tot))
        writer.add_scalar('Dice/test', tot, global_step)
        logging.info(
            'Validation Pixel Accuracy: {}'.format(PA))
        writer.add_scalar('pA/test', PA, global_step)
        logging.info(
            'Validation Overall Accuracy: {}'.format(OA))
        writer.add_scalar('oA/test', OA, global_step)
        logging.info(
            'Validation Precision: {}'.format(pre))
        writer.add_scalar('precision/test',
                          pre, global_step)
        logging.info(
            'Validation Recall: {}'.format(recal))
        writer.add_scalar('recall/test', recal, global_step)
        logging.info(
            'Validation F1-score: {}'.format(f1s))
        writer.add_scalar(
            'F1-score/test', f1s, global_step)
        logging.info(f'Params in this model is: {params}')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--network', metavar='NETWORK', type=str,
                        default=conf['MODEL']['MODEL_NAME'], help='network type', dest='network')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=conf['BATCH_SIZE'],
                        help='Batch size', dest='batchsize')
    parser.add_argument('-f', '--load', dest='load', type=str,
                        help='Load model from a .pth file', required=True)
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=conf['SCALE'],
                        help='Downscaling factor of the images')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available(
    ) and conf['DEVICE'].lower() == 'cuda' else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    network = args.network.lower()
    switch = {'unet': U_Net,
              'r2unet': R2U_Net,
              'attunet': AttU_Net,
              'r2attunet': R2AttU_Net,
              'hsunet': HSU_Net,
              'fcn8s': FCN8s,
              'fcn16s': FCN16s,
              'fcn32s': FCN32s,
              'fcn1s': FCN1s,
              }
    net = switch.get(network, None)(
        n_channels=3, n_classes=conf['DATASET']['NUM_CLASSES'])
    assert net is not None, f'check your argument --network'
    # net = AttU_Net(n_channels=3,n_classes=1)
    # net = AttU_Net(n_channels=3, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling\n')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        evaluate_net(net=net,
                     batch_size=args.batchsize,
                     device=device,
                     img_scale=args.scale,
                     classes=conf['DATASET']['NUM_CLASSES'])
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
