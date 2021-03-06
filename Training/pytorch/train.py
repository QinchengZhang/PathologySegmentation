# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-10-26 10:26:51
LastEditors: TJUZQC
LastEditTime: 2020-11-20 19:23:55
Description: None
'''
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluation import eval_net
from models import ChooseModel, init_weights
from utils.dataset import BasicDataset

conf = yaml.load(open(os.path.join(
    sys.path[0], 'config', 'config.yaml')), Loader=yaml.FullLoader)
dir_img = conf['DATASET']['IMGS_DIR']
dir_mask = conf['DATASET']['MASKS_DIR']
dir_checkpoint = conf['MODEL']['CHECKPOINT_DIR']


def train_net(net,
              device,
              epochs=5,
              batch_size=16,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              use_apex=False,
              optimizer='adam',
              classes=2,
              lr_scheduler='steplr',
              lr_scheduler_cfgs: dict = {'step_size': 10}):

    dataset = BasicDataset(dir_img, dir_mask, img_scale,
                           train=True, classes=classes)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(
        comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Use apex: {use_apex}
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
    optimizer = optimizers.get(optimizer, None)(
        net.parameters(), lr=lr, weight_decay=1e-8)
    lr_scheduler_getter = {
        'lambdalr': torch.optim.lr_scheduler.LambdaLR,
        'multiplicativelr': torch.optim.lr_scheduler.MultiplicativeLR,
        'steplr': torch.optim.lr_scheduler.StepLR,
        'multisteplr': torch.optim.lr_scheduler.MultiStepLR,
        'exponentiallr': torch.optim.lr_scheduler.ExponentialLR,
        'cosineannealinglr': torch.optim.lr_scheduler.CosineAnnealingLR,
        'reducelronplateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'cycliclr': torch.optim.lr_scheduler.CyclicLR,
        'onecyclelr': torch.optim.lr_scheduler.OneCycleLR,
    }
    lr_scheduler = lr_scheduler_getter.get(
        lr_scheduler.lower(), None)(optimizer, **lr_scheduler_cfgs)
    if use_apex:
        try:
            from apex import amp
            net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
        except ImportError as e:
            print(e)
            use_apex = False

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                if net.n_classes > 1:
                    b, c, w, h = true_masks.shape
                    true_masks = true_masks.view(b, w, h)
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                if not use_apex:
                    loss.backward()
                else:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                dataset_len = len(dataset)
                a1 = dataset_len // 10
                a2 = dataset_len / 10
                b1 = global_step % a1
                b2 = global_step % a2

                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    dice_coeff, pA, oA, precision, recall, f1score = eval_net(
                        net, val_loader, device, n_val)
                    if net.n_classes > 1:
                        logging.info(
                            'Validation cross entropy: {}'.format(dice_coeff))
                        writer.add_scalar('Loss/test', dice_coeff, global_step)

                    else:
                        logging.info(
                            'Validation Dice Coeff: {}'.format(dice_coeff))
                        writer.add_scalar('Dice/test', dice_coeff, global_step)
                        logging.info(
                            'Validation Pixel Accuracy: {}'.format(pA))
                        writer.add_scalar('pA/test', pA, global_step)
                        logging.info(
                            'Validation Overall Accuracy: {}'.format(oA))
                        writer.add_scalar('oA/test', oA, global_step)
                        logging.info(
                            'Validation Precision: {}'.format(precision))
                        writer.add_scalar('precision/test',
                                          precision, global_step)
                        logging.info(
                            'Validation Recall: {}'.format(recall))
                        writer.add_scalar('recall/test', recall, global_step)
                        logging.info(
                            'Validation F1-score: {}'.format(f1score))
                        writer.add_scalar(
                            'F1-score/test', f1score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images(
                            'masks/true', true_masks, global_step)
                        writer.add_images(
                            'masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                lr_scheduler.step()

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       os.path.join(dir_checkpoint, f'CP_epoch{epoch + 1}_loss_{str(loss.item())}.pth'))
            logging.info(
                f'Checkpoint {epoch + 1} saved ! loss (batch) = ' + str(loss.item()))

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--network', metavar='NETWORK', type=str,
                        default=conf['MODEL']['MODEL_NAME'], help='network type', dest='network')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=conf['NUM_EPOCHS'],
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=conf['BATCH_SIZE'],
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=conf['LR'],
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=conf['MODEL']['PRETRAINED_MODEL_DIR'],
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=conf['SCALE'],
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=conf['VALIDATION'],
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-t', '--init-type', dest='init_type', type=str, default=conf['INIT_TYPE'],
                        help='Init weights type')
    parser.add_argument('-a', '--use-apex', dest='use_apex', type=str, default=conf['APEX'],
                        help='Automatic Mixed Precision')
    parser.add_argument('-o', '--optimizer', dest='optimizer',
                        type=str, default=conf['OPTIMIZER'], help='Optimizer type')
    parser.add_argument('-ls', '--lr-scheduler', dest='lr_scheduler',
                        type=str, default=conf['LR_SCHEDULER'], help='lr scheduler type')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available(
    ) and conf['DEVICE'].lower() == 'cuda' else 'cpu')
    logging.info(f'Using device {device}')

    network = args.network.lower()
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = ChooseModel(network)(
        n_channels=3, n_classes=conf['DATASET']['NUM_CLASSES'])
    assert net is not None, f'check your argument --network'

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling\n'
                 f'\tApex is {"using" if args.use_apex == "True" else "not using"}')
    init_weights(net, args.init_type)
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  use_apex=(args.use_apex == "True"),
                  optimizer=args.optimizer.lower(),
                  classes=conf['DATASET']['NUM_CLASSES'],
                  lr_scheduler=args.lr_scheduler,
                  lr_scheduler_cfgs=conf['LR_SCHEDULER_CFGS'])
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
