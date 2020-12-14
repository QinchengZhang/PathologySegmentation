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

import os

import cv2
import numpy as np
import paddle
import tqdm

from paddleseg import utils
from PIL import Image
import paddleseg.utils.logger as logger


def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

def get_color_map_list(num_classes):
    """ Returns the color map for visualizing the segmentation mask,
        which can support arbitrary number of classes.
    Args:
        num_classes: Number of classes
    Returns:
        The color map
    """
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3

    return color_map

def predict(model,
            model_path,
            transforms,
            image_list,
            image_dir=None,
            save_dir='output'):
    """
    predict and visualize the image_list.

    Args:
        model (nn.Layer): Used to predict for input image.
        model_path (str): The path of pretrained model.
        transforms (transform.Compose): Preprocess for input image.
        image_list (list): A list of images to be predicted.
        image_dir (str): The directory of the images to be predicted. Default: None.
        save_dir (str): The directory to save the visualized results. Default: 'output'.

    """
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    model.eval()

    added_saved_dir = os.path.join(save_dir, 'added_prediction')
    pred_saved_dir = os.path.join(save_dir, 'pseudo_color_prediction')
    pred_one_channel_saved_dir = os.path.join(save_dir, 'one_channel_pseudo_color_prediction')

    logger.info("Start to predict...")
    for im_path in tqdm.tqdm(image_list):
        im, im_info, _ = transforms(im_path)
        im = im[np.newaxis, ...]
        im = paddle.to_tensor(im)
        logits = model(im)
        pred = paddle.argmax(logits[0], axis=1)
        pred = pred.numpy()
        pred = np.squeeze(pred).astype('uint8')
        for info in im_info[::-1]:
            if info[0] == 'resize':
                h, w = info[1][0], info[1][1]
                pred = cv2.resize(pred, (w, h), cv2.INTER_NEAREST)
            elif info[0] == 'padding':
                h, w = info[1][0], info[1][1]
                pred = pred[0:h, 0:w]
            else:
                raise ValueError("Unexpected info '{}' in im_info".format(
                    info[0]))

        # get the saved name
        if image_dir is not None:
            im_file = im_path.replace(image_dir, '')
        else:
            im_file = os.path.basename(im_path)
        if im_file[0] == '/':
            im_file = im_file[1:]

        # save added image
        added_image = utils.visualize(im_path, pred, weight=0.6)
        added_image_path = os.path.join(added_saved_dir, im_file)
        mkdir(added_image_path)
        cv2.imwrite(added_image_path, added_image)

        # save pseudo color prediction
        pred_im = utils.visualize(im_path, pred, weight=0.0)
        pred_saved_path = os.path.join(pred_saved_dir, im_file)
        mkdir(pred_saved_path)
        cv2.imwrite(pred_saved_path, pred_im)

        # save one channel pseudo color prediction
        pred_im_one_channel = Image.fromarray(pred)
        pred_im_one_channel = pred_im_one_channel.convert('P')
        colormap = get_color_map_list(256)
        pred_im_one_channel.putpalette(colormap)
        pred_one_channel_saved_path = os.path.join(pred_one_channel_saved_dir, im_file)
        mkdir(pred_one_channel_saved_path.replace('jpg', 'png'))
        pred_im_one_channel.save(pred_one_channel_saved_path.replace('jpg', 'png'))
