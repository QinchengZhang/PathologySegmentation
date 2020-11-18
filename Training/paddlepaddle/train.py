# -*- coding: utf-8 -*-
'''
Author: TJUZQC
Date: 2020-11-17 12:40:20
LastEditors: TJUZQC
LastEditTime: 2020-11-18 11:46:38
Description: None
'''
import paddle
from paddle.fluid.layers.nn import scale
from models import HSU_Net
from utils import *
from paddle.static import InputSpec

# paddle.set_device('cpu')
num_classes = 1
network = HSU_Net()
model = paddle.Model(network)
model.summary((-1, 3, 512, 512))

train_dataset = SegDataset("F:/DATASET/Beijing-small_cell_lung_cancer-pathology/imgs_color_normalized",
                           "F:/DATASET/Beijing-small_cell_lung_cancer-pathology/masks", scale=0.5) # 训练数据集
val_dataset = SegDataset("F:/DATASET/Beijing-small_cell_lung_cancer-pathology/test_images",
                         "F:/DATASET/Beijing-small_cell_lung_cancer-pathology/test_masks", train=False, scale=0.5) # 验证数据集

train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=1, shuffle=True)
val_loader = paddle.io.DataLoader(val_dataset, places=paddle.CPUPlace(), batch_size=1, shuffle=False)
network.train()

optim = paddle.optimizer.RMSProp(learning_rate=0.001,
                                 rho=0.9,
                                 momentum=0.0,
                                 epsilon=1e-07,
                                 centered=False,
                                 parameters=model.parameters())
# loss_func = paddle.nn.BCEWithLogitsLoss()

# for i in range(5):
#     for batch_id, data in enumerate(train_loader()):
#         x_data = data[0]            # 训练数据
#         y_data = data[1]            # 训练数据标签
#         x_data = x_data.astype('float32')
#         y_data = y_data.astype('float32')
#         print(x_data.dtype, y_data.dtype)
#         predicts = network(x_data)    # 预测结果
#         # print(predicts)

#         # 计算损失 等价于 prepare 中loss的设置
#         loss = loss_func(predicts, y_data)

#         # 计算准确率 等价于 prepare 中metrics的设置
#         m_iou, _, _ = paddle.metric.mean_iou(predicts.astype('int32'), y_data.astype('int32'), 1)
#         # print(m_iou)

#         # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中

#         # 反向传播
#         loss.backward()

#         if (batch_id+1) % 5 == 0:
#             print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(i, batch_id, loss.numpy(), m_iou.numpy()))

#         # 更新参数
#         optim.step()

#         # 梯度清零
#         optim.clear_grad()
input = InputSpec([None, 3, 512, 512], 'float32', 'x')
label = InputSpec([None, 1, 512, 512], 'int64', 'label')
model.prepare(optim, paddle.nn.BCEWithLogitsLoss()) 
# model.load('E:/WorkSpaces/PythonWorkSpace/PathologySegmentation/Training/paddlepaddle/checkpoints_WSI/train/final.pdparams')
# model.save('E:/WorkSpaces/PythonWorkSpace/PathologySegmentation/Training/paddlepaddle/checkpoints_WSI/test', training=False)
model.fit(train_dataset,
          val_dataset,
          epochs=1,
          batch_size=1,
          verbose=1,
          save_dir="E:/WorkSpaces/PythonWorkSpace/PathologySegmentation/Training/paddlepaddle/checkpoints_WSI",
          num_workers=8)
# model.load('E:/WorkSpaces/PythonWorkSpace/PathologySegmentation/Training/paddlepaddle/checkpoints_WSI/train/final.pdparams')
model.save('E:/WorkSpaces/PythonWorkSpace/PathologySegmentation/Training/paddlepaddle/checkpoints_WSI/test', training=False)