<!--
 * @Author: TJUZQC
 * @Date: 2020-11-22 14:57:05
 * @LastEditors: TJUZQC
 * @LastEditTime: 2020-11-26 13:00:18
 * @Description: None
-->
# PaddleSeg（动态图版本）

本目录提供了PaddleSeg的动态图版本修改版，基于PaddleSeg的动态图版本

## 模型库

|模型\骨干网络|ResNet50|ResNet101|HRNetw18|HRNetw48|
|-|-|-|-|-|
|[ANN](./configs/ann)|✔|✔|||
|[BiSeNetv2](./configs/bisenet)|-|-|-|-|
|[DANet](./configs/danet)|✔|✔|||
|[Deeplabv3](./configs/deeplabv3)|✔|✔|||
|[Deeplabv3P](./configs/deeplabv3p)|✔|✔|||
|[Fast-SCNN](./configs/fastscnn)|-|-|-|-|
|[FCN](./configs/fcn)|||✔|✔|
|[GCNet](./configs/gcnet)|✔|✔|||
|[OCRNet](./configs/ocrnet/)|||✔|✔|
|[PSPNet](./configs/pspnet)|✔|✔|||
|[UNet](./configs/unet)|-|-|-|-|
|[HS-UNet](./configs/unet)|-|-|-|-|

## 数据集

- [x] Cityscapes
- [x] Pascal VOC
- [x] ADE20K
- [ ] Pascal Context
- [ ] COCO stuff

## 安装

1. 安装PaddlePaddle

版本要求

* PaddlePaddle >= 2.0.0rc

* Python >= 3.6+

由于图像分割模型计算开销大，推荐在GPU版本的PaddlePaddle下使用PaddleSeg。推荐安装10.0以上的CUDA环境。


安装教程请见[PaddlePaddle官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-beta/install/index_cn.html)。


2. 下载PaddleSeg代码
```shell
git clone https://github.com/PaddlePaddle/PaddleSeg
```

3. 安装PaddleSeg依赖
通过以下命令安装python包依赖，请确保在该分支上至少执行过一次以下命令：


```shell
cd PaddleSeg/dygraph
export PYTHONPATH=`pwd`
# windows下请执行以下命令
# set PYTHONPATH=%cd%
pip install -r requirements.txt
```

## 训练
```shell
python train.py --config configs/quick_start/bisenet_optic_disc_512x512_1k.yml
```

## 使用教程

* [快速入门](./docs/quick_start.md)
* [数据集准备](./docs/data_prepare.md)
* [配置项](./configs/)
