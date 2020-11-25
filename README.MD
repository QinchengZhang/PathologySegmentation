<!--
 * @Author: TJUZQC
 * @Date: 2020-09-27 12:40:56
 * @LastEditors: TJUZQC
 * @LastEditTime: 2020-11-25 11:34:35
 * @Description: None
-->
# Pathology segmentation implement with PyTorch.
## Network architecture: U-Net, Attention U-Net, R2 U-Net, R2 Attention U-Net,  HSU-Net

**Segmentation Demo Result:**

**对比模式**
![Segmentation](http://pic.tjuzqc.top/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20200927130059.png)

**合并模式**
![Segmentation](http://pic.tjuzqc.top/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20200927130120.png)

---
## Dependencies

```
见各模块的Dependencies
```

---
## 文件结构
```
│  README.MD
│
├─Preprocessing
│  ├─ColorNormalization
|  |
│  |    
│  ├─DataAugmentation
│  │
|  |        
│  └─DataProcess
│          
├─SegmentationViewer
│
│          
└─Training
            
```
- ### [Preprocessing模块](./Preprocessing/README.md)

- ### [SegmentationViewer模块](./SegmentationViewer/README.md)

- ### [Training模块-PyTorch实现](./Training/pytorch/README.md)

- ### [Training模块-PaddlePaddle实现](./Training/paddlepaddle/README.md)

---
## Thanks

The birth of this project is inseparable from the following projects:

- **[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)：PyTorch implementation of the U-Net for image semantic segmentation with high quality images**

- **[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg): PaddleSeg是基于PaddlePaddle开发的端到端图像分割开发套件，覆盖了DeepLabv3+, U-Net, ICNet, PSPNet, HRNet, Fast-SCNN等主流分割网络。通过模块化的设计，以配置化方式驱动模型组合，帮助开发者更便捷地完成从训练到部署的全流程图像分割应用。**

- **[VisualDL](https://github.com/PaddlePaddle/VisualDL): VisualDL是飞桨可视化分析工具，以丰富的图表呈现训练参数变化趋势、模型结构、数据样本、高维数据分布等。可帮助用户更清晰直观地理解深度学习模型训练过程及模型结构，进而实现高效的模型优化。**

- **[Fast_WSI_Color_Norm](https://github.com/abhinavdhere/Fast_WSI_Color_Norm.git): Codes for Fast GPU-Enabled Color Normalization of Whole Slide Images in Digital Pathology**

- **[ASAP](https://github.com/computationalpathologygroup/ASAP.git): ASAP is an open source platform for visualizing, annotating and automatically analyzing whole-slide histopathology images.**

---