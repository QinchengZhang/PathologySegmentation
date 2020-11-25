<!--
 * @Author: TJUZQC
 * @Date: 2020-09-27 12:40:56
 * @LastEditors: TJUZQC
 * @LastEditTime: 2020-11-25 10:08:46
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

- **[Fast_WSI_Color_Norm](https://github.com/abhinavdhere/Fast_WSI_Color_Norm.git): Codes for Fast GPU-Enabled Color Normalization of Whole Slide Images in Digital Pathology**

- **[ASAP](https://github.com/computationalpathologygroup/ASAP.git): ASAP is an open source platform for visualizing, annotating and automatically analyzing whole-slide histopathology images.**

---