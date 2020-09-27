<!--
 * @Author: TJUZQC
 * @Date: 2020-09-27 12:40:56
 * @LastEditors: TJUZQC
 * @LastEditTime: 2020-09-27 15:27:40
 * @Description: None
-->
# Pathology segmentation implement with PyTorch.
## Network architecture: U-Net, Attention U-Net, R2 U-Net, R2 Attention U-Net, Capsule U-Net

**Segmentation Demo Result:**

**对比模式**
![Segmentation](http://qhax9tu5e.hb-bkt.clouddn.com/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20200927130059.png)

**合并模式**
![Segmentation](http://qhax9tu5e.hb-bkt.clouddn.com/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20200927130120.png)

---
## Dependencies

- Python 3.6
- PyTorch >= 1.1.0
- Torchvision >= 0.3.0
- future >= 0.18.2
- matplotlib >= 3.1.3
- numpy >= 1.16.0
- Pillow >= 6.2.0
- protobuf >= 3.11.3
- tensorboard >= 1.14.0
- tqdm >= 4.42.1
- openslide >= 1.1.1

---
## 文件结构
```
│  README.MD
│
├─Preprocessing
│  ├─ColorNormalisation
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
- ### [Preprocessing模块](./Preprocessing/README.MD)

- ### [SegmentationViewer模块](./SegmentationViewer/README.md)

- ### [Training模块](./Training/README.md)

---
## Thanks

The birth of this project is inseparable from the following projects:

- **[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)：PyTorch implementation of the U-Net for image semantic segmentation with high quality images**

---

