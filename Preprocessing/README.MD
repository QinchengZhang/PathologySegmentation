<!--
 * @Author: TJUZQC
 * @Date: 2020-09-27 14:20:01
 * @LastEditors: TJUZQC
 * @LastEditTime: 2020-09-29 00:33:18
 * @Description: None
-->
# Preprocessing pathology data
**Dependencies**
```
python 3.6.1
ASAP 1.9(ASAP不是一个python库，不能用pip安装，请打开ASAP github主页下载对应release)
numpy == 1.19.2
opencv == 4.4
Pillow == 7.2.0
matplotlib == 3.3.2
```
[ASAP github主页](https://github.com/computationalpathologygroup/ASAP.git)
**ASAP forpython安装教程**
```
Configure your PYTHONPATH environment variable to contain the /bin directory path.
大体就是建立一个PYTHONPATH名字的环境变量，找到你安装ASAP的路径，将路径>>到/bin 这个路径放在PYTHONPATH环境变量中，这样，python就可以自动搜索到这个包了哦。
```
## 文件结构
```
│  README.MD
│  
├─ColorNormalisation
│      Estimate_W.py
│      main.py
│      Run_ColorNorm.py
│      Run_StainSep.py
│      warwick_target_p.png
│      
├─DataAugmentation
│  │  test.ipynb
│  │  
│  └─Augmentation
│      │  augmentOps.py
│      │  basicOps.py
│      │  __init__.py
│              
├─DataProcess
│      ASAP.py
│      DataProcess.py
│      getPatch.py
│      transformFormat.py
|      main.py
|      test.py
        
```
### ColorNormalization模块
病理图片的成像和很多因素有关系，制作过程，扫描的机器等都会对图像造成颜色的结构上面的不同。这种不同对人眼来说影响比较小，虽然颜色略有不同，但是还是可以分辨的。但是对于深度学习等算法来说就不太能适应这种不同。还有一个很大的原因是，也就是这种颜色和结构上的不同，使得不同机型， 不同机构甚至是不同操作员扫描出来的病理图片不能够放在一起用。
处理这种颜色差异的方法，就是颜色的标准化。叫做color normalization或者是stain normalization。原理就是将所有的图片都标准化到同一张模板图片的颜色分布上。

### DataAugmentation模块
通过翻转、镜像、平移等操作扩大样本量

### DataProcess模块
将ndpi格式文件转其他格式、xml格式标注转图片mask、切patch等操作