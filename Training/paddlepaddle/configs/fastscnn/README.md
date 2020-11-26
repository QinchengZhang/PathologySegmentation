<!--
 * @Author: TJUZQC
 * @Date: 2020-11-25 16:12:55
 * @LastEditors: TJUZQC
 * @LastEditTime: 2020-11-26 12:56:56
 * @Description: None
-->
# Fast-SCNN: Fast Semantic Segmentation Network

Fast-SCNN [7] 是一个面向实时的语义分割网络。在双分支的结构基础上，大量使用了深度可分离卷积和逆残差（inverted-residual）模块，并且使用特征融合构造金字塔池化模块 (Pyramid Pooling Module)来融合上下文信息。这使得Fast-SCNN在保持高效的情况下能学习到丰富的细节信息。

整个网络结构如下：

![](../_imgs_/fast-scnn.png)
## Reference

> [Poudel, Rudra PK, Stephan Liwicki, and Roberto Cipolla. "Fast-scnn: Fast semantic segmentation network." arXiv preprint arXiv:1902.04502 (2019).](https://arxiv.org/abs/1902.04502)

## Performance

### Cityscapes

| Model | Backbone | Resolution | Training Iters | mIoU | mIoU (multi-scale) | Links |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Fast SCNN|-|1024x1024|160000|69.31%|-|[model](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fastscnn_cityscapes_1024x1024_160k/model.pdparams) \| [log](https://bj.bcebos.com/paddleseg/dygraph/cityscapes/fastscnn_cityscapes_1024x1024_160k/train.log) \| [vdl](https://paddlepaddle.org.cn/paddle/visualdl/service/app/scalar?id=3b4c3f01c9213cac14e53c69d262a337)|
