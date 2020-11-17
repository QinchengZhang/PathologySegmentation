# Pathology segmentation implement with PyTorch.
## Network architecture: U-Net, Attention U-Net, R2 U-Net, R2 Attention U-Net, Capsule U-Net

**Segmentation Demo Result:**

**对比模式**
![Segmentation](http://qhax9tu5e.hb-bkt.clouddn.com/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20200927130059.png)

**合并模式**
![Segmentation](http://qhax9tu5e.hb-bkt.clouddn.com/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20200927130120.png)

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


###### We augment the number of images by perturbing them withrotation and scaling. Four rotation angles{−45◦,−22◦,22◦,45◦}and four scales{0.6,0.8,1.2,1.5}are used. We also apply four different Gamma transforms toincrease color variation. The Gamma values are{0.5,0.8,1.2,1.5}. After thesetransforms, we have 18K training images. 

## Run locally
**Note : Use Python 3**
### Prediction

You can easily test the output masks on your images via the CLI.

To predict a single image and save it:

```bash
$ python predict.py -i image.jpg -o output.jpg
```

To predict a multiple images and show them without saving them:

```bash
$ python predict.py -i image1.jpg image2.jpg --viz --no-save
```

```shell script
> python predict.py -h
usage: predict.py [-h] [--network NETWORK] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit

  --network NETWORK, -w NETWORK
                        network type (default: UNet)
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```
You can specify which model file to use with `--model MODEL.pth`.

### Training

```shell script
> python train.py -h
usage: train.py [-h] [-n NETWORK] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL] [-t INIT_TYPE] [-a USE_APEX]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -n NETWORK, --network N     network type (default: UNet)
  -e E, --epochs E      Number of epochs (default: 30)
  -b [B], --batch-size [B]
                        Batch size (default: 8)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.0001)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 10.0)
  -t INIT_TYPE, --init-type INIT_TYPE
                        init weights type (default: kaiming)
  -a USE_APEX, --use-apex USE_APEX
                        Automatic Mixed Precision (default: True)

```
By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

## Thanks

The birth of this project is inseparable from the following projects:

- **[Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)：PyTorch implementation of the U-Net for image semantic segmentation with high quality images**

---

