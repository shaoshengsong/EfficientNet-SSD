# EfficientNet-SSD
Object Detection using EfficientNet

环境

操作系统: Ubuntu18.04

Python: 3.6

PyTorch: 1.1.0


论文 EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

为了实现将EfficientNet应用于目标检测，需要对其网络进行改造,改造后的结果，起名字就是EfficientNet-SSDLite 或者 EfficientNet-SSD
该代码也可以在CPU下运行

还有一篇是改造MobileNetV3的，将MobileNetV3应用于目标检测

https://github.com/shaoshengsong/MobileNetV3-SSD


以 efficient_net_b0_ssd300_voc0712 为例说明是如何改造的

其他的
EfficientNet-B1
EfficientNet-B2
EfficientNet-B3
EfficientNet-B4
EfficientNet-B5
EfficientNet-B6
EfficientNet-B7
都可以按照此方法就行改造，改造后至于是否使用SSD的 其他变种就是另一码事了

配置efficient_net_b0_ssd300_voc0712.yaml如下
```
MODEL:
  NUM_CLASSES: 21
  BOX_PREDICTOR: 'SSDLiteBoxPredictor'
  BACKBONE:
    NAME: 'efficient_net-b0'
    OUT_CHANNELS: (40, 112, 320, 256, 256, 256)
INPUT:
  IMAGE_SIZE: 300
DATASETS:
  TRAIN: ("voc_2007_trainval","voc_2012_trainval")
  TEST: ("voc_2007_test", )
SOLVER:
  MAX_ITER: 160000
  LR_STEPS: [105000, 135000]
  GAMMA: 0.1
  BATCH_SIZE: 2
  LR: 1e-3

OUTPUT_DIR: 'outputs/efficient_net_b0_ssd300_voc0712'
```

代码改造
```
efficientnet-b3的数据
INDICES = {
    'efficientnet-b3': [7, 17, 25]
}

EXTRAS = {
    'efficientnet-b3': [
        # in,  out, k, s, p
        [(384, 128, 1, 1, 0), (128, 256, 3, 2, 1)],  # 5 x 5
        [(256, 128, 1, 1, 0), (128, 256, 3, 1, 0)],  # 3 x 3
        [(256, 128, 1, 1, 0), (128, 256, 3, 1, 0)],  # 1 x 1

    ]
}

efficientnet-b0的数据
INDICES = {
    'efficientnet-b0': [4, 10, 15]
}
EXTRAS = {
    'efficientnet-b0': [
        # in,  out, k, s, p
        [(320, 128, 1, 1, 0), (128, 256, 3, 2, 1)],  # 5 x 5
        [(256, 128, 1, 1, 0), (128, 256, 3, 1, 0)],  # 3 x 3
        [(256, 128, 1, 1, 0), (128, 256, 3, 1, 0)],  # 1 x 1

    ]
}
```
其他详细的改造见github



数据集的下载
下载VOC数据集
可以通过以下命令下载数据集
```
切换到项目的数据目录
cd data
 下载2007年的训练数据
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
 下载2007年的测试数据
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
 下载2012年的训练数据
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```
解压数据集
下载完成之后，要解压数据集到当前目录
```
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtrainval_11-May-2012.tar

VOC的目录是这样的
VOC_ROOT
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ ...

COCO的目录是这样的
COCO_ROOT
|__ annotations
    |_ instances_valminusminival2014.json
    |_ instances_minival2014.json
    |_ instances_train2014.json
    |_ instances_val2014.json
    |_ ...
|__ train2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ val2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ ...
```
下载数据集之后唯一需要做的是更改数据集的路径，
SSD/ssd/config/path_catlog.py

改成自己数据集的路径
```
class DatasetCatalog:
    DATA_DIR = '/media/santiago/b/dataset/VOC'

训练方法
python train.py --config-file configs/efficient_net_b0_ssd300_voc0712.yaml


代码部分严重参考以下repository

EfficientNet的PyTorch的实现
https://github.com/lukemelas/EfficientNet-PyTorch

High quality, fast, modular reference implementation of SSD in PyTorch
https://github.com/lufficc/SSD
```
