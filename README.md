# ST++

This is the official PyTorch implementation of our CVPR 2022 paper:

> [**ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation**](https://arxiv.org/abs/2106.05095)       
> Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, Yang Gao        
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2022*

We have another simple yet stronger end-to-end framework **UniMatch** accepted by CVPR 2023:

> **[Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2208.09910)** [[Code](https://github.com/LiheYoung/UniMatch)]</br>
> Lihe Yang, Lei Qi, Litong Feng, Wayne Zhang, Yinghuan Shi</br>
> *In Conference on Computer Vision and Pattern Recognition (CVPR), 2023*

## Getting Started

### Data Preparation

#### Pre-trained Model

[ResNet-50](https://download.pytorch.org/models/resnet50-0676ba61.pth) | [ResNet-101](https://download.pytorch.org/models/resnet101-63fe2227.pth) | [DeepLabv2-ResNet-101](https://drive.google.com/file/d/14be0R1544P5hBmpmtr8q5KeRAvGunc6i/view?usp=sharing)


#### File Organization

```
├── ./pretrained
    ├── resnet50.pth
    ├── resnet101.pth
    └── deeplabv2_resnet101_coco_pretrained.pth
    
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
```


### Training and Testing

```
To run, type in the command line ./dataset1.sh, ./dataset2.sh and ./lisc.sh```.


## Acknowledgement

The DeepLabv2 MS COCO pre-trained model is borrowed and converted from **AdvSemiSeg**.
The image partitions are borrowed from **Context-Aware-Consistency** and **PseudoSeg**. 
Part of the training hyper-parameters and network structures are adapted from **PyTorch-Encoding**. The strong data augmentations are borrowed from **MoCo v2** and **PseudoSeg**.
 
+ AdvSemiSeg: [https://github.com/hfslyc/AdvSemiSeg](https://github.com/hfslyc/AdvSemiSeg).
+ Context-Aware-Consistency: [https://github.com/dvlab-research/Context-Aware-Consistency](https://github.com/dvlab-research/Context-Aware-Consistency).
+ PseudoSeg: [https://github.com/googleinterns/wss](https://github.com/googleinterns/wss).
+ PyTorch-Encoding: [https://github.com/zhanghang1989/PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding).
+ MoCo: [https://github.com/facebookresearch/moco](https://github.com/facebookresearch/moco).
+ OpenSelfSup: [https://github.com/open-mmlab/OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup).


