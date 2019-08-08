# Bonnetal! 

[![Build Status](https://travis-ci.org/PRBonn/bonnetal.svg?branch=master)](https://travis-ci.org/PRBonn/bonnetal)

![](https://i.ibb.co/6gyDvB6/bonnetal.jpg" )
![](https://i.ibb.co/8xLz81j/segmentation-only.png )
Example semantic segmentation of People vs Background using one of the included, real-time, architectures (running at 100FPS).

By [Andres Milioto](http://www.ipb.uni-bonn.de/people/andres-milioto/) _et.al_ @ University of Bonn.

In early 2018 we released [Bonnet](https://github.com/PRBonn/bonnet), which is a real-time, robotics oriented semantic segmentation framework using Convolutional Neural Networks (CNNs).
Bonnet provides an easy pipeline to add architectures and datasets for semantic segmentation, in order to train and deploy CNNs on a robot. It contains a full training pipeline in Python using Tensorflow and OpenCV, and it also some C++ apps to deploy a CNN in ROS and standalone. The C++ library is made in a way which allows to add other backends (such as TensorRT).

Back then, most of my research was in the field of semantic segmentation, so that was what the framework was therefore tailored specifically to do. Since then, we have found a way to make things even more awesome, allowing for a suite of other tasks, like classification, detection, instance and semantic segmentation, feature extraction, counting, etc. Hence, the new name of this new framework: "Bonnetal", reflects that this is nothing but the old Bonnet, and then some. Hopefully, the explict _et.al._ will also spawn further collaboration and many pull requests! :smile:

We've also switched to PyTorch to allow for easier mixing of backbones, decoders, and heads for different tasks. If you are still comfortable with just semantic segmentation, and/or you're a fan of TensorFlow, you can still find the original Bonnet [here](https://github.com/PRBonn/bonnet). Otherwise, keep on reading, and I'll try to explain why Bonnetal rules!

```
DISCLAIMER: I am currently bringing all the functionality out from a previously closed-source framework, so be patient if the task/weights are a placeholder, and send me an email to ask for a schedule on the particular part that you need.
```
---
## Description

This code provides a framework to mix-match popular, imagenet-trained, backbones with different decoders to achieve different CNN-enabled tasks. All of these have pre-trained imagenet weights when used, that get downloaded by default if the conditions are met.

- Backbones included are (so far):
  - [ResNet 18](https://arxiv.org/abs/1512.03385)
  - [ResNet 34](https://arxiv.org/abs/1512.03385)
  - [ResNet 50](https://arxiv.org/abs/1512.03385)
  - [ResNet 101](https://arxiv.org/abs/1512.03385)
  - [ResNet 152](https://arxiv.org/abs/1512.03385)
  - [Darknet 53](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
  - [Darknet 21](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
  - [ERFNet](http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)
  - [MobilenetsV2](https://arxiv.org/abs/1801.04381)

The main reason for the "lack" of variety of backbones so far is that imagenet pre-training takes a while, and it is pretty resource intensive. If you want a new backbone implemented we can talk about it, and you can share your resources to pretrain it :smiley: (PR's welcome :wink:)

- Tasks included are:
  
  - Full-image classification: [/train](train/tasks/classification), [/deploy](deploy/src/classification).
  - Semantic Segmentation: [/train](train/tasks/segmentation), [/deploy](deploy/src/segmentation).
  - More coming...
  <!-- - Instance Segmentation: [/train](train/tasks/instances), [/deploy](deploy/src/instances).
  - Object Detection: [/train](train/tasks/detection), [/deploy](deploy/src/detection),
  - CNN Keypoint/Feature Extraction: [/train](train/tasks/features), [/deploy](deploy/src/features).
  - Counting: [/train](train/tasks/counting), [/deploy](deploy/src/counting). -->

The code is (like the original Bonnet) separated into a [training](train/) part developed in Python, using Pytorch, and a [deployment/inference](deploy/) part, which is fully written in C++, and contains the code to run on the robot, either using ROS or standalone.

### Docker!

An [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) container is provided to run the full framework, and as a dependency check, as well as for the continuous integration. You can check the instructions to run the containers in [/docker](docker/).

### Training

[_/train_](train/) contains Python code to easily mix and match backbones and decoders in order to train them for different image recognition tasks. It also contains helper scripts for other tasks such as converting graphs to ONNX for inference, getting image statistics for normalization, class statistics in the dataset, inference tests, accuracy assessment, etc, etc.

### Deployment

[_/deploy_](deploy/) contains C++ code for deployment on edge. Every task has its own library and namespace, and every package is a catkin package. Therefore, each `task` has 4 catkin packages:
  - A `lib` package that contains all inference files for the library.
  - A `standalone` package that shows how to use the library linked to a standalone C++ application.
  - A `ros` package that contains a node handler and some nodes to use the library with ROS for the sensor data message-passing, and
  - (optionally) a `msg` package that defines the messages required for a specific task, should this be required.

Inference is done either:
  - By generating a PyTorch traced model through the python interface that can be infered with the `libtorch` library, both on GPU and CPU, or
  - By generating an ONNX model through the python interface, that is later picked up by TensorRT, profiled in the individual computer looking at available memory and half precision capabilities, and inferer with the TensorRT engine. Notice that not all architectures are supported by TensorRT and we cannot take responsibility for this, so when you implement an architecture, do a quick test that it works with tensorRT before training it and it will make your life easier. 

### Pre-trained models

Imagenet pretrained weights for the backbones are downloaded directly to the backbones in first use, so they never start from scratch. Whenever you use a backbone for a task, if the image is RGB, then the weights from imagenet are downloaded into the backbone (unless a specific pretrained model is otherwise explicitly stated in the parameters).

These are the currently trained models we have:

- Pretrained Backbones:
  - [ResNet 18](http://ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet18-5c106cde.pth)
  - [ResNet 34](http://ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet34-333f7ec4.pth)
  - [ResNet 50](http://ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet50-19c8e357.pth)
  - [ResNet 101](http://ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet101-5d3b4d8f.pth)
  - [ResNet 152](http://ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet152-b121ed2d.pth)
  - [Darknet 21](http://ipb.uni-bonn.de/html/projects/bonnetal/extractors/darknet/darknet21-4f2301c9.pth)
  - [Darknet 53](http://ipb.uni-bonn.de/html/projects/bonnetal/extractors/darknet/darknet53-0883870c.pth)
  - [ERFNet](http://ipb.uni-bonn.de/html/projects/bonnetal/extractors/erfnet/erfnet-87729049.pth)
  - [MobilenetsV2](http://ipb.uni-bonn.de/html/projects/bonnetal/extractors/mobilenetv2/mobilenetsv2-ee4468eb.pth)

- Classification
  - Imagenet:
    - [ResNet50 - 256px](http://ipb.uni-bonn.de/html/projects/bonnetal/classification/imagenet-resnet50-74.tar.gz)
    - [ResNet152 - 256px](http://ipb.uni-bonn.de/html/projects/bonnetal/classification/imagenet-resnet152-77.tar.gz)
    - [Darknet53 - 256px](http://ipb.uni-bonn.de/html/projects/bonnetal/classification/imagenet-darknet53-256-74.tar.gz)
    - [Darknet53 - 448px](http://ipb.uni-bonn.de/html/projects/bonnetal/classification/imagenet-darknet53-448-75.tar.gz)
    - [Darknet21 - 256px](http://ipb.uni-bonn.de/html/projects/bonnetal/classification/imagenet-darknet21-256-67.tar.gz)
    - [Darknet21 - 448px](http://ipb.uni-bonn.de/html/projects/bonnetal/classification/imagenet-darknet21-448-69.tar.gz)
    - [ERFNet - 256px](http://ipb.uni-bonn.de/html/projects/bonnetal/classification/imagenet-erfnet-62.tar.gz)
    - [MobilenetsV2 - 256px](http://ipb.uni-bonn.de/html/projects/bonnetal/classification/imagenet-mobilenetsv2-70.tar.gz)


- Semantic Segmentation:
  - Persons (super fast, [jetson benchmark](train/tasks/segmentation/jetsonbenchmark.md)):
    - [ERFNet - VGA](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/persons-erfnet-512-88.tar.gz)
    - [MobilenetsV2 ASPP - VGA](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/persons-mobilenetsv2-aspp-512-88.tar.gz)
    - [MobilenetsV2 ASPP Res - VGA](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/persons-mobilenetsv2-aspp-res-512-88.tar.gz)
  - Cityscapes:
    - [DarkNet21 ASPP - 512x1024px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/cityscapes_darknet21_aspp_1024_os8_69.tar.gz)
    - [DarkNet53 ASPP - 512x1024px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/cityscapes_darknet53_aspp_1024_os8_74.tar.gz)
    - [DarkNet53 ASPP Residual - 512x1024px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/cityscapes_darknet53_aspp_res_1024_os8_73.tar.gz)
    - [DarkNet53 ASPP - 1024x2048px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/cityscapes_darknet53_aspp_2048_os8_76.tar.gz)
    - [MobilenetsV2 ASPP - 512x1024px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/cityscapes_mobilenetsv2_aspp_1024_os8_70.tar.gz)
    - [MobilenetsV2 ASPP res attention - 512x1024px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/cityscapes_mobilenetsv2_aspp_res_attention_1024_os8_70.tar.gz)
    - [MobilenetsV2 ASPP Residual- 512x1024px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/cityscapes_mobilenetsv2_aspp_res_1024_os8_70.tar.gz)
    - [ERFNet - 512x1024px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/cityscapes_erfnet_1024_70.tar.gz)
    - [ERFNet - 1024x2048px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/cityscapes_erfnet_2048_68.tar.gz)
  - Synthia:
    - [MobilenetsV2 ASPP Res - 512px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/synthia_mobilenetsv2_aspp_res_512_os8_71.tar.gz)
  - Mapillary Vistas:
    - [MobilenetsV2 ASPP Res - 512px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/mapillary_mobilenetsv2_aspp_res_512_os8_34.tar.gz)
    - [DarkNet53 ASPP Res - 512px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/mapillary_darknet53_aspp_res_512_os8_40.tar.gz)
  - Pascal VOC 2012:
    - [MobilenetsV2 ASPP Res - 512px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/voc2012-mobilenetsv2_aspp_res_512_63.tar.gz)
    - [MobilenetsV2 ASPP Res attention - 512px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/voc2012-mobilenetsv2_aspp_res_att_512_63.tar.gz)
  - MS-COCO (Panoptic):
    - [MobilenetsV2 ASPP Res - 512px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/coco-mobilenetsv2_aspp_res_512_34.tar.gz)
    - [MobilenetsV2 ASPP Res att - 512px](http://ipb.uni-bonn.de/html/projects/bonnetal/segmentation/coco-mobilenetsv2_aspp_res_att_512_40.tar.gz)

---
## License

### Bonnetal: MIT

Copyright 2019, Andres Milioto, Cyrill Stachniss. University of Bonn.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Pretrained models: Model and Dataset Dependent

The pretrained models with a specific dataset maintain the copyright of such dataset.

- Imagenet: [Link](http://www.image-net.org/challenges/LSVRC/)
- Synthia: [Link](http://synthia-dataset.net)
- Cityscapes: [Link](https://www.cityscapes-dataset.com)
- Mapillary: [Link](https://blog.mapillary.com/product/2017/05/03/mapillary-vistas-dataset.html)
- Berkeley100k: [Link](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
- ApolloScape: [Link](http://apolloscape.auto/)
- Persons: [Link](https://supervise.ly)
- Coco: [Link](http://cocodataset.org/#home)
- Pascal: [Link](http://host.robots.ox.ac.uk/pascal/VOC/)
- Crop-Weed (CWC): [Link](http://www.ipb.uni-bonn.de/data/sugarbeets2016/)

---
## Citations

If you use our framework for any academic work, please cite the original [paper](http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019icra.pdf).

```
@InProceedings{milioto2019icra,
  author     = {A. Milioto and C. Stachniss},
  title      = {{Bonnet: An Open-Source Training and Deployment Framework for Semantic Segmentation in Robotics using CNNs}},
  booktitle  = {Proc. of the IEEE Intl. Conf. on Robotics \& Automation (ICRA)},
  year       = 2019,
  codeurl    = {https://github.com/Photogrammetry-Robotics-Bonn/bonnet},
  videourl   = {https://www.youtube.com/watch?v=tfeFHCq6YJs},
}
```

If you use our Instance Segmentation code, please cite its paper [paper](http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019icra-fiass.pdf):

```
@InProceedings{milioto2019icra-fiass,
  author     = {A. Milioto and L. Mandtler and C. Stachniss},
  title      = {{Fast Instance and Semantic Segmentation Exploiting Local Connectivity, Metric Learning, and One-Shot Detection for Robotics }},
  booktitle  = {Proc. of the IEEE Intl. Conf. on Robotics \& Automation (ICRA)},
  year       = 2019,
}
```


Our networks are either built directly on top of, or strongly based on, the following architectures, so if you use them for any academic work, please give a look at their papers and cite them if you think proper:

- ResNet: [Link](https://arxiv.org/abs/1512.03385)
- DarkNet: [Link](https://pjreddie.com/darknet/yolo/)
- YoloV3: [Link](https://arxiv.org/abs/1804.02767)
- MobileNetsV2: [Link](https://arxiv.org/abs/1801.04381)
- SegNet: [Link](https://arxiv.org/abs/1511.00561)
- E-Net: [Link](https://arxiv.org/abs/1606.02147)
- ERFNet: [Link](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)
- PSPNet: [Link](https://arxiv.org/abs/1612.01105)
- DeeplabV3: [Link](https://arxiv.org/abs/1802.02611)

---
## Other useful GitHub's:
- [Sync Batchnorm](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch). Allows to train bigger nets in multi-gpu setup with larger batch sizes so that batch norm doesn't diverge to something that doesn't represent the data.
- [Queueing tool](https://github.com/alexanderrichard/queueing-tool): Very nice
queueing tool to share GPU, CPU and Memory resources in a multi-GPU environment.
- [Pytorch](https://github.com/pytorch/pytorch): The backbone of everything.
- [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt): ONNX graph to TensorRT engine for fast inference.
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker): Docker that allows you to also exploit your nvidia GPU.

---
## Internal Contributors (not present in open-source commits)

- Andres Milioto
  - [GitHub](https://github.com/tano297)
  - [University of Bonn](http://www.ipb.uni-bonn.de/people/andres-milioto/)
  - [LinkedIn](https://www.linkedin.com/in/amilioto/)  
  - [ResearchGate](https://www.researchgate.net/profile/Andres_Milioto)
  - [Google Scholar](https://scholar.google.de/citations?user=LzsKE7IAAAAJ&hl=en)

- Ignacio Vizzo
  - [GitHub](https://github.com/nachovizzo)
  - [University of Bonn](http://www.ipb.uni-bonn.de/people/ignacio-vizzo/)
  - [LinkedIn](https://www.linkedin.com/in/vizzoignacio/)
  - [ResearchGate](https://www.researchgate.net/profile/Ignacio_Vizzo)
  - [Google Scholar](https://scholar.google.de/citations?user=nTjF-kkAAAAJ&hl=en)

- Leonard Mandtler
  - [LinkedIn](https://www.linkedin.com/in/leonard-mandtler-00a6a7121/)

- Jens Behley
  - [GitHub](https://github.com/jbehley)
  - [University of Bonn](http://www.ipb.uni-bonn.de/people/jens-behley/)
  - [Google Scholar](https://scholar.google.com/citations?user=L4LZHXsAAAAJ)

- Cyrill Stachniss
  - [University of Bonn](http://www.ipb.uni-bonn.de/people/cyrill-stachniss/)
  - [Google Scholar](https://scholar.google.com/citations?user=8vib2lAAAAAJ)
  - [YouTube](https://www.youtube.com/channel/UCi1TC2fLRvgBQNe-T4dp8Eg)
  
---
## Acknowledgements

This work has partly been supported by the German Research Foundation under Germany's Excellence Strategy, EXC-2070 - 390732324 (PhenoRob).
We also thank NVIDIA Corporation for providing a Quadro P6000 GPU partially used to develop this framework.