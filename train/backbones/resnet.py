# This file was modified from Pytorch's official torchvision models
# It needed to be modified in order to accomodate for different output
# strides (for example, for semantic segmentation)

from __future__ import print_function
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  """3x3 convolution with padding"""
  padding = int((3 + ((dilation - 1) * 2) - 1) / 2)
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=padding, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
  # basic resnet block
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_d=0.1):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride, dilation)
    self.bn1 = nn.BatchNorm2d(planes, momentum=bn_d)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, momentum=bn_d)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  # bottleneck resnet block
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, bn_d=0.1):
    super(Bottleneck, self).__init__()
    self.padding = int((3 + ((dilation - 1) * 2) - 1) / 2)
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes, momentum=bn_d)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=self.padding, dilation=dilation, bias=False)
    self.bn2 = nn.BatchNorm2d(planes, momentum=bn_d)
    self.conv3 = nn.Conv2d(
        planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_d)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


# ******************************************************************************
# weight files
model_urls = {
    'resnet18': 'http://www.ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet18-5c106cde.pth',
    'resnet34': 'http://www.ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet34-333f7ec4.pth',
    'resnet50': 'http://www.ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet50-19c8e357.pth',
    'resnet101': 'http://www.ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet101-5d3b4d8f.pth',
    'resnet152': 'http://www.ipb.uni-bonn.de/html/projects/bonnetal/extractors/resnet/resnet152-b121ed2d.pth',


}

# number of layers per model
model_layers = {
    'resnet18': [2, 2, 2, 2],
    'resnet34': [3, 4, 6, 3],
    'resnet50': [3, 4, 6, 3],
    'resnet101': [3, 4, 23, 3],
    'resnet152': [3, 8, 36, 3],
}

# constitutional block
model_block = {
    'resnet18': BasicBlock,
    'resnet34': BasicBlock,
    'resnet50': Bottleneck,
    'resnet101': Bottleneck,
    'resnet152': Bottleneck,
}


class Backbone(nn.Module):
  def __init__(self, input_size=[224, 224, 3], OS=32, dropout=0.0, bn_d=0.1, extra=None, weights_online=True):
    super(Backbone, self).__init__()
    self.inplanes = 64
    self.dropout_r = dropout
    self.bn_d = bn_d
    self.resnet = extra["resnet"]
    self.OS = OS
    self.weights_online = weights_online

    # check that resnet exists
    assert self.resnet in model_layers.keys()

    # generate layers depending on resnet type
    self.layers = model_layers[self.resnet]
    self.block = model_block[self.resnet]
    self.url = model_urls[self.resnet]
    self.strides = [2, 2, 1, 2, 2, 2]
    self.dilations = [1, 1, 1, 1, 1, 1]
    self.last_depth = input_size[2]
    self.last_channel_depth = 512 * self.block.expansion

    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Original OS: ", current_os)

    if OS > current_os:
      print("Can't do OS, ", OS, " because it is bigger than original ", current_os)
    else:
      # redo strides according to needed stride
      current_dil = int(current_os / OS)
      for i, stride in enumerate(reversed(self.strides), 0):
        self.dilations[-1 - i] *= int(current_dil)
        if int(current_os) != OS:
          if stride == 2:
            current_os /= 2
            current_dil /= 2
            self.strides[-1 - i] = 1
          if int(current_os) == OS:
            break
      print("New OS: ", int(current_os))
      print("Strides: ", self.strides)
      print("Dilations: ", self.dilations)

    # check input sizes to see if strides will be valid (0=h, 1=w)
    assert input_size[0] % OS == 0 and input_size[1] % OS == 0

    # input block
    padding = int((7 + ((self.dilations[0] - 1) * 2) - 1) / 2)
    self.conv1 = nn.Conv2d(self.last_depth, 64, kernel_size=7,
                           stride=self.strides[0], padding=padding, bias=False)
    self.bn1 = nn.BatchNorm2d(64, momentum=self.bn_d)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3,
                                stride=self.strides[1],
                                padding=1)

    # block 1
    self.layer1 = self._make_layer(self.block, 64, self.layers[0], stride=self.strides[2],
                                   dilation=self.dilations[2], bn_d=self.bn_d)

    # block 2
    self.layer2 = self._make_layer(self.block, 128, self.layers[1], stride=self.strides[3],
                                   dilation=self.dilations[3], bn_d=self.bn_d)

    # block 3
    self.layer3 = self._make_layer(self.block, 256, self.layers[2], stride=self.strides[4],
                                   dilation=self.dilations[4], bn_d=self.bn_d)

    # block 4
    self.layer4 = self._make_layer(self.block, 512, self.layers[3], stride=self.strides[5],
                                   dilation=self.dilations[5], bn_d=self.bn_d)

    self.dropout = nn.Dropout2d(self.dropout_r)

    # load weights from internet
    # strict needs to be false because we don't have fc layer in backbone
    if self.weights_online:
      print("Trying to get backbone weights online from Bonnetal server.")
      # only load if images are RGB
      if input_size[2] == 3:
        print("Using pretrained weights from bonnetal server for backbone")
        self.load_state_dict(model_zoo.load_url(self.url,
                                                map_location=lambda storage,
                                                loc: storage),
                            strict=False)
      else:
        print("Can't get bonnetal weights for backbone due to different input depth")

  # make layer useful function
  def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_d=0.1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(planes * block.expansion, momentum=bn_d),
      )

    layers = []
    layers.append(block(self.inplanes, planes,
                        stride, dilation, downsample, bn_d))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, bn_d=bn_d))

    return nn.Sequential(*layers)

  def get_last_depth(self):
    return self.last_channel_depth

  def run_layer(self, x, layer, skips, os):
    y = layer(x)
    if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
      # only equal downsampling is contemplated
      assert(x.shape[2]/y.shape[2] == x.shape[3]/y.shape[3])
      skips[os] = x.detach()
      os *= 2
    x = y
    return x, skips, os

  def forward(self, x):
    # store for skip connections
    skips = {}
    os = 1

    # run cnn
    x, skips, os = self.run_layer(x, self.conv1, skips, os)
    x, skips, os = self.run_layer(x, self.bn1, skips, os)
    x, skips, os = self.run_layer(x, self.relu, skips, os)
    x, skips, os = self.run_layer(x, self.maxpool, skips, os)

    x, skips, os = self.run_layer(x, self.layer1, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.layer2, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.layer3, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.layer4, skips, os)

    # for key in skips.keys():
    #   print(key, skips[key].shape)

    return x, skips
