# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different output
# strides (for example, for semantic segmentation)

import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo


class BasicBlock(nn.Module):
  def __init__(self, inplanes, planes, bn_d=0.1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                           stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
    self.relu1 = nn.LeakyReLU(0.1)
    self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
    self.relu2 = nn.LeakyReLU(0.1)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)

    out += residual
    return out


# ******************************************************************************
# weight files
model_urls = {
    'darknet21': 'http://www.ipb.uni-bonn.de/html/projects/bonnetal/extractors/darknet/darknet21-4f2301c9.pth',
    'darknet53': 'http://www.ipb.uni-bonn.de/html/projects/bonnetal/extractors/darknet/darknet53-0883870c.pth',
}

# number of layers per model
model_layers = {
    'darknet21': [1, 1, 2, 2, 1],
    'darknet53': [1, 2, 8, 8, 4],
}

# constitutional block
model_block = {
    'darknet21': BasicBlock,
    'darknet53': BasicBlock,
}


class Backbone(nn.Module):
  def __init__(self, input_size=[224, 224, 3], OS=32, dropout=0.0, bn_d=0.1, extra=None, weights_online=True):
    super(Backbone, self).__init__()
    self.inplanes = 32
    self.dropout_r = dropout
    self.bn_d = bn_d
    self.darknet = extra["darknet"]
    self.OS = OS
    self.weights_online = weights_online

    # check that darknet exists
    assert self.darknet in model_layers.keys()

    # generate layers depending on darknet type
    self.layers = model_layers[self.darknet]
    self.block = model_block[self.darknet]
    self.url = model_urls[self.darknet]
    self.strides = [2, 2, 2, 2, 2]
    self.dilations = [1, 1, 1, 1, 1]
    self.last_depth = input_size[2]
    self.last_channel_depth = 1024

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

    # input layer
    self.conv1 = nn.Conv2d(self.last_depth, self.inplanes, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(self.inplanes, momentum=self.bn_d)
    self.relu1 = nn.LeakyReLU(0.1)

    self.layer1 = self._make_layer(self.block, [32, 64], self.layers[0], stride=self.strides[0],
                                   dilation=self.dilations[0], bn_d=self.bn_d)
    self.layer2 = self._make_layer(self.block, [64, 128], self.layers[1], stride=self.strides[1],
                                   dilation=self.dilations[1], bn_d=self.bn_d)
    self.layer3 = self._make_layer(self.block, [128, 256], self.layers[2], stride=self.strides[2],
                                   dilation=self.dilations[2], bn_d=self.bn_d)
    self.layer4 = self._make_layer(self.block, [256, 512], self.layers[3], stride=self.strides[3],
                                   dilation=self.dilations[3], bn_d=self.bn_d)
    self.layer5 = self._make_layer(self.block, [512, 1024], self.layers[4], stride=self.strides[4],
                                   dilation=self.dilations[4], bn_d=self.bn_d)

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.dropout_r)

    # init with He
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

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
    layers = []

    # padding from k and dil
    padding = int((3 + ((dilation - 1) * 2) - 1) / 2)

    #  downsample
    layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                        stride=stride, dilation=dilation,
                                        padding=padding, bias=False)))
    layers.append(("ds_bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
    layers.append(("ds_relu", nn.LeakyReLU(0.1)))

    #  blocks
    self.inplanes = planes[1]
    for i in range(0, blocks):
      layers.append(("residual_{}".format(i),
                     block(self.inplanes, planes, bn_d)))

    return nn.Sequential(OrderedDict(layers))

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
    # first layers
    x, skips, os = self.run_layer(x, self.conv1, skips, os)
    x, skips, os = self.run_layer(x, self.bn1, skips, os)
    x, skips, os = self.run_layer(x, self.relu1, skips, os)

    # all blocks with intermediate dropouts
    x, skips, os = self.run_layer(x, self.layer1, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.layer2, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.layer3, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.layer4, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)
    x, skips, os = self.run_layer(x, self.layer5, skips, os)
    x, skips, os = self.run_layer(x, self.dropout, skips, os)

    # for key in skips.keys():
    #   print(key, skips[key].shape)

    return x, skips
