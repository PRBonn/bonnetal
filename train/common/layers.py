# This file is covered by the LICENSE file in the root of this project.


from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class ConvBnRelu(nn.Module):
  # standard conv, BN, relu
  def __init__(self, inp, oup, k=3, stride=1, dilation=1, pad="same", bn_d=0.1):
    super(ConvBnRelu, self).__init__()
    self.inp = inp
    self.oup = oup
    self.k = k
    self.stride = stride
    self.pad = pad
    self.bn_d = bn_d  # batchnorm decay

    # check pad
    if pad == "same":
      padding = int((k + ((dilation - 1) * 2) - 1) / 2)
    elif pad == "valid":
      padding = 0
    else:
      raise Exception("Invalid padding. Must be same or valid")

    # build layer
    self.conv = nn.Sequential(
        nn.Conv2d(inp, oup, k, stride, padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(oup, momentum=self.bn_d),
        nn.ReLU6(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)


class View(nn.Module):
  # reshape module
  def __init__(self, *shape):
    super(View, self).__init__()
    self.shape = shape

  def forward(self, x):
    return x.view(*self.shape)


class InvertedResidual(nn.Module):
  # inverted residual from mobilenets v2
  def __init__(self, inp, oup, stride=1, dilation=1, expand_ratio=6, dropout=0.0, bn_d=0.1):
    super(InvertedResidual, self).__init__()
    self.stride = stride
    self.bn_d = bn_d  # batchnorm decay
    assert stride in [1, 2]

    # check pad
    pad = int((3 + ((dilation - 1) * 2) - 1) / 2)

    self.use_res_connect = self.stride == 1 and inp == oup

    self.inv_dwise_conv = nn.Sequential(
        # pw
        nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(inp * expand_ratio, momentum=self.bn_d),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3,
                  stride, pad, dilation=dilation,
                  groups=inp * expand_ratio, bias=False),
        nn.BatchNorm2d(inp * expand_ratio, momentum=self.bn_d),
        nn.Dropout2d(p=dropout),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup, momentum=self.bn_d),
    )

  def forward(self, x):
    if self.use_res_connect:
      return x + self.inv_dwise_conv(x)
    else:
      return self.inv_dwise_conv(x)


class ASPP(nn.Module):
  # no global pooling, for robotics it doesn't make much sense
  def __init__(self, feature_depth, input_h, input_w, OS, filters, bn_d=0.1, dropout=0.0):
    super(ASPP, self).__init__()
    self.feature_depth = feature_depth
    self.input_h = input_h
    self.input_w = input_w
    self.OS = OS
    self.filters = filters
    self.dropout = dropout
    self.bn_d = bn_d  # batchnorm decay

    # features
    self.feat_h = int(self.input_h / self.OS)
    self.feat_w = int(self.input_w / self.OS)

    # atrous rates 1 1x1 conv, 3 3x3 convs
    if self.OS == 32:
      self.rates3x3 = [3, 6, 9]
    elif self.OS == 16:
      self.rates3x3 = [6, 12, 18]
    elif self.OS == 8:
      self.rates3x3 = [12, 24, 36]
    else:
      self.rates3x3 = [6, 12, 18]
      print("UNKNOWN RATE FOR OS: ", self.OS, "USING DEFAULT")

    # aspp
    self.aspp_conv_1x1 = ConvBnRelu(self.feature_depth,
                                    self.filters,
                                    k=1, stride=1,
                                    dilation=1,
                                    bn_d=self.bn_d)
    self.aspp_conv_3x3_0 = ConvBnRelu(self.feature_depth,
                                      self.filters,
                                      stride=1,
                                      dilation=self.rates3x3[0],
                                      bn_d=self.bn_d)

    self.aspp_conv_3x3_1 = ConvBnRelu(self.feature_depth,
                                      self.filters,
                                      stride=1,
                                      dilation=self.rates3x3[1],
                                      bn_d=self.bn_d)

    self.aspp_conv_3x3_2 = ConvBnRelu(self.feature_depth,
                                      self.filters,
                                      stride=1,
                                      dilation=self.rates3x3[2],
                                      bn_d=self.bn_d)

    self.aspp_squash = ConvBnRelu(self.filters * 4,
                                  self.filters,
                                  k=1, stride=1,
                                  dilation=1,
                                  bn_d=self.bn_d)

  def forward(self, features):
    # aspp convs and avg pool
    aspp_c1 = self.aspp_conv_1x1(features)
    aspp_c2 = self.aspp_conv_3x3_0(features)
    aspp_c3 = self.aspp_conv_3x3_1(features)
    aspp_c4 = self.aspp_conv_3x3_2(features)

    # concat all
    aspp_concat = torch.cat(
        (aspp_c1, aspp_c2, aspp_c3, aspp_c4), dim=1)

    # squash
    aspp_squash = self.aspp_squash(aspp_concat)

    return aspp_squash

# ERFNet


class non_bottleneck_1d(nn.Module):
  def __init__(self, chann, dilated, bn_d, dropprob):
    super().__init__()

    self.conv3x1_1 = nn.Conv2d(
        chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

    self.conv1x3_1 = nn.Conv2d(
        chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

    self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

    self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
        1 * dilated, 0), bias=True, dilation=(dilated, 1))

    self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
        0, 1 * dilated), bias=True, dilation=(1, dilated))

    self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

    self.dropout = nn.Dropout2d(dropprob)

  def forward(self, input):

    output = self.conv3x1_1(input)
    output = F.relu(output)
    output = self.conv1x3_1(output)
    output = self.bn1(output)
    output = F.relu(output)

    output = self.conv3x1_2(output)
    output = F.relu(output)
    output = self.conv1x3_2(output)
    output = self.bn2(output)

    if (self.dropout.p != 0):
      output = self.dropout(output)

    return F.relu(output + input)  # +input = identity (residual connection)
