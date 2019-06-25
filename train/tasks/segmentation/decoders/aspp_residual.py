# This file is covered by the LICENSE file in the root of this project.

from __future__ import print_function
import torch.nn as nn
import torch
from common.layers import *


class Decoder(nn.Module):
  def __init__(self, input_size=[224, 224, 3], OS=32, feature_depth=1280, dropout=0.0, bn_d=0.1, extra=None, skips=None):
    super(Decoder, self).__init__()

    # parameters
    self.backbone_OS = OS
    self.backbone_input_size = input_size
    self.backbone_feature_depth = feature_depth
    self.dropout = dropout
    self.aspp_channels = extra["aspp_channels"]
    self.skip_os = extra["skip_os"]
    self.last_channels = extra["last_channels"]
    self.bn_d = bn_d  # batchnorm decay

    # os1 can't skip (it is last channel size)
    assert(1 not in self.skip_os)
    for os in self.skip_os:
      assert(os < self.backbone_OS and os > 1)

    # atrous spatial piramid pool (no global pool, not useful for scenes)
    self.ASPP = ASPP(feature_depth=self.backbone_feature_depth,
                     input_h=self.backbone_input_size[0],
                     input_w=self.backbone_input_size[1],
                     OS=self.backbone_OS,
                     filters=self.aspp_channels,
                     dropout=self.dropout,
                     bn_d=self.bn_d)

    # decoder part is somewhat like deeplabv3+ (starts with an ASPP) and then
    # decodes and skips
    # it does one upconv to match skip depth, and then adds residual from skip
    # except for OS=1, because it is usually the input, so do do a trick here
    current_os = self.backbone_OS
    current_depth = self.aspp_channels
    self.upconvs = nn.ModuleList()
    self.mixconvs = nn.ModuleList()
    while current_os > 1:
      current_os //= 2
      skip_depth = skips[current_os].shape[1]
      # only add skip if OS is more than 1, otherwise just upconv
      if current_os > 1:
        next_depth = skip_depth
      else:
        next_depth = self.last_channels
      print("[Decoder] os: ", current_os, "in: ",
            current_depth, "skip:", skip_depth, "out: ", next_depth)
      self.upconvs.append(nn.ConvTranspose2d(current_depth,
                                             next_depth,
                                             kernel_size=2,
                                             stride=2,
                                             padding=0))
      self.mixconvs.append(nn.Sequential(InvertedResidual(next_depth,
                                                          next_depth,
                                                          dropout=dropout,
                                                          bn_d=self.bn_d),
                                         InvertedResidual(next_depth,
                                                          next_depth,
                                                          dropout=dropout,
                                                          bn_d=self.bn_d)))
      current_depth = next_depth

  def forward(self, features, skips):
    features = self.ASPP(features)
    current_os = self.backbone_OS
    # do one upconv and one residual
    for upconv, mixconv in zip(self.upconvs, self.mixconvs):
      current_os //= 2
      # upconv (convolution + upsampling)
      features = upconv(features)  # upsample to match skip
      # add from skip (no grad)
      if current_os in self.skip_os:
        features = features + skips[current_os].detach()

      # do mixing of lower and higher levels
      features = mixconv(features)
    return features

  def get_last_depth(self):
    return self.last_channels
