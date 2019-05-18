# This file is covered by the LICENSE file in the root of this project.

from __future__ import print_function
import torch.nn as nn
import torch
from common.layers import *


class Decoder(nn.Module):
  def __init__(self, input_size=[224, 224, 3], OS=32, feature_depth=128, dropout=0.0, bn_d=0.1, extra=None, skips=None):
    super(Decoder, self).__init__()

    # parameters
    self.backbone_OS = OS
    self.backbone_input_size = input_size
    self.backbone_feature_depth = feature_depth
    self.dropout = dropout
    self.os_chan = extra["os_chan"]
    self.skips = extra["skips"]
    self.bn_d = bn_d  # batchnorm decay

    # assert that I am given all the OS channels that I need
    for i in range(0, 5):
      current_os = 2**i
      if current_os < self.backbone_OS:
        try:
          print("OS: ", current_os, ", channels: ", self.os_chan[current_os])
        except:
          print("Can't access OS channels for ERF decoder at OS: ", current_os)
      else:
        break

    # decoder part is like ERFNet, but gives the possibility of doing a concat
    # with skip and a mix
    current_os = self.backbone_OS
    current_depth = self.backbone_feature_depth
    self.upconvs = nn.ModuleList()
    self.mixconvs = nn.ModuleList()
    self.erfconvs = nn.ModuleList()
    while current_os > 1:
      current_os //= 2
      next_depth = self.os_chan[current_os]
      if self.skips:
        skip_depth = skips[current_os].shape[1]
      else:
        skip_depth = 0
      print("[Decoder] os: ", current_os, "in: ",
            current_depth, "skip:", skip_depth, "out: ", next_depth)
      # upconvolutions
      self.upconvs.append(nn.ConvTranspose2d(current_depth,
                                             next_depth,
                                             kernel_size=4,
                                             stride=2,
                                             padding=1))
      # mixers for skips
      if self.skips:
        self.mixconvs.append(ConvBnRelu(next_depth + skip_depth,
                                        next_depth,
                                        k=3,
                                        bn_d=self.bn_d))
      else:
        self.mixconvs.append(nn.Sequential())  # no skip

      # finally necessary mixers
      if current_os > 1:
        self.erfconvs.append(nn.Sequential(non_bottleneck_1d(chann=next_depth,
                                                             dilated=1,
                                                             bn_d=self.bn_d,
                                                             dropprob=self.dropout),
                                           non_bottleneck_1d(chann=next_depth,
                                                             dilated=1,
                                                             bn_d=self.bn_d,
                                                             dropprob=self.dropout)))
      else:
        self.erfconvs.append(nn.Sequential())  # nothing for last layer

      current_depth = next_depth

  def forward(self, features, skips):
    current_os = self.backbone_OS
    # do one upconv and one residual
    for upconv, mixconv, erfconv in zip(self.upconvs, self.mixconvs, self.erfconvs):
      current_os //= 2
      # upconv (convolution + upsampling)
      features = upconv(features)  # upsample to match skip
      # add from skip (no grad)
      if self.skips:
        features = torch.cat([features, skips[current_os].detach()], dim=1)
      # do mixing of lower and higher levels
      features = mixconv(features)  # noop if I did not define anything
      # do further convs
      features = erfconv(features)  # noop if I did not define anything
    return features

  def get_last_depth(self):
    return self.os_chan[1]
