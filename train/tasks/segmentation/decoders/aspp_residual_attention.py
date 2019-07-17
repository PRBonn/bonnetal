# This file is covered by the LICENSE file in the root of this project.

from __future__ import print_function
import torch.nn as nn
import torch
import torch.functional as F
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

    # decoder part runs a matchconv to get to the skip_depth,
    # then upsamples with nearest neighbor, attends the logits with
    # the sigmoid of the
    current_os = self.backbone_OS
    current_depth = self.aspp_channels
    self.mixconvs = nn.ModuleList()
    self.matchconvs = nn.ModuleList()
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
      self.matchconvs.append(InvertedResidual(current_depth,
                                              next_depth,
                                              dropout=dropout,
                                              bn_d=self.bn_d))
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
    for matchconv, mixconv in zip(self.matchconvs, self.mixconvs):
      current_os //= 2
      # upconv (convolution + upsampling)
      features = matchconv(features)
      # upsample
      _, _, H, W = features.shape
      features = F.interpolate(features,
                               size=[int(2 * H), int(2 * W)],
                               mode="nearest")
      # add from skip (no grad)
      if current_os in self.skip_os:
        # this elementwise product with the sigmoid should scale the features
        # so that the ones corresponding to high activations in the encoder
        # are scaled up and the other ones scaled down
        attention = torch.sigmoid(skips[current_os].detach())
        features = features * attention

      # do mixing of lower and higher levels
      features = mixconv(features)
    return features

  def get_last_depth(self):
    return self.last_channels
