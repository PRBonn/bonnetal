#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn


class HeadConfig():
  def __init__(self, n_class, dropout, weights=None):
    self.n_class = n_class
    self.dropout = dropout
    self.weights = weights


class Head(nn.Module):
  def __init__(self, n_class, feat_d, dropout, weights=None):
    super().__init__()
    self.head = nn.Sequential(nn.Dropout2d(p=dropout),
                              nn.Conv2d(feat_d, n_class, 1))
    if weights is not None:
      # using normalized weights as biases
      print("Using normalized weights as bias for head.")
      self.weights = weights / weights.max()
      for m in self.head.modules():
        if isinstance(m, nn.Conv2d):
          m.bias = nn.Parameter(self.weights)

  def forward(self, input):
    return self.head(input)


class DecoderConfig():
  def __init__(self, name, dropout, bn_d, extra=None):
    self.name = name
    self.dropout = dropout
    self.bn_d = bn_d
    self.extra = extra  # specifics of each arch
