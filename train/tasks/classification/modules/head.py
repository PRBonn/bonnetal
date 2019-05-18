#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import common.layers as lyr


class HeadConfig():
  def __init__(self, n_class, dropout):
    self.n_class = n_class
    self.dropout = dropout


class Head(nn.Module):
  def __init__(self, n_class, feat_h, feat_w, feat_d, dropout):
    super().__init__()
    self.head = nn.Sequential(nn.AvgPool2d((int(feat_h), int(feat_w))),
                              nn.Dropout(p=dropout),
                              lyr.View((-1, feat_d)),
                              nn.Linear(feat_d, n_class),
                              )

  def forward(self, input):
    return self.head(input)
