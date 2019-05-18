# This file is covered by the LICENSE file in the root of this project.


class BackboneConfig():
  def __init__(self, name, os, h, w, d, dropout, bn_d, extra=None):
    self.name = name
    self.os = os
    self.h = h
    self.w = w
    self.d = d
    self.dropout = dropout
    self.bn_d = bn_d
    self.extra = extra  # extra cnn-specific config
