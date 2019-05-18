#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import __init__ as booger
import collections
import copy
import imp

from tasks.classification.modules.head import *

class Classifier(nn.Module):
  def __init__(self, backbone_cfg, head_cfg, path=None, strict=False):
    super().__init__()
    self.backbone_cfg = backbone_cfg
    self.head_cfg = head_cfg
    self.weights_online = True if path is None else False

    # get the model
    bboneModule = imp.load_source("bboneModule",
                                  booger.TRAIN_PATH + '/backbones/' +
                                  self.backbone_cfg.name + '.py')
    self.backbone = bboneModule.Backbone(input_size=[self.backbone_cfg.h,
                                                     self.backbone_cfg.w,
                                                     self.backbone_cfg.d],
                                         OS=self.backbone_cfg.os,
                                         dropout=self.backbone_cfg.dropout,
                                         bn_d=self.backbone_cfg.bn_d,
                                         extra=self.backbone_cfg.extra,
                                         weights_online=self.weights_online)

    self.head = Head(n_class=self.head_cfg.n_class,
                     feat_h=self.backbone_cfg.h / self.backbone_cfg.os,
                     feat_w=self.backbone_cfg.w / self.backbone_cfg.os,
                     feat_d=self.backbone.get_last_depth(),
                     dropout=self.head_cfg.dropout)

    # get weights
    if path is not None:
      # try backbone
      try:
        w_dict = torch.load(path + "/backbone",
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print()
        print("Couldn't load backbone, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try head
      try:
        w_dict = torch.load(path + "/classification_head",
                            map_location=lambda storage, loc: storage)
        # False strict in case number of classes changed
        self.head.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e
    else:
      print("No path to pretrained, using bonnetal Imagenet backbone weights")

  def forward(self, x):
    x, _ = self.backbone(x)  # second output is skip connections
    x = self.head(x)
    return x
