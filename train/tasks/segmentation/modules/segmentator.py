#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp
import torch
import torch.nn as nn
import __init__ as booger

from tasks.segmentation.modules.head import *


class Segmentator(nn.Module):
  def __init__(self, backbone_cfg, decoder_cfg, head_cfg, path=None, path_append="", strict=False):
    super().__init__()
    self.backbone_cfg = backbone_cfg
    self.decoder_cfg = decoder_cfg
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

    # do a pass of the backbone to initialize the skip connections
    stub = torch.zeros((1, 3, self.backbone_cfg.h, self.backbone_cfg.w))
    if torch.cuda.is_available():
      self.backbone.cuda()
      stub = stub.cuda()
    _, skips = self.backbone(stub)

    decoderModule = imp.load_source("decoderModule",
                                    booger.TRAIN_PATH + '/tasks/segmentation/decoders/' +
                                    self.decoder_cfg.name + '.py')
    self.decoder = decoderModule.Decoder(input_size=[self.backbone_cfg.h,
                                                     self.backbone_cfg.w,
                                                     self.backbone_cfg.d],
                                         OS=self.backbone_cfg.os,
                                         feature_depth=self.backbone.get_last_depth(),
                                         dropout=self.decoder_cfg.dropout,
                                         bn_d=self.decoder_cfg.bn_d,
                                         extra=self.decoder_cfg.extra,
                                         skips=skips)

    self.head = Head(n_class=self.head_cfg.n_class,
                     feat_d=self.decoder.get_last_depth(),
                     dropout=self.head_cfg.dropout,
                     weights=self.head_cfg.weights)

    # get weights
    if path is not None:
      # try backbone
      try:
        w_dict = torch.load(path + "/backbone" + path_append,
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print()
        print("Couldn't load backbone, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try decoder
      try:
        w_dict = torch.load(path + "/segmentation_decoder" + path_append,
                            map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model decoder weights")
      except Exception as e:
        print("Couldn't load decoder, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try head
      try:
        w_dict = torch.load(path + "/segmentation_head" + path_append,
                            map_location=lambda storage, loc: storage)
        self.head.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e
    else:
      print("No path to pretrained, using bonnetal Imagenet backbone weights and random decoder.")
      

  def forward(self, x):
    x, skips = self.backbone(x)
    x = self.decoder(x, skips)
    x = self.head(x)
    return x
