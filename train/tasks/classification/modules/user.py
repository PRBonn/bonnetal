#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import imp
import yaml
import time
import __init__ as booger
import cv2
import os
import numpy as np

from backbones.config import *
from tasks.classification.modules.head import *
from tasks.classification.modules.classifier import *


class User():
  def __init__(self, path, force_img_prop=(None, None)):
    # parameters
    self.path = path

    # config from path
    try:
      yaml_path = self.path + "/cfg.yaml"
      print("Opening config file %s" % yaml_path)
      self.CFG = yaml.load(open(yaml_path, 'r'))
    except Exception as e:
      print(e)
      print("Error opening cfg.yaml file from trained model.")
      quit()

    # if force img prop is a tuple with 2 elements, force image props
    if force_img_prop[0] is not None and force_img_prop[1] is not None:
      self.CFG["dataset"]["img_prop"]["height"] = force_img_prop[0]
      self.CFG["dataset"]["img_prop"]["width"] = force_img_prop[1]
      print("WARNING: FORCING IMAGE PROPERTIES TO")
      print(self.CFG["dataset"]["img_prop"])

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/classification/dataset/' +
                                   self.CFG["dataset"]["name"] + '/parser.py')
    self.parser = parserModule.Parser(img_prop=self.CFG["dataset"]["img_prop"],
                                      img_means=self.CFG["dataset"]["img_means"],
                                      img_stds=self.CFG["dataset"]["img_stds"],
                                      classes=self.CFG["dataset"]["labels"],
                                      train=False)

    self.data_h, self.data_w, self.data_d = self.parser.get_img_size()
    self.means, self.stds = self.parser.get_means_stds()
    self.means = torch.tensor(self.means)
    self.stds = torch.tensor(self.stds)

    # get architecture and build backbone (with pretrained weights)
    self.bbone_cfg = BackboneConfig(name=self.CFG["backbone"]["name"],
                                    os=self.CFG["backbone"]["OS"],
                                    h=self.data_h,
                                    w=self.data_w,
                                    d=self.data_d,
                                    dropout=self.CFG["backbone"]["dropout"],
                                    bn_d=self.CFG["backbone"]["bn_d"],
                                    extra=self.CFG["backbone"]["extra"])

    self.head_cfg = HeadConfig(n_class=self.parser.get_n_classes(),
                               dropout=self.CFG["head"]["dropout"])

    # concatenate the encoder and the head
    self.model = Classifier(self.bbone_cfg,
                            self.head_cfg,
                            self.path,
                            strict=True)

    # don't train
    self.model.eval()
    for w in self.model.backbone.parameters():
      w.requires_grad = False
    for w in self.model.head.parameters():
      w.requires_grad = False

    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel()
                        for p in self.model.parameters())
    weights_grad = sum(p.numel()
                       for p in self.model.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # GPU?
    self.gpu = False
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inferring in device: ", self.device)
    if torch.cuda.is_available():
      self.gpu = True
      cudnn.benchmark = True
      cudnn.fastest = True
      self.model.cuda()
      self.means = self.means.cuda()
      self.stds = self.stds.cuda()

  def infer(self, bgr_img, topk=1, verbose=True):
    # get sizes
    original_h, original_w, original_d = bgr_img.shape

    # resize
    bgr_img = cv2.resize(bgr_img, (self.data_w, self.data_h),
                         interpolation=cv2.INTER_LINEAR)

    # check if network is RGB or mono
    if self.CFG["dataset"]["img_prop"]["depth"] == 3:
      # get make rgb
      if verbose:
        print("Converting bgr to rgb")
      rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    elif self.CFG["dataset"]["img_prop"]["depth"] == 1:
      # get grayscale
      if verbose:
        print("Converting to grayscale")
      rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    else:
      raise NotImplementedError(
          "Network has to have 1 or 3 channels. Anything else must be implemented.")

    # to tensor
    rgb_tensor = torch.from_numpy(rgb_img)

    # to gpu
    if self.gpu:
      rgb_tensor = rgb_tensor.cuda()

    # permute and normalize
    rgb_tensor = (rgb_tensor.float() / 255.0 - self.means) / self.stds
    rgb_tensor = rgb_tensor.permute(2, 0, 1)

    # add batch dimension
    rgb_tensor = rgb_tensor.unsqueeze(0)

    # gpu?
    with torch.no_grad():
      start = time.time()
      # infer
      logits = self.model(rgb_tensor)
      max_k = logits[0].topk(topk, dim=-1, largest=True, sorted=True)
      if self.gpu:
        torch.cuda.synchronize()
      time_to_infer = time.time() - start

      # print result, and put in output lists
      classes = []
      classes_str = []
      if verbose:
        print("Time to infer: {:.3f}s".format(time_to_infer))
      for i, idx in enumerate(max_k[1], 1):
        idx_cpu = idx.cpu().item()
        class_string = self.parser.get_class_string(idx_cpu)
        classes.append(idx_cpu)
        classes_str.append(class_string)
        if verbose:
          print("[{}]: {}, {}".format(i, idx_cpu, class_string))

    return classes, classes_str
