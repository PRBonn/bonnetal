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
from tasks.segmentation.modules.head import *
from tasks.segmentation.modules.segmentator import *
from tasks.segmentation.modules.colorizer import *

import caffe2.python.onnx.backend as backend
import onnx


class UserCaffe2():
  def __init__(self, path):
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

    # make a colorizer
    self.colorizer = Colorizer(self.CFG["dataset"]["color_map"])

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/segmentation/dataset/' +
                                   self.CFG["dataset"]["name"] + '/parser.py')
    self.parser = parserModule.Parser(img_prop=self.CFG["dataset"]["img_prop"],
                                      img_means=self.CFG["dataset"]["img_means"],
                                      img_stds=self.CFG["dataset"]["img_stds"],
                                      classes=self.CFG["dataset"]["labels"],
                                      train=False)

    # some useful data
    self.data_h, self.data_w, self.data_d = self.parser.get_img_size()
    self.means, self.stds = self.parser.get_means_stds()
    self.means = np.array(self.means, dtype=np.float32)
    self.stds = np.array(self.stds, dtype=np.float32)
    self.nclasses = self.parser.get_n_classes()

    # architecture definition
    # get weights?
    try:
      self.onnx_path = os.path.join(self.path, "model.onnx")
      self.model = onnx.load(self.onnx_path)
      print("Successfully ONNX weights from ", self.onnx_path)
    except Exception as e:
      print("Couldn't load ONNX network. Error: ", e)
      quit()

    # prepare caffe2 model in proper device
    if torch.cuda.is_available():
      self.device = "CUDA"
    else:
      self.device = "CPU"
    print("Building backend ONXX Caffe2 with device ", self.device)
    self.engine = backend.prepare(self.model, device=self.device)

  def infer(self, bgr_img, verbose=True, color=True):
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

    # permute and normalize
    rgb_tensor = (rgb_img.astype(np.float32) /
                  255.0 - self.means) / self.stds
    rgb_tensor = np.transpose(rgb_tensor, (2, 0, 1))

    # add batch dimension
    rgb_tensor = rgb_tensor[np.newaxis, ...]

    # dictionary to feed to caffe2 onnx inference model
    w = {self.model.graph.input[0].name: rgb_tensor}

    # infer
    start = time.time()
    logits = self.engine.run(w)[0]
    time_to_infer = time.time() - start
    argmax = logits[0].argmax(axis=0).astype(np.uint8)

    # time
    if verbose:
      print("Time to infer: {:.3f}s".format(time_to_infer))

    # resize to original size
    argmax = cv2.resize(argmax, (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST)

    # color (if I don't want it, just return original image)
    color_mask = bgr_img
    if color:
      color_mask = self.colorizer.do(argmax).astype(np.uint8)

    return argmax, color_mask
