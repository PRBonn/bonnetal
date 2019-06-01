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


import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


class UserTensorRT():
  def __init__(self, path, workspace):
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

    # try to deserialize the engine first
    self.engine = None
    self.engine_serialized_path = path + "/model.trt"
    try:
      with open(self.engine_serialized_path, "rb") as f:
        self.runtime = trt.Runtime(TRT_LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(f.read())
    except Exception as e:
      print("Could not deserialize engine. Generate instead. Error: ", e)
      self.engine = None

    # architecture definition from onnx if no engine is there
    # get weights?
    if self.engine is None:
      try:
        # basic stuff for onnx parser
        self.model_path = path + "/model.onnx"
        self.builder = trt.Builder(TRT_LOGGER)
        self.network = self.builder.create_network()
        self.onnxparser = trt.OnnxParser(self.network, TRT_LOGGER)
        self.model = open(self.model_path, 'rb')
        self.onnxparser.parse(self.model.read())
        print("Successfully ONNX weights from ", self.model_path)
        # now add the argmax op in gpu
        self.topk = self.network.add_topk(
            self.network.get_output(0), trt.TopKOperation.MAX, 1, 1)
        assert(self.topk != None)
        self.topk.get_output(1).name = 'out'
        self.network.unmark_output(self.network.get_output(0))
        self.network.mark_output(self.topk.get_output(1))
        print("Successfully added argmax layer")
      except Exception as e:
        print("Couldn't load ONNX network. Error: ", e)
        quit()

      print("Wait while tensorRT profiles the network and build engine")
      # trt parameters
      try:
        self.builder.max_batch_size = 1
        self.builder.max_workspace_size = workspace
        self.builder.fp16_mode = self.builder.platform_has_fast_fp16
        print("Platform has fp16 mode: ", self.builder.platform_has_fast_fp16)
        print("Calling build_cuda_engine")
        self.engine = self.builder.build_cuda_engine(self.network)
        assert(self.engine is not None)
      except Exception as e:
        print("Failed creating engine for TensorRT. Error: ", e)
        quit()
      print("Done generating tensorRT engine.")

      # serialize for later
      print("Serializing tensorRT engine for later (for example in the C++ interface)")
      try:
        self.serialized_engine = self.engine.serialize()
        with open(self.engine_serialized_path, "wb") as f:
          f.write(self.serialized_engine)
      except Exception as e:
        print("Couln't serialize engine. Not critical, so I continue. Error: ", e)
    else:
      print("Successfully opened engine from inference directory.")
      print("WARNING: IF YOU WANT TO PROFILE FOR THIS COMPUTER DELETE model.trt FROM THAT DIRECTORY")

    # create execution context
    self.context = self.engine.create_execution_context()

    # Determine dimensions and create CUDA memory buffers
    # to hold host inputs/outputs.
    self.d_input_size = self.data_h * self.data_w * self.data_d * 4
    self.d_output_size = self.data_h * self.data_w * 4
    # Allocate device memory for inputs and outputs.
    self.d_input = cuda.mem_alloc(self.d_input_size)
    self.d_output = cuda.mem_alloc(self.d_output_size)
    # Create a stream in which to copy inputs/outputs and run inference.
    self.stream = cuda.Stream()
    # try:
    # except Exception as e:
    #   print("Coulnd't allocate i/o for tensorrt inference. Error: ", e)
    #   quit()

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

    # placeholders
    h_input = np.ascontiguousarray(rgb_tensor, dtype=np.float32)
    h_output = np.empty(
        (self.data_h * self.data_w), dtype=np.int32, order='C')

    # infer
    start = time.time()

    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(self.d_input, h_input, self.stream)
    # Run inference.
    self.context.execute_async(bindings=[int(self.d_input), int(
        self.d_output)], stream_handle=self.stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, self.d_output, self.stream)
    # Synchronize the stream
    self.stream.synchronize()

    argmax = h_output.reshape((self.data_h, self.data_w)).astype(np.uint8)
    time_to_infer = time.time() - start

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
