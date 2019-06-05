#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
import cv2
import os
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
  def __init__(self, model_path, calibration_files, batch_size, h, w, means, stds):
    trt.IInt8EntropyCalibrator2.__init__(self)
    self.batch_size = batch_size
    self.h = h
    self.w = w
    self.input_size = [h, w]
    self.means = np.array(stds, dtype=np.float32)
    self.stds = np.array(stds, dtype=np.float32)
    assert(isinstance(calibration_files, list))
    self.calib_image_paths = calibration_files
    assert(os.path.exists(model_path))
    self.cache_file = os.path.join(model_path, "model.trt.int8calib")
    self.shape = [self.batch_size, 3] + self.input_size
    self.device_input = cuda.mem_alloc(
        trt.volume(self.shape) * trt.float32.itemsize)
    self.indices = np.arange(len(self.calib_image_paths))
    np.random.shuffle(self.indices)

    def load_batches():
      for i in range(0, len(self.calib_image_paths) - self.batch_size+1, self.batch_size):
        indexs = self.indices[i:i+self.batch_size]
        paths = [self.calib_image_paths[i] for i in indexs]
        files = self.read_batch_file(paths)
        yield files

    self.batches = load_batches()

  def read_batch_file(self, filenames):
    tensors = []
    for filename in filenames:
      assert os.path.exists(filename)
      bgr_img = cv2.imread(filename)
      assert bgr_img.data
      bgr_img = cv2.resize(bgr_img, (self.w, self.h),
                           interpolation=cv2.INTER_LINEAR)
      rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB).astype(np.float32)
      rgb_tensor = (rgb_img / 255.0 - self.means) / self.stds
      rgb_tensor = np.transpose(rgb_tensor, (2, 0, 1))
      rgb_tensor = np.ascontiguousarray(rgb_tensor, dtype=np.float32)
      tensors.append(rgb_tensor)
    return np.ascontiguousarray(tensors, dtype=np.float32)

  def get_batch_size(self):
    return self.batch_size

  def get_batch(self, bindings, names):
    try:
      data = next(self.batches)
      cuda.memcpy_htod(self.device_input, data)
      return [int(self.device_input)]
    except StopIteration:
      return None

  def read_calibration_cache(self):
    if os.path.exists(self.cache_file):
      with open(self.cache_file, 'rb') as f:
        return f.read()

  def write_calibration_cache(self, cache):
    with open(self.cache_file, 'wb') as f:
      f.write(cache)
