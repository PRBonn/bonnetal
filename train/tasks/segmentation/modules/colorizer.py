# This file is covered by the LICENSE file in the root of this project.


import numpy as np


class Colorizer:
  def __init__(self, color_map):
    self.color_map = color_map
    self.n_classes = len(self.color_map)
    self.lut = np.zeros((self.n_classes, 3))
    for key in self.color_map:
      self.lut[key] = self.color_map[key]

  def do(self, argmax):
    # make color image
    color_image = self.lut[argmax]

    return color_image
