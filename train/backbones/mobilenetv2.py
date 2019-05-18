# This file is covered by the LICENSE file in the root of this project.


from __future__ import print_function
import torch.nn as nn
import math
import common.layers as lyr
import torch.utils.model_zoo as model_zoo


class Backbone(nn.Module):
  def __init__(self, input_size=[224, 224, 3], OS=32, dropout=0.0, bn_d=0.1, extra=None, weights_online=True):
    super(Backbone, self).__init__()
    self.width_mult = extra["width_mult"]
    self.shallow_feats = extra["shallow_feats"]
    self.bn_d = bn_d
    self.weights_online = weights_online
    # setting of inverted residual blocks
    self.layers = [
        # t, c, n, s, d, type
        [1, 32, 1, 2, 1, "input"],
        [1, 16, 1, 1, 1, "inv_res"],
        [6, 24, 2, 2, 1, "inv_res"],
        [6, 32, 3, 2, 1, "inv_res"],
        [6, 64, 4, 2, 1, "inv_res"],
        [6, 96, 3, 1, 1, "inv_res"],
        [6, 160, 3, 2, 1, "inv_res"],
        [6, 320, 1, 1, 1, "inv_res"]
    ]

    # check current stride
    current_os = 1
    for t, c, n, s, d, ty in self.layers:
      current_os *= s
    print("Original OS: ", current_os)

    if OS > current_os:
      print("Can't do OS, ", OS, " because it is bigger than original ", current_os)
      self.new_layers = self.layers
    else:
      # redo strides and dilations according to needed stride
      # strides
      self.new_layers = self.layers.copy()
      for i, layer in enumerate(reversed(self.layers), 0):
        t, c, n, s, d, ty = layer
        if int(current_os) != OS:
          if s == 2:
            s = 1
            current_os /= 2
            # set the layer
            self.new_layers[-1 - i] = [t, c, n, s, d, ty]
          if current_os == OS:
            break
      print("New OS: ", current_os)

      # dilations
      self.current_d = 1
      for i, layer in enumerate(self.layers, 0):
        # if new stride is different, up the dilation
        if self.new_layers[i][3] < layer[3]:
          self.current_d *= 2
        self.new_layers[i][4] *= self.current_d

      # check that the output stride mod worked
      assert current_os == OS

    # check input sizes to see if strides will be valid (0=h, 1=w)
    assert input_size[0] % OS == 0 and input_size[1] % OS == 0

    # first and last layer count
    self.last_depth = input_size[2]
    if self.shallow_feats:
      # c component of last layer
      self.last_channel_depth = int(self.new_layers[-1][1] * self.width_mult)
    else:
      self.last_channel_depth = int(
          1280 * self.width_mult) if self.width_mult > 1.0 else 1280

    # building model from inverted residuals
    self.features = []

    # build all layers except last
    for t, c, n, s, d, ty in self.new_layers:
      if ty == "input":
        next_depth = int(c * self.width_mult)
        self.features.append(lyr.ConvBnRelu(self.last_depth, next_depth,
                                            k=3, stride=s, dilation=d,
                                            bn_d=self.bn_d))
        self.last_depth = next_depth

      elif ty == "inv_res":
        next_depth = int(c * self.width_mult)
        # do all modules in block
        for i in range(n):
          # if it is the first one, do stride
          if i == 0:
            self.features.append(lyr.InvertedResidual(
                self.last_depth, next_depth, s, d, t, dropout, self.bn_d))
          else:
            self.features.append(lyr.InvertedResidual(
                self.last_depth, next_depth, 1, d, t, dropout, self.bn_d))
          self.last_depth = next_depth
      else:
        print("unrecognized layer")
        quit()

    # building last layer to squash to final depth
    if not self.shallow_feats:
      self.features.append(lyr.ConvBnRelu(self.last_depth,
                                          self.last_channel_depth,
                                          k=1, stride=1,
                                          bn_d=self.bn_d))

    # make it nn.Sequential
    self.features = nn.ModuleList(self.features)

    # init with He
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    # load weights from internet (it is overloaded after by bonnetal if necessary)
    # strict needs to be false because we don't have fc layer in backbone
    if self.weights_online:
      print("Trying to get backbone weights online from Bonnetal server.")
      # only load if images are RGB
      if input_size[2] == 3:
        print("Using pretrained weights from bonnetal server for backbone")
        url = 'http://www.ipb.uni-bonn.de/html/projects/bonnetal/extractors/mobilenetv2/mobilenetsv2-ee4468eb.pth'
        self.load_state_dict(model_zoo.load_url(url,
                                                map_location=lambda storage,
                                                loc: storage),
                             strict=False)
      else:
        print("Can't get bonnetal weights for backbone due to different input depth")

  def forward(self, x):
    # store for skip connections
    skips = {}
    # run cnn
    current_os = 1
    for layer in self.features:
      y = layer(x)
      if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
        # only equal downsampling is contemplated
        assert(x.shape[2]/y.shape[2] == x.shape[3]/y.shape[3])
        skips[current_os] = x.detach()
        current_os *= 2
      x = y

    # for key in skips.keys():
    #   print(key, skips[key].shape)

    return x, skips

  def get_last_depth(self):
    return self.last_channel_depth
