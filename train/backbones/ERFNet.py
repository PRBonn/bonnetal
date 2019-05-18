# This file is covered by the LICENSE file in the root of this project.


from __future__ import print_function
import torch.nn as nn
import math
import common.layers as lyr
import torch.utils.model_zoo as model_zoo


class Backbone(nn.Module):
  def __init__(self, input_size=[224, 224, 3], OS=8, dropout=0.0, bn_d=0.1, extra=None, weights_online=True):
    super(Backbone, self).__init__()
    self.bn_d = bn_d
    self.dropout = dropout
    self.weights_online = weights_online
    # setting of all encoder layers
    self.layers = [
        # n, c, s, d, type
        [1, 16, 2, 1, "conv-bn-relu"],
        [1, 64, 2, 1, "conv-bn-relu"],
        [5, 64, 1, 1, "non-bt-1d"],
        [1, 128, 2, 1, "conv-bn-relu"],
        [1, 128, 1, 2, "non-bt-1d"],
        [1, 128, 1, 4, "non-bt-1d"],
        [1, 128, 1, 8, "non-bt-1d"],
        [1, 128, 1, 16, "non-bt-1d"],
        [1, 128, 1, 2, "non-bt-1d"],
        [1, 128, 1, 4, "non-bt-1d"],
        [1, 128, 1, 8, "non-bt-1d"],
        [1, 128, 1, 16, "non-bt-1d"],
    ]

    # check current stride
    current_os = 1
    for n, c, s, d, ty in self.layers:
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
        n, c, s, d, ty = layer
        if int(current_os) != OS:
          if s == 2:
            s = 1
            current_os /= 2
            # set the layer
            self.new_layers[-1 - i] = [n, c, s, d, ty]
          if current_os == OS:
            break
      print("New OS: ", current_os)

      # dilations
      self.current_d = 1
      for i, layer in enumerate(self.layers, 0):
        # if new stride is different, up the dilation
        if self.new_layers[i][2] < layer[2]:
          self.current_d *= 2
        self.new_layers[i][3] *= self.current_d

      # check that the output stride mod worked
      assert current_os == OS

    # check input sizes to see if strides will be valid (0=h, 1=w)
    assert input_size[0] % OS == 0 and input_size[1] % OS == 0

    # first and last layer count
    self.last_depth = input_size[2]
    self.last_channel_depth = self.layers[-1][1]

    # building model from inverted residuals
    self.features = nn.ModuleList()

    # build all layers except last
    for n, c, s, d, ty in self.new_layers:
      if ty == "conv-bn-relu":
        self.features.append(lyr.ConvBnRelu(self.last_depth, c,
                                            k=3, stride=s, dilation=d,
                                            bn_d=self.bn_d))
        self.last_depth = c

      elif ty == "non-bt-1d":
        # do all modules in block
        for i in range(n):
          # if it is the first one, do stride
          self.features.append(lyr.non_bottleneck_1d(chann=c,
                                                     dilated=d,
                                                     bn_d=self.bn_d,
                                                     dropprob=self.dropout))

          self.last_depth = c

      else:
        print("unrecognized layer")
        quit()

    # load weights from internet (it is overloaded after by bonnetal if necessary)
    # strict needs to be false because we don't have fc layer in backbone
    if self.weights_online:
      print("Trying to get backbone weights online from Bonnetal server.")
      # only load if images are RGB
      if input_size[2] == 3:
        print("Using pretrained weights from bonnetal server for backbone")
        url = 'http://www.ipb.uni-bonn.de/html/projects/bonnetal/extractors/erfnet/erfnet-87729049.pth'
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
