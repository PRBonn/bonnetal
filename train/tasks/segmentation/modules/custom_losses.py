#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.onehot import *

class mIoULoss(nn.Module):
  def __init__(self, weight):
    super(mIoULoss, self).__init__()
    self.classes = len(weight)
    self.weight = nn.Parameter(weight, requires_grad=False)

  def forward(self, inputs, target):
    # inputs => N x Classes x H x W
    # target => N x H x W

    # target to onehot
    target_oneHot = to_one_hot(target, self.classes)

    # map to (0,1)
    inputs = F.softmax(inputs, dim=1)

    # batch size
    N, C, H, W = inputs.size()

    # Numerator Product
    inter = inputs * target_oneHot
    # Average over all pixels N x C x H x W => N x C
    inter = inter.view(N, self.classes, -1).mean(2) + 1e-8

    # Denominator
    union = inputs + target_oneHot - (inputs * target_oneHot) + 1e-8
    # Average over all pixels N x C x H x W => N x C
    union = union.view(N, self.classes, -1).mean(2)

    # Weights for loss
    frequency = target_oneHot.view(N, self.classes, -1).sum(2).float()
    present = (frequency > 0).float()

    # -log(iou) is a good surrogate for loss
    loss = -torch.log(inter / union) * present * self.weight
    loss = loss.sum(1) / present.sum(1)  # pseudo average

    # Return average loss over batch
    return loss.mean()
