# This file is covered by the LICENSE file in the root of this project.

import torch
from common.onehot import to_one_hot


class iouEval:
  def __init__(self, n_classes, device, ignoreIndex=-1):
    self.n_classes = n_classes
    self.device = device
    # if ignoreIndex is larger than n_classes, consider no ignoreIndex
    self.ignoreIndex = ignoreIndex if n_classes > ignoreIndex else -1
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    classes = self.n_classes if self.ignoreIndex == -1 else self.n_classes - 1
    self.tp = torch.zeros(classes).double()
    self.fp = torch.zeros(classes).double()
    self.fn = torch.zeros(classes).double()

  def addBatch(self, x, y):  # x=preds, y=targets
    # sizes should be "batch_size x H x W"

    # scatter to onehot
    x_onehot = to_one_hot(x, self.n_classes).float()
    y_onehot = to_one_hot(y, self.n_classes).float()

    if (self.ignoreIndex != -1):
      ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)
      x_onehot = x_onehot[:, :self.ignoreIndex]
      y_onehot = y_onehot[:, :self.ignoreIndex]
    else:
      ignores = 0

    # print(type(x_onehot))
    # print(type(y_onehot))
    # print(x_onehot.size())
    # print(y_onehot.size())

    tpmult = x_onehot * y_onehot  # times prediction and gt coincide is 1
    tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True),
                             dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
    self.tp += tp.double().cpu()
    del tpmult
    del tp

    # times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
    fpmult = x_onehot * (1 - y_onehot - ignores)
    fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True),
                             dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
    self.fp += fp.double().cpu()
    del fp
    del fpmult

    # times prediction says its not that class and gt says it is
    fnmult = (1 - x_onehot) * (y_onehot)
    fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True),
                             dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
    self.fn += fn.double().cpu()
    del fn
    del fnmult

  def getIoU(self):
    num = self.tp
    den = self.tp + self.fp + self.fn + 1e-15
    iou = num / den
    return torch.mean(iou), iou  # returns "iou mean", "iou per class"

  def getacc(self):
    num = torch.sum(self.tp)
    den = torch.sum(self.tp) + torch.sum(self.fp) + 1e-15
    return num / den  # returns "acc mean"
