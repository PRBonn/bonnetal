#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.

import torch


class iouEval:
  def __init__(self, n_classes, device, ignore=None):
    self.n_classes = n_classes
    self.device = device
    # if ignore is larger than n_classes, consider no ignoreIndex
    self.ignore = torch.tensor(ignore).long()
    self.include = torch.tensor(
        [n for n in range(self.n_classes) if n not in self.ignore]).long()
    print("[IOU EVAL] IGNORE: ", self.ignore)
    print("[IOU EVAL] INCLUDE: ", self.include)
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = torch.zeros(
        (self.n_classes, self.n_classes), device=self.device).long()
    self.ones = None

  def addBatch(self, x, y):  # x=preds, y=targets
    # sizes should be "batch_size x H x W"
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify

    # idxs are labels and predictions
    idxs = torch.stack([x_row, y_row], dim=0)

    # ones is what I want to add to conf when I
    if self.ones is None:
      self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()

    # make confusion matrix (cols = gt, rows = pred)
    self.conf_matrix = self.conf_matrix.index_put_(
        tuple(idxs), self.ones, accumulate=True)

    # print(self.tp.shape)
    # print(self.fp.shape)
    # print(self.fn.shape)

  def getStats(self):
    # remove fp and fn from confusion on the ignore classes cols and rows
    conf = self.conf_matrix.clone().double()
    conf[self.ignore] = 0
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = conf.diag()
    fp = conf.sum(dim=1) - tp
    fn = conf.sum(dim=0) - tp
    return tp, fp, fn

  def getIoU(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection / union
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"


if __name__ == "__main__":
  # mock problem
  nclasses = 2
  ignore = []

  # test with 2 squares and a known IOU
  lbl = torch.zeros((7, 7)).long()
  argmax = torch.zeros((7, 7)).long()

  # put squares
  lbl[2:4, 2:4] = 1
  argmax[3:5, 3:5] = 1

  # make evaluator
  eval = iouEval(nclasses, torch.device('cpu'), ignore)

  # run
  eval.addBatch(argmax, lbl)
  m_iou, iou = eval.getIoU()
  print("IoU: ", m_iou)
  print("IoU class: ", iou)
  m_acc = eval.getacc()
  print("Acc: ", m_acc)
