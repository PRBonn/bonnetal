#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import numpy as np

from common.logger import Logger
from backbones.config import *
from common.avgmeter import *
from common.sync_batchnorm.batchnorm import convert_model
from common.oneshot import OneShot_LR
from tasks.classification.modules.classifier import *
from tasks.classification.modules.head import *


class Trainer():
  def __init__(self, config, logdir, path=None, only_eval=False, block_bn=False):
    # parameters
    self.CFG = config
    self.log = logdir
    self.path = path
    self.only_eval = only_eval
    self.block_bn = block_bn

    # put logger where it belongs
    self.tb_logger = Logger(self.log + "/tb")
    self.info = {"train_update": 0,
                 "train_loss": 0,
                 "train_top_1": 0,
                 "train_top_5": 0,
                 "valid_loss": 0,
                 "valid_top_1": 0,
                 "valid_top_5": 0,
                 "valid_loss_avg_models": 0,
                 "valid_top_1_avg_models": 0,
                 "valid_top_5_avg_models": 0,
                 "feat_lr": 0,
                 "head_lr": 0}

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/classification/dataset/' +
                                   self.CFG["dataset"]["name"] + '/parser.py')
    self.parser = parserModule.Parser(img_prop=self.CFG["dataset"]["img_prop"],
                                      img_means=self.CFG["dataset"]["img_means"],
                                      img_stds=self.CFG["dataset"]["img_stds"],
                                      classes=self.CFG["dataset"]["labels"],
                                      train=True,
                                      location=self.CFG["dataset"]["location"],
                                      batch_size=self.CFG["train"]["batch_size"],
                                      workers=self.CFG["dataset"]["workers"])

    self.data_h, self.data_w, self.data_d = self.parser.get_img_size()

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
    self.model = Classifier(self.bbone_cfg, self.head_cfg, self.path)

    # train backbone?
    if not self.CFG["backbone"]["train"]:
      for w in self.model.backbone.parameters():
        w.requires_grad = False

    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel()
                        for p in self.model.parameters())
    weights_grad = sum(p.numel()
                       for p in self.model.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)
    # breakdown by layer
    weights_enc = sum(p.numel()
                      for p in self.model.backbone.parameters())
    weights_head = sum(p.numel()
                       for p in self.model.head.parameters())
    print("Param encoder ", weights_enc)
    print("Param head ", weights_head)

    # GPU?
    self.gpu = False
    self.multi_gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      self.gpu = True
      # cudnn.benchmark = True
      self.model.cuda()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      self.model = nn.DataParallel(self.model)  # spread in gpus
      self.model = convert_model(self.model).cuda()  # sync batchnorm
      self.model_single = self.model.module  # single model to get weight names
      self.multi_gpu = True

    # weights for loss
    self.loss_w = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
    for idx, w in self.CFG["dataset"]["labels_w"].items():
      self.loss_w[idx] = torch.tensor(w)

    # loss
    self.criterion = nn.CrossEntropyLoss(weight=self.loss_w).to(self.device)

    # optimizer
    train_dicts = [{'params': self.model_single.head.parameters()}]
    if self.CFG["backbone"]["train"]:
      train_dicts.append({'params': self.model_single.backbone.parameters()})

    # Use SGD optimizer to train
    self.optimizer = optim.SGD(train_dicts,
                               lr=self.CFG["train"]["max_lr"],
                               momentum=self.CFG["train"]["min_momentum"],
                               weight_decay=self.CFG["train"]["w_decay"])

    # Use one shot learning rate
    # post decay and step sizes come in epochs and we want it in steps
    steps_per_epoch = self.parser.get_train_size()
    up_steps = int(self.CFG["train"]["up_epochs"] * steps_per_epoch)
    down_steps = int(self.CFG["train"]["down_epochs"] * steps_per_epoch)
    final_decay = self.CFG["train"]["final_decay"] ** (1/steps_per_epoch)

    self.scheduler = OneShot_LR(self.optimizer,
                                base_lr=self.CFG["train"]["min_lr"],
                                max_lr=self.CFG["train"]["max_lr"],
                                step_size_up=up_steps,
                                step_size_down=down_steps,
                                cycle_momentum=True,
                                base_momentum=self.CFG["train"]["min_momentum"],
                                max_momentum=self.CFG["train"]["max_momentum"],
                                post_decay=final_decay)

    # buffer to save the best N models
    self.best_n_models = self.CFG["train"]["avg_N"]
    self.best_backbones = collections.deque(maxlen=self.best_n_models)
    self.best_heads = collections.deque(maxlen=self.best_n_models)

  def save_checkpoint(self, bbone, head, suffix=""):
    # Save the weights
    torch.save(bbone, self.log + "/backbone" + suffix)
    torch.save(head, self.log + "/classification_head" + suffix)

  def save_to_log(self, logger, info, epoch, summary=False):
    # save scalars
    for tag, value in self.info.items():
      logger.scalar_summary(tag, value, epoch)

    # save summaries of weights and biases
    if summary:
      for tag, value in self.model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
        if value.grad is not None:
          logger.histo_summary(
              tag + '/grad', value.grad.data.cpu().numpy(), epoch)

  def train(self):
    best_prec1 = 0.0

    if self.only_eval:
      print("*" * 80)
      print("Only evaluation, no training is being used")
      self.validate(val_loader=self.parser.get_valid_set(),
                    model=self.model,
                    criterion=self.criterion)
      print("*" * 80)
      return

    # train for n epochs
    for epoch in range(self.CFG["train"]["max_epochs"]):
      # get info for learn rate currently
      groups = self.optimizer.param_groups
      if len(groups) == 2:
        self.info["feat_lr"] = groups[0]['lr']
        self.info["head_lr"] = groups[1]['lr']
      elif len(groups) == 1:
        self.info["feat_lr"] = 0
        self.info["head_lr"] = groups[0]['lr']
      else:
        print("Invalid learning rate groups optimizer")

      # train for 1 epoch
      prec1, prec5, loss, update_mean = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                         model=self.model,
                                                         criterion=self.criterion,
                                                         optimizer=self.optimizer,
                                                         epoch=epoch,
                                                         block_bn=self.block_bn,
                                                         scheduler=self.scheduler)

      # update info
      self.info["train_update"] = update_mean
      self.info["train_loss"] = loss
      self.info["train_top_1"] = prec1
      self.info["train_top_5"] = prec5

      # evaluate on validation set
      print("*" * 80)
      prec1, prec5, loss, = self.validate(val_loader=self.parser.get_valid_set(),
                                          model=self.model,
                                          criterion=self.criterion)
      print("*" * 80)

      # update info
      self.info["valid_loss"] = loss
      self.info["valid_top_1"] = prec1
      self.info["valid_top_5"] = prec5

      # remember best prec@1 and save checkpoint
      if prec1 > best_prec1:
        print("Best top1 acc so far, save model!")
        best_prec1 = prec1

        # now average the models and evaluate again
        print("Averaging the best {0} models".format(
            self.best_n_models))

        # append current backbone to its circular buffer
        current_backbone = self.model_single.backbone.state_dict()
        avg_backbone = copy.deepcopy(
            self.model_single.backbone).cpu().state_dict()
        self.best_backbones.append(copy.deepcopy(
            self.model_single.backbone).cpu().state_dict())

        # now average the backbone
        for i, backbone in enumerate(self.best_backbones):
          # for each weight key
          for key, val in backbone.items():
            # if it is the first time, zero the entry first
            if i == 0:
              avg_backbone[key].data.zero_()
            # then sum the avg contribution
            avg_backbone[key] += backbone[key] / \
                float(len(self.best_backbones))

        # append current head to its circular buffer
        current_head = self.model_single.head.state_dict()
        avg_head = copy.deepcopy(self.model_single.head).cpu().state_dict()
        self.best_heads.append(copy.deepcopy(
            self.model_single.head).cpu().state_dict())

        # now average the head
        for i, head in enumerate(self.best_heads):
          print
          # for each weight key
          for key, val in head.items():
            # if it is the first time, zero the entry first
            if i == 0:
              avg_head[key].data.zero_()
            # then sum the avg contribution
            avg_head[key] += head[key] / float(len(self.best_heads))

        # put averaged weights in dictionary and evaluate again
        self.model_single.backbone.load_state_dict(avg_backbone)
        self.model_single.head.load_state_dict(avg_head)

        # evaluate on validation set
        prec1, prec5, loss, = self.validate(val_loader=self.parser.get_valid_set(),
                                            model=self.model,
                                            criterion=self.criterion)
        print("*" * 80)

        # update info
        self.info["valid_loss_avg_models"] = loss
        self.info["valid_top_1_avg_models"] = prec1
        self.info["valid_top_5_avg_models"] = prec5

        # restore the current weights into model
        self.model_single.backbone.load_state_dict(current_backbone)
        self.model_single.head.load_state_dict(current_head)

        # save the weights!
        self.save_checkpoint(current_backbone, current_head, suffix="_single")
        self.save_checkpoint(avg_backbone, avg_head, suffix="")

      # save to log
      self.save_to_log(self.tb_logger, self.info, epoch,
                       self.CFG["train"]["save_summary"])

    print('Finished Training')

    return

  def train_epoch(self, train_loader, model, criterion, optimizer, epoch, block_bn, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    update_ratio_meter = AverageMeter()

    # switch to train mode
    model.train()

    # switch batchnorm to eval mode if I want to block rolling averages
    if block_bn:
      for m in model.modules():
        if isinstance(m, nn.modules.BatchNorm1d):
          m.eval()
        if isinstance(m, nn.modules.BatchNorm2d):
          m.eval()
        if isinstance(m, nn.modules.BatchNorm3d):
          m.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
      # measure data loading time
      data_time.update(time.time() - end)
      if not self.multi_gpu and self.gpu:
        input = input.cuda()
      if self.gpu:
        target = target.cuda(non_blocking=True)

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      prec1, prec5 = self.accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), input.size(0))
      top1.update(prec1[0], input.size(0))
      top5.update(prec5[0], input.size(0))

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      # get gradient updates and weights, so I can print the relationship of
      # their norms
      lr = self.optimizer.param_groups[0]["lr"]
      update_ratios = []
      for _, value in self.model.named_parameters():
        if value.grad is not None:
          w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
          update = np.linalg.norm(-lr * value.grad.cpu().numpy().reshape((-1)))
          update_ratios.append(update / w)
      update_ratios = np.array(update_ratios)
      update_mean = update_ratios.mean()
      update_std = update_ratios.std()
      update_ratio_meter.update(update_mean)  # over the epoch

      if i % self.CFG["train"]["report_batch"] == 0:
        print('Lr: {lr:.3e} | '
              'Update: {umean:.3e} mean,{ustd:.3e} std | '
              'Epoch: [{0}][{1}/{2}] | '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
              'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
              'Loss {loss.val:.4f} ({loss.avg:.4f}) | '
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) | '
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr,
                  umean=update_mean, ustd=update_std))

      # step scheduler
      scheduler.step()

    return top1.avg, top5.avg, losses.avg, update_ratio_meter.avg

  def validate(self, val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
      end = time.time()
      for i, (input, target) in enumerate(val_loader):
        if not self.multi_gpu and self.gpu:
          input = input.cuda()
        if self.gpu:
          target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = self.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

      print('Validation set:\n'
            'Time avg per batch {batch_time.avg:.3f}\n'
            'Loss avg {loss.avg:.4f}\n'
            'Prec@1 avg {top1.avg:.3f}\n'
            'Prec@5 avg {top5.avg:.3f}'.format(batch_time=batch_time,
                                               loss=losses,
                                               top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

  def accuracy(self, output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
      maxk = max(topk)
      batch_size = target.size(0)

      _, pred = output.topk(maxk, 1, True, True)
      pred = pred.t()
      correct = pred.eq(target.view(1, -1).expand_as(pred))

      res = []
      for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
      return res
