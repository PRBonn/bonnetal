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
import cv2
import os
import numpy as np

from common.logger import Logger
from backbones.config import *
from common.avgmeter import *
from common.sync_batchnorm.batchnorm import convert_model
from common.oneshot import OneShot_LR
from tasks.segmentation.modules.head import *
from tasks.segmentation.modules.segmentator import *
from tasks.segmentation.modules.colorizer import *
from tasks.segmentation.modules.ioueval import *


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
                 "train_acc": 0,
                 "train_iou": 0,
                 "valid_loss": 0,
                 "valid_acc": 0,
                 "valid_iou": 0,
                 "valid_loss_avg_models": 0,
                 "valid_acc_avg_models": 0,
                 "valid_iou_avg_models": 0,
                 "feat_lr": 0,
                 "decoder_lr": 0,
                 "head_lr": 0}

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/segmentation/dataset/' +
                                   self.CFG["dataset"]["name"] + '/parser.py')
    self.parser = parserModule.Parser(img_prop=self.CFG["dataset"]["img_prop"],
                                      img_means=self.CFG["dataset"]["img_means"],
                                      img_stds=self.CFG["dataset"]["img_stds"],
                                      classes=self.CFG["dataset"]["labels"],
                                      train=True,
                                      location=self.CFG["dataset"]["location"],
                                      batch_size=self.CFG["train"]["batch_size"],
                                      crop_prop=self.CFG["train"]["crop_prop"],
                                      workers=self.CFG["dataset"]["workers"])

    self.data_h, self.data_w, self.data_d = self.parser.get_img_size()

    # weights for loss (and bias)
    self.loss_w = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
    for idx, w in self.CFG["dataset"]["labels_w"].items():
      self.loss_w[idx] = torch.tensor(w)

    # get architecture and build backbone (with pretrained weights)
    self.bbone_cfg = BackboneConfig(name=self.CFG["backbone"]["name"],
                                    os=self.CFG["backbone"]["OS"],
                                    h=self.data_h,
                                    w=self.data_w,
                                    d=self.data_d,
                                    dropout=self.CFG["backbone"]["dropout"],
                                    bn_d=self.CFG["backbone"]["bn_d"],
                                    extra=self.CFG["backbone"]["extra"])

    self.decoder_cfg = DecoderConfig(name=self.CFG["decoder"]["name"],
                                     dropout=self.CFG["decoder"]["dropout"],
                                     bn_d=self.CFG["decoder"]["bn_d"],
                                     extra=self.CFG["decoder"]["extra"])

    self.head_cfg = HeadConfig(n_class=self.parser.get_n_classes(),
                               dropout=self.CFG["head"]["dropout"],
                               weights=self.loss_w)

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.bbone_cfg,
                               self.decoder_cfg,
                               self.head_cfg,
                               self.path)

    # train backbone?
    if not self.CFG["backbone"]["train"]:
      self.CFG["backbone"]["train"] = False
      for w in self.model.backbone.parameters():
        w.requires_grad = False

    # train decoder?
    if not self.CFG["decoder"]["train"]:
      self.CFG["decoder"]["train"] = False
      for w in self.model.decoder.parameters():
        w.requires_grad = False

    # print number of parameters and the ones requiring gradients
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
    weights_dec = sum(p.numel()
                      for p in self.model.decoder.parameters())
    weights_head = sum(p.numel()
                       for p in self.model.head.parameters())
    print("Param encoder ", weights_enc)
    print("Param decoder ", weights_dec)
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
      self.model = nn.DataParallel(self.model)   # spread in gpus
      self.model = convert_model(self.model).cuda()  # sync batchnorm
      self.model_single = self.model.module  # single model to get weight names
      self.multi_gpu = True

    # loss
    if "loss" in self.CFG["train"].keys() and self.CFG["train"]["loss"] == "xentropy":
      self.criterion = nn.CrossEntropyLoss(weight=self.loss_w).to(self.device)
    elif "loss" in self.CFG["train"].keys() and self.CFG["train"]["loss"] == "iou":
      self.criterion = mIoULoss(weight=self.loss_w).to(self.device)
    else:
      raise Exception('Loss not defined in config file')

    # optimizer
    train_dicts = [{'params': self.model_single.head.parameters()}]
    if self.CFG["backbone"]["train"]:
      train_dicts.append({'params': self.model_single.backbone.parameters()})
    if self.CFG["decoder"]["train"]:
      train_dicts.append({'params': self.model_single.decoder.parameters()})

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
    self.best_decoders = collections.deque(maxlen=self.best_n_models)
    self.best_heads = collections.deque(maxlen=self.best_n_models)

  def save_checkpoint(self, bbone, decoder, head, suffix=""):
    # Save the weights
    torch.save(bbone, self.log + "/backbone" + suffix)
    torch.save(decoder, self.log + "/segmentation_decoder" + suffix)
    torch.save(head, self.log + "/segmentation_head" + suffix)

  def save_to_log(self, logdir, logger, info, epoch, w_summary=False, rand_imgs=[], img_summary=False):
    # save scalars
    for tag, value in info.items():
      logger.scalar_summary(tag, value, epoch)

    # save summaries of weights and biases
    if w_summary:
      for tag, value in self.model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
        if value.grad is not None:
          logger.histo_summary(
              tag + '/grad', value.grad.data.cpu().numpy(), epoch)

    if img_summary:
      directory = os.path.join(logdir, "predictions")
      if not os.path.isdir(directory):
        os.makedirs(directory)
      for i, img in enumerate(rand_imgs):
        name = os.path.join(directory, str(i) + ".png")
        cv2.imwrite(name, img)

  def train(self):
    # accuracy and IoU stuff
    best_train_iou = 0.0
    best_val_iou = 0.0

    self.ignore_class = -1
    for i, w in enumerate(self.loss_w):
      if w < 1e-10:
        self.ignore_class = i
        print("Ignoring class ", i, " in IoU evaluation")
    self.evaluator = iouEval(self.parser.get_n_classes(),
                             self.device, self.ignore_class)

    # image colorizer
    self.colorizer = Colorizer(self.CFG["dataset"]["color_map"])

    # check if I only want to evaluate
    if self.only_eval:
      print("*" * 80)
      print("Only evaluation, no training is being used")
      acc, iou, loss, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                               model=self.model,
                                               criterion=self.criterion,
                                               evaluator=self.evaluator,
                                               save_images=self.CFG["train"]["save_imgs"],
                                               class_dict=self.CFG["dataset"]["labels"])
      # colorize images
      for i, img in enumerate(rand_img[:]):
        rand_img[i] = self.colorizer.do(img)

      # update info
      self.info["valid_loss"] = loss
      self.info["valid_acc"] = acc
      self.info["valid_iou"] = iou

      # save to log
      self.save_to_log(logdir=self.log,
                       logger=self.tb_logger,
                       info=self.info,
                       epoch=1,
                       w_summary=self.CFG["train"]["save_summary"],
                       rand_imgs=rand_img,
                       img_summary=self.CFG["train"]["save_imgs"])
      print("*" * 80)
      return

    # train for n epochs
    for epoch in range(self.CFG["train"]["max_epochs"]):
      # get info for learn rate currently

      groups = self.optimizer.param_groups
      if len(groups) == 3:
        self.info["head_lr"] = groups[0]['lr']
        self.info["decoder_lr"] = groups[1]['lr']
        self.info["feat_lr"] = groups[2]['lr']
      elif len(groups) == 2:
        self.info["head_lr"] = groups[0]['lr']
        self.info["decoder_lr"] = groups[1]['lr']
        self.info["feat_lr"] = 0
      elif len(groups) == 1:
        self.info["head_lr"] = groups[0]['lr']
        self.info["decoder_lr"] = 0
        self.info["feat_lr"] = 0
      else:
        print("Invalid learning rate groups optimizer")

      # train for 1 epoch
      acc, iou, loss, update_mean = self.train_epoch(train_loader=self.parser.get_train_set(),
                                                     model=self.model,
                                                     criterion=self.criterion,
                                                     optimizer=self.optimizer,
                                                     epoch=epoch,
                                                     evaluator=self.evaluator,
                                                     block_bn=self.block_bn,
                                                     scheduler=self.scheduler)

      # update info
      self.info["train_update"] = update_mean
      self.info["train_loss"] = loss
      self.info["train_acc"] = acc
      self.info["train_iou"] = iou

      # remember best iou and save checkpoint
      if iou > best_train_iou:
        print("Best mean iou in training set so far, save model!")
        best_train_iou = iou
        self.save_checkpoint(bbone=self.model_single.backbone.state_dict(),
                             decoder=self.model_single.decoder.state_dict(),
                             head=self.model_single.head.state_dict(),
                             suffix="_train")

      if epoch % self.CFG["train"]["report_epoch"] == 0:
        # evaluate on validation set
        print("*" * 80)
        acc, iou, loss, rand_img = self.validate(val_loader=self.parser.get_valid_set(),
                                                 model=self.model,
                                                 criterion=self.criterion,
                                                 evaluator=self.evaluator,
                                                 save_images=self.CFG["train"]["save_imgs"],
                                                 class_dict=self.CFG["dataset"]["labels"])

        # colorize images
        for i, img in enumerate(rand_img[:]):
          rand_img[i] = self.colorizer.do(img)

        # update info
        self.info["valid_loss"] = loss
        self.info["valid_acc"] = acc
        self.info["valid_iou"] = iou

        # remember best iou and save checkpoint
        if iou > best_val_iou:
          print("Best mean iou in validation so far, save model!")
          print("*" * 80)
          best_val_iou = iou

          # now average the models and evaluate again
          print("Averaging the best {0} models".format(self.best_n_models))

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

          # append current backbone to its circular buffer
          current_decoder = self.model_single.decoder.state_dict()
          avg_decoder = copy.deepcopy(
              self.model_single.decoder).cpu().state_dict()
          self.best_decoders.append(copy.deepcopy(
              self.model_single.decoder).cpu().state_dict())

          # now average the decoder
          for i, decoder in enumerate(self.best_decoders):
            # for each weight key
            for key, val in decoder.items():
              # if it is the first time, zero the entry first
              if i == 0:
                avg_decoder[key].data.zero_()
              # then sum the avg contribution
              avg_decoder[key] += decoder[key] / \
                  float(len(self.best_decoders))

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
          self.model_single.decoder.load_state_dict(avg_decoder)
          self.model_single.head.load_state_dict(avg_head)

          # evaluate on validation set
          acc, iou, loss, _ = self.validate(val_loader=self.parser.get_valid_set(),
                                            model=self.model,
                                            criterion=self.criterion,
                                            evaluator=self.evaluator,
                                            save_images=self.CFG["train"]["save_imgs"],
                                            class_dict=self.CFG["dataset"]["labels"])

          # update info
          self.info["valid_loss_avg_models"] = loss
          self.info["valid_acc_avg_models"] = acc
          self.info["valid_iou_avg_models"] = iou

          # restore the current weights into model
          self.model_single.backbone.load_state_dict(current_backbone)
          self.model_single.decoder.load_state_dict(current_decoder)
          self.model_single.head.load_state_dict(current_head)

          # save the weights!
          self.save_checkpoint(
              current_backbone, current_decoder, current_head, suffix="_single")
          self.save_checkpoint(avg_backbone, avg_decoder, avg_head, suffix="")

        print("*" * 80)

        # save to log
        self.save_to_log(logdir=self.log,
                         logger=self.tb_logger,
                         info=self.info,
                         epoch=epoch,
                         w_summary=self.CFG["train"]["save_summary"],
                         rand_imgs=rand_img,
                         img_summary=self.CFG["train"]["save_imgs"])

    print('Finished Training')

    return

  def train_epoch(self, train_loader, model, criterion, optimizer, epoch, evaluator, block_bn, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()
    update_ratio_meter = AverageMeter()

    # empty the cache to train now
    if self.gpu:
      torch.cuda.empty_cache()

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
        target = target.cuda(non_blocking=True).long()

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      with torch.no_grad():
        evaluator.reset()
        evaluator.addBatch(output.argmax(dim=1), target)
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
      losses.update(loss.item(), input.size(0))
      acc.update(accuracy.item(), input.size(0))
      iou.update(jaccard.item(), input.size(0))

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
              'acc {acc.val:.3f} ({acc.avg:.3f}) | '
              'IoU {iou.val:.3f} ({iou.avg:.3f})'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                  data_time=data_time, loss=losses, acc=acc, iou=iou, lr=lr,
                  umean=update_mean, ustd=update_std))

      # step scheduler
      scheduler.step()

    return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg

  def validate(self, val_loader, model, criterion, evaluator, save_images, class_dict):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    iou = AverageMeter()
    rand_imgs = []

    # switch to evaluate mode
    model.eval()
    evaluator.reset()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()
      for i, (input, target) in enumerate(val_loader):
        if not self.multi_gpu and self.gpu:
          input = input.cuda()
        if self.gpu:
          target = target.cuda(non_blocking=True).long()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # save a random image, if desired
        if(save_images):
          rand_imgs.append(self.make_log_image(output[0], target[0]))

        # measure accuracy and record loss
        evaluator.addBatch(output.argmax(dim=1), target)
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

      accuracy = evaluator.getacc()
      jaccard, class_jaccard = evaluator.getIoU()
      acc.update(accuracy.item(), input.size(0))
      iou.update(jaccard.item(), input.size(0))

      print('Validation set:\n'
            'Time avg per batch {batch_time.avg:.3f}\n'
            'Loss avg {loss.avg:.4f}\n'
            'Acc avg {acc.avg:.3f}\n'
            'IoU avg {iou.avg:.3f}'.format(batch_time=batch_time,
                                           loss=losses,
                                           acc=acc, iou=iou))
      # print also classwise
      for i, jacc in enumerate(class_jaccard):
        print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
            i=i, class_str=class_dict[i], jacc=jacc))

    return acc.avg, iou.avg, losses.avg, rand_imgs

  def make_log_image(self, pred, target):
    # colorize and put in format
    pred = pred.cpu().numpy().argmax(0)
    target = target.cpu().numpy()
    output = np.concatenate((pred, target), axis=1)

    return output
