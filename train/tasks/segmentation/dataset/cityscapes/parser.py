# This file is covered by the LICENSE file in the root of this project.


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import random
import torchvision.transforms.functional as TF
import cv2

'''
********************************************************************************
Num of pixels:  6239027200
Frequency:  [0.32634084 0.05385963 0.20191867 0.00580399 0.00777159 0.0108621
 0.00184483 0.004892   0.14084477 0.01025243 0.03549584 0.01077129
 0.00119328 0.06194921 0.00236816 0.00208299 0.00206185 0.00087288
 0.00366238 0.11515129]
********************************************************************************
Log strategy
Weights:  [ 3.36258308 14.0332588   4.9894646  39.25158044 36.5057383  32.8996697
 46.27562166 40.67150678  6.70475007 33.55271029 18.51486878 32.99530091
 47.68301907 12.69611686 45.20456246 45.78191067 45.82529538 48.40734227
 42.75923387  7.88855432]
Linear strategy
Weights:  [0.67365916 0.94614037 0.79808133 0.99419601 0.99222841 0.9891379
 0.99815517 0.995108   0.85915523 0.98974757 0.96450416 0.98922871
 0.99880672 0.93805079 0.99763184 0.99791701 0.99793815 0.99912712
 0.99633762 0.88484871]
Squared strategy
Weights:  [0.45381667 0.89518161 0.63693381 0.98842572 0.98451722 0.97839379
 0.99631374 0.99023994 0.73814771 0.97960025 0.93026828 0.97857344
 0.99761486 0.87993928 0.99526929 0.99583837 0.99588055 0.99825501
 0.99268866 0.78295723]
1/w strategy
Weights:  [   3.06428082   18.56677964    4.95248873  172.29510449  128.6736306
   92.06314882  542.05250028  204.41510864    7.10001462   97.53773954
   28.17230941   92.8393159   838.0188499    16.14225275  422.26714411
  480.07770874  484.99891486 1145.6235226   273.04603536    8.68422655]

'''

EXTENSIONS = ['.jpg', '.png']
SCALES = [1.0]


class ToLabel:
  def __call__(self, image):
    return torch.from_numpy(np.array(image)).long()


def load_image(file):
  return Image.open(file)


def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS)


def is_label(filename):
  return filename.endswith("_labelTrainIds.png")


def image_path_city(root, name):
  return os.path.join(root, name)


def image_basename(filename):
  return os.path.basename(os.path.splitext(filename)[0])


class cityscapes(Dataset):

  def __init__(self, root, subset, h, w, means, stds, crop_h=None, crop_w=None):
    self.images_root = os.path.join(root, 'leftImg8bit/')
    self.labels_root = os.path.join(root, 'gtFine/')

    self.images_root += subset
    self.labels_root += subset

    self.subset = subset
    assert self.subset == 'train' or self.subset == 'val'

    self.w = w
    self.h = h
    self.means = means
    self.stds = stds

    if self.subset == 'train':
      self.crop_h = crop_h
      self.crop_w = crop_w

      # check that parameters make sense
      assert(self.crop_h <= self.h)
      assert(self.crop_w <= self.w)

      self.resize_crop_img = transforms.Resize((self.crop_h, self.crop_w),
                                               Image.BILINEAR)
      self.resize_crop_lbl = transforms.Resize((self.crop_h, self.crop_w),
                                               Image.NEAREST)

    print("Images from: ", self.images_root)
    print("Labels from: ", self.labels_root)

    self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
    self.filenames.sort()

    self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
    self.filenamesGt.sort()

    assert len(self.filenames) == len(self.filenamesGt)

    # transformations for images
    self.jitter = transforms.ColorJitter(brightness=0.03,
                                         contrast=0.03,
                                         saturation=0.03,
                                         hue=0.03)
    self.h_flip = TF.hflip
    self.crop_param = transforms.RandomCrop.get_params
    self.crop = TF.crop

    self.resize_img = transforms.Resize((self.h, self.w), Image.BILINEAR)
    self.resize_lbl = transforms.Resize((self.h, self.w), Image.NEAREST)

    # transformations for tensors
    self.norm = transforms.Normalize(mean=self.means, std=self.stds)
    self.tensorize_img = transforms.ToTensor()
    self.tensorize_lbl = ToLabel()

  def __getitem__(self, index):
    filename = self.filenames[index]
    filenameGt = self.filenamesGt[index]

    with open(image_path_city(self.images_root, filename), 'rb') as f:
      image = load_image(f).convert('RGB')
    with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
      label = load_image(f).convert('L')

    # resize (resizing is different if we are in train or valid mode)
    image = self.resize_img(image)
    label = self.resize_lbl(label)

    # augment data and tensorize
    if self.subset == 'train':
      # crop randomly sized patches
      scale = SCALES[random.randrange(len(SCALES))]
      size = (int(self.crop_h * scale), int(self.crop_w * scale))
      i, j, h, w = self.crop_param(image, output_size=size)
      image = self.resize_crop_img(self.crop(image, i, j, h, w))
      label = self.resize_crop_lbl(self.crop(label, i, j, h, w))

      # flip
      if random.random() > 0.5:
        image = self.h_flip(image)
        label = self.h_flip(label)

      # jitter
      if random.random() > 0.5:
        image = self.jitter(image)

      # show (set workers = 0)
      # cv2.imshow("train_img", np.array(image)[:, :, ::-1])
      # cv2.waitKey(0)

    # tensorize
    image = self.tensorize_img(image)
    label = self.tensorize_lbl(label)

    # normalize
    image = self.norm(image)

    return image, label

  def __len__(self):
    return len(self.filenames)


class Parser():
  # standard conv, BN, relu
  def __init__(self, img_prop, img_means, img_stds, classes, train, location=None, batch_size=None, crop_prop=None, workers=2):
    super(Parser, self).__init__()

    self.img_prop = img_prop
    self.img_means = img_means
    self.img_stds = img_stds
    self.classes = classes
    self.train = train

    if self.train:
      # if I am training, get the dataset
      self.location = location
      self.batch_size = batch_size
      self.crop_prop = crop_prop
      self.workers = workers

      # Data loading code
      self.train_dataset = cityscapes(root=self.location,
                                      subset='train',
                                      h=self.img_prop["height"],
                                      w=self.img_prop["width"],
                                      means=self.img_means,
                                      stds=self.img_stds,
                                      crop_h=self.crop_prop["height"],
                                      crop_w=self.crop_prop["width"])

      self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     num_workers=self.workers,
                                                     pin_memory=True,
                                                     drop_last=True)
      assert len(self.trainloader) > 0
      self.trainiter = iter(self.trainloader)

      # calculate validation batch from train batch and image sizes
      factor_val_over_train = float(self.img_prop["height"] * self.img_prop["width"]) / float(
          self.crop_prop["height"] * self.crop_prop["width"])
      self.val_batch_size = max(
          1, int(self.batch_size / factor_val_over_train))

      # if gpus are available make val_batch_size at least the number of gpus
      if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        self.val_batch_size = max(
            self.val_batch_size, torch.cuda.device_count())

      print("Inference batch size: ", self.val_batch_size)

      self.valid_dataset = cityscapes(root=self.location,
                                      subset='val',
                                      h=self.img_prop["height"],
                                      w=self.img_prop["width"],
                                      means=self.img_means,
                                      stds=self.img_stds)

      self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                     batch_size=self.val_batch_size,
                                                     shuffle=False,
                                                     num_workers=self.workers,
                                                     pin_memory=True,
                                                     drop_last=True)
      assert len(self.validloader) > 0
      self.validiter = iter(self.validloader)

  def get_train_batch(self):
    images, labels = self.trainiter.next()
    return images, labels

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    images, labels = self.validiter.next()
    return images, labels

  def get_valid_set(self):
    return self.validloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_img_size(self):
    h = self.img_prop["height"]
    w = self.img_prop["width"]
    d = self.img_prop["depth"]
    return h, w, d

  def get_n_classes(self):
    return len(self.classes)

  def get_class_string(self, idx):
    return self.classes[idx]

  def get_means_stds(self):
    return self.img_means, self.img_stds
