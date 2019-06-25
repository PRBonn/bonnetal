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
means(rgb):  [0.39306904 0.37922944 0.3555977]
stds(rgb):  [0.20292485 0.19964703 0.19909012]

********************************************************************************
Num of pixels:  9266918400
Frequency:  [0.21914592 0.03913536 0.         0.27824563 0.         0.
 0.         0.         0.         0.         0.         0.        ]
********************************************************************************
Log strategy
Weights:  [ 4.66369259 17.40556773 50.49834979  3.8312145  50.49834979 50.49834979
 50.49834979 50.49834979 50.49834979 50.49834979 50.49834979 50.49834979]
Linear strategy
Weights:  [0.78085408 0.96086464 1.         0.72175437 1.         1.
 1.         1.         1.         1.         1.         1.        ]
Squared strategy
Weights:  [0.6097331  0.92326085 1.         0.52092937 1.         1.
 1.         1.         1.         1.         1.         1.        ]
1/w strategy
Weights:  [4.56316944e+00 2.55523306e+01 1.00000000e+08 3.59394670e+00
 1.00000000e+08 1.00000000e+08 1.00000000e+08 1.00000000e+08
 1.00000000e+08 1.00000000e+08 1.00000000e+08 1.00000000e+08]


'''

EXTENSIONS = ['.jpg', '.png']
SCALES = [1.0]

# synthia doesn't have split, so we generate it
TRAIN_SPLIT = 0.9
VALID_SPLIT = 1 - TRAIN_SPLIT

# COLOR DICT (rgb)
COLOR_MAP = {0: [0, 0, 0],
             1: [128, 128, 128],
             2: [128, 0, 0],
             3: [128, 64, 128],
             4: [0, 0, 192],
             5: [64, 64, 128],
             6: [128, 128, 0],
             7: [192, 192, 128],
             8: [64, 0, 128],
             9: [192, 128, 128],
             10: [64, 64, 0],
             11: [0, 128, 192]}

# LUT
COLOR_LUT = np.zeros((256, 256, 256), dtype=np.uint8)
for xent_clas, color_rgb in COLOR_MAP.items():
  COLOR_LUT[color_rgb[0], color_rgb[1], color_rgb[2]] = xent_clas


class ToLabel:
  def __call__(self, image):
    label_color = np.array(image)
    label_mono = COLOR_LUT[label_color[..., 0],
                           label_color[..., 1],
                           label_color[..., 2]]
    return torch.from_numpy(label_mono).long()


def load_image(file):
  return Image.open(file)


def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS)


def image_path_city(root, name):
  return os.path.join(root, name)


def image_basename(filename):
  return os.path.basename(os.path.splitext(filename)[0])


class synthia(Dataset):

  def __init__(self, root, subset, h, w, means, stds, crop_h=None, crop_w=None):
    self.images_root = os.path.join(root, 'RGB/')
    self.labels_root = os.path.join(root, 'GT/')

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

    # get the range for data depending on set
    self.n_images = len(self.filenames)
    self.split = int(TRAIN_SPLIT * self.n_images)
    if self.subset == 'train':
      self.filenames = self.filenames[:self.split]
      self.filenamesGt = self.filenamesGt[:self.split]
    else:
      self.filenames = self.filenames[self.split:]
      self.filenamesGt = self.filenamesGt[self.split:]
    print("Number of images for", self.subset, len(self.filenamesGt))

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
      label = load_image(f).convert('RGB')  # synthia labels are rgb

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
      self.train_dataset = synthia(root=self.location,
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

      self.valid_dataset = synthia(root=self.location,
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
