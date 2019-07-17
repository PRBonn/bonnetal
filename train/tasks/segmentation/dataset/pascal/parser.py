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
means(rgb):  [0.45223407 0.4311933  0.39899281]
stds(rgb):  [0.27660984 0.27312623 0.28520525]

********************************************************************************
Num of pixels:  521582502
Frequency:  [0.6931993  0.00710222 0.00301227 0.00840544 0.0054889  0.00659333
 0.01667306 0.0135898  0.02391466 0.00953882 0.00963945 0.01197583
 0.01798247 0.00944371 0.01049897 0.04792238 0.0055694  0.00802801
 0.01359605 0.01515243 0.00790093 0.05477256]
********************************************************************************
Log strategy
Weights:  [ 1.85748429 37.39511976 43.95318014 35.70218637 39.73067016 38.10123389
 27.76497476 30.26818736 23.26785978 34.35132469 34.23638246 31.77100104
 26.82483072 34.46068846 33.28548492 15.21721153 39.60714259 36.17629464
 30.26264657 28.94465548 36.33880067 13.86788149]
Linear strategy
Weights:  [0.3068007  0.99289778 0.99698773 0.99159456 0.9945111  0.99340667
 0.98332694 0.9864102  0.97608534 0.99046118 0.99036055 0.98802417
 0.98201753 0.99055629 0.98950103 0.95207762 0.9944306  0.99197199
 0.98640395 0.98484757 0.99209907 0.94522744]
Squared strategy
Weights:  [0.09412667 0.985846   0.99398453 0.98325977 0.98905233 0.98685682
 0.96693188 0.97300508 0.9527426  0.98101334 0.98081402 0.97619177
 0.96435844 0.98120176 0.97911228 0.90645179 0.98889222 0.98400843
 0.97299274 0.96992473 0.98426056 0.89345491]
1/w strategy
Weights:  [  1.44258654 140.8008728  331.97430826 118.97038718 182.18561016
 151.66831133  59.97697307  73.5845345   41.81534498 104.83460948
 103.74023964  83.50146797  55.60969236 105.89043     95.24731376
  20.86707183 179.55220709 124.56376145  73.55069678  65.99596618
 126.56715217  18.25731376]



'''

IMG_EXT = ['.jpg']
LBL_EXT = ['.png']
SCALES = [1.0]

# make lut to map classes
# everything (except 0:20) maps to 21 (void)
LUT = np.full(256, 21, dtype=np.uint8)
for i in range(21):
  LUT[i] = i  # all classes up to 20 map the same


class ToLabel:
  def __call__(self, label):
    label = np.array(label)
    return torch.from_numpy(label).long()


def load_image(file):
  return Image.open(file)


def load_label(file):
  # this is gross, but pascal is small, so I don't care
  label = LUT[np.array(Image.open(file))]  # throw away color map
  # print(np.unique(label))
  return Image.fromarray(label)


def is_image(filename):
  return any(filename.endswith(ext) for ext in IMG_EXT)


def is_label(filename):
  return any(filename.endswith(ext) for ext in LBL_EXT)


def resize_and_fit(img, new_h, new_w, img_type):
  # check img_type
  assert(img_type is "RGB" or img_type is "L")

  # get current size
  w, h = img.size

  # generate new img
  out_img = Image.new(img_type, (new_w, new_h))

  # now do size magic
  curr_asp_ratio = h / w
  new_asp_ratio = new_h / new_w

  # do resizing according to aspect ratio
  if curr_asp_ratio > new_asp_ratio:
    # fit h to h
    new_tmp_h = new_h
    new_tmp_w = int(w * new_h / h)
  else:
    # fit w to w
    new_tmp_w = new_w
    new_tmp_h = int(h * new_w / w)

  # resize the original image
  if img_type is "RGB":
    tmp_img = img.resize((new_tmp_w, new_tmp_h), Image.BILINEAR)
  else:
    tmp_img = img.resize((new_tmp_w, new_tmp_h), Image.NEAREST)

  # put in padded image
  out_img.paste(tmp_img, (int((new_w-new_tmp_w)//2),
                          int((new_h-new_tmp_h)//2)))

  return out_img


class Pascal(Dataset):
  def __init__(self, root, subset, h, w, means, stds, crop_h=None, crop_w=None):
    self.images_root = os.path.join(root, "JPEGImages/")
    self.labels_root = os.path.join(root, "SegmentationClass/")

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

    if self.subset == 'train':
      self.path_list_txt = "train.txt"
    else:
      self.path_list_txt = "val.txt"

    # file
    self.path_list_txt = os.path.join(
        root, "ImageSets/Segmentation/", self.path_list_txt)

    print("File list from: ", self.path_list_txt)

    # actual list
    with open(self.path_list_txt) as f:
      self.path_list = f.read().splitlines()

    self.filenames = [os.path.join(self.images_root, f + IMG_EXT[0])
                      for f in self.path_list]
    self.filenames.sort()

    print("Number of ", self.subset, "images:", len(self.filenames))

    self.filenamesGt = [os.path.join(self.labels_root, f + LBL_EXT[0])
                        for f in self.path_list]
    self.filenamesGt.sort()
    print("Number of ", self.subset, "labels:", len(self.filenamesGt))

    assert len(self.filenames) == len(self.filenamesGt)

    # transformations for images
    self.jitter = transforms.ColorJitter(brightness=0.05,
                                         contrast=0.05,
                                         saturation=0.05,
                                         hue=0.05)
    self.h_flip = TF.hflip
    self.crop_param = transforms.RandomCrop.get_params
    self.crop = TF.crop

    # transformations for tensors
    self.norm = transforms.Normalize(mean=self.means, std=self.stds)
    self.tensorize_img = transforms.ToTensor()
    self.tensorize_lbl = ToLabel()

  def __getitem__(self, index):
    filename = self.filenames[index]
    filenameGt = self.filenamesGt[index]

    with open(filename, 'rb') as f:
      image = load_image(f).convert('RGB')
    with open(filenameGt, 'rb') as f:
      label = load_label(f)

    # resize (resizing is different if we are in train or valid mode)
    # generate resizer
    new_h = self.h
    new_w = self.w      
    image = resize_and_fit(image, new_h, new_w, "RGB")
    label = resize_and_fit(label, new_h, new_w, "L")

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
      # cv2.imshow("train_lbl", LUT[np.array(label)].astype(np.float32) / 21.0)
      # cv2.waitKey(0)

    # if self.subset == 'val':
    #   show (set workers = 0)
    #   cv2.imshow("valid_img", np.array(image)[:, :, ::-1])
    #   cv2.waitKey(0)

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
      self.train_dataset = Pascal(root=self.location,
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

      self.valid_dataset = Pascal(root=self.location,
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
