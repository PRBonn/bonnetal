# This file is covered by the LICENSE file in the root of this project.

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image, ImageMath
import random
import torchvision.transforms.functional as TF
import cv2

EXTENSIONS = ['.jpg', '.jpeg', '.png']
SCALES = [1.0]


class ToLabel:
  def __call__(self, image):
    return torch.from_numpy(np.array(image)).long()


def load_image(file):
  return Image.open(file)


def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS)


class Mapillary(Dataset):
  def __init__(self, root, subset, h, w, means, stds, crop_h=None, crop_w=None):
    self.images_root = os.path.join(root, subset, "images")
    self.labels_root = os.path.join(root, subset, "instances")

    self.subset = subset
    assert self.subset == 'training' or self.subset == 'validation'

    self.w = w
    self.h = h
    self.means = means
    self.stds = stds

    if self.subset == 'training':
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
        os.path.expanduser(self.labels_root)) for f in fn if is_image(f)]
    self.filenamesGt.sort()

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
      # open
      label = load_image(f)
      # label in upper byte
      label = (np.array(label) // 256).astype(np.uint8)
      # back to pil
      label = Image.fromarray(label).convert('L')

    # resize (resizing is different if we are in train or valid mode)
    # generate resizer
    if self.subset == 'training':
      new_size = max(self.crop_w, self.crop_h)
    else:
      new_size = max(self.w, self.h)
    resize_img = transforms.Resize(new_size, Image.BILINEAR)
    resize_lbl = transforms.Resize(new_size, Image.NEAREST)

    image = resize_img(image)
    label = resize_lbl(label)

    # augment data and tensorize
    if self.subset == 'training':
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
      # cv2.imshow("train_label", np.array(label))
      # cv2.waitKey(0)

    if self.subset == 'validation':
      # crop
      image = self.crop(image, 0, 0, self.h, self.w)
      label = self.crop(label, 0, 0, self.h, self.w)

      # show (set workers = 0)
      # cv2.imshow("valid_img", np.array(image)[:, :, ::-1])
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
      self.train_dataset = Mapillary(root=self.location,
                                     subset='training',
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

      self.valid_dataset = Mapillary(root=self.location,
                                     subset='validation',
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


# Frequency:  [9.83478885e-06 2.61177410e-05 8.25302258e-03 1.24475992e-02
#              2.34856513e-03 4.01102850e-03 8.45642312e-03 3.67755262e-03
#              2.53977244e-03 8.67037632e-04 2.12182379e-03 6.52594188e-03
#              1.00791227e-03 1.86677277e-01 2.43326577e-03 3.31755064e-02
#              7.32614806e-03 1.19798200e-01 6.64038120e-04 3.16104843e-03
#              3.74913293e-04 2.56731127e-04 5.74237829e-06 7.83188745e-03
#              1.34846566e-02 2.14708124e-03 1.59920755e-04 5.49865951e-02
#              3.92615717e-03 1.24808674e-02 1.46343000e-01 9.96375155e-04
#              6.63253709e-04 1.51291787e-04 4.19959714e-05 5.74272483e-03
#              3.26610450e-04 2.13034730e-05 4.69177059e-05 6.16120734e-04
#              2.89454731e-05 5.70617118e-04 1.05586847e-04 1.16979761e-04
#              5.77192404e-04 8.58744012e-03 1.05670490e-03 4.21331584e-03
#              1.74059847e-03 7.86244263e-04 5.05391809e-03 5.49681742e-04
#              5.91759307e-04 8.67158483e-05 2.29324571e-03 3.26198624e-02
#              7.29259642e-05 6.22935762e-04 3.06526238e-04 4.07481045e-04
#              1.11046866e-04 3.87534294e-03 6.37432551e-05 7.71546877e-04
#              2.02701328e-02 1.78632878e-02]
# ********************************************************************************
# Log strategy
# Weights:  [50.4737741  50.43313846 35.89212136 31.31626014 45.24376012 42.1455515
#            35.63911283 42.7321446  44.86416279 48.42074971 45.70240615 38.19676243
#            48.0993802   5.3228146  45.07480853 19.3013334  37.09273919  7.64226678
#            48.89154736 43.67403033 49.57828263 49.86463544 50.4839975  36.42772259
#            30.36167986 45.65085132 50.10170631 13.8296917  42.29329134 31.28469147
#            6.49885698 48.12553707 48.89338451 50.12294763 50.39357643 39.34380858
#            49.69491703 50.44514604 50.38132621 49.00403015 50.42608832 49.11133187
#            50.2357603  50.20759142 49.09579741 35.478048   47.98907484 41.79759559
#            46.49510156 48.60702458 40.41185463 49.16085902 49.06141766 50.28248898
#            45.35479829 19.49995729 50.31669128 48.98800046 49.74357666 49.49995457
#            50.22225647 42.38224891 50.33949273 48.64106623 25.32900959 26.90771039]
# Linear strategy
# Weights:  [0.99999017 0.99997388 0.99174698 0.9875524  0.99765143 0.99598897
#            0.99154358 0.99632245 0.99746023 0.99913296 0.99787818 0.99347406
#            0.99899209 0.81332272 0.99756673 0.96682449 0.99267385 0.8802018
#            0.99933596 0.99683895 0.99962509 0.99974327 0.99999426 0.99216811
#            0.98651534 0.99785292 0.99984008 0.9450134  0.99607384 0.98751913
#            0.853657   0.99900362 0.99933675 0.99984871 0.999958   0.99425728
#            0.99967339 0.9999787  0.99995308 0.99938388 0.99997105 0.99942938
#            0.99989441 0.99988302 0.99942281 0.99141256 0.9989433  0.99578668
#            0.9982594  0.99921376 0.99494608 0.99945032 0.99940824 0.99991328
#            0.99770675 0.96738014 0.99992707 0.99937706 0.99969347 0.99959252
#            0.99988895 0.99612466 0.99993626 0.99922845 0.97972987 0.98213671]
# Squared strategy
# Weights:  [0.99998033 0.99994777 0.98356207 0.97525974 0.99530839 0.99199403
#            0.98315866 0.99265842 0.99492691 0.99826668 0.99576085 0.9869907
#            0.99798519 0.66149385 0.99513939 0.9347496  0.98540138 0.77475521
#            0.99867236 0.9936879  0.99925031 0.9994866  0.99998852 0.98439756
#            0.97321252 0.99571045 0.99968018 0.89305034 0.9921631  0.97519404
#            0.72873027 0.99800824 0.99867393 0.99969744 0.99991601 0.98854753
#            0.99934689 0.99995739 0.99990617 0.99876814 0.99994211 0.99885909
#            0.99978884 0.99976605 0.99884595 0.98289886 0.99788771 0.99159112
#            0.99652183 0.99842813 0.98991771 0.99890094 0.99881683 0.99982658
#            0.99541877 0.93582433 0.99985415 0.99875452 0.99938704 0.9991852
#            0.99977792 0.99226433 0.99987252 0.9984575  0.95987061 0.96459252]
# 1/w strategy
# Weights:  [1.01576582e+05 3.82734964e+04 1.21167582e+02 8.03367124e+01
#            4.25790084e+02 2.49311992e+02 1.18253167e+02 2.71919231e+02
#            3.93734513e+02 1.15333917e+03 4.71290450e+02 1.53234351e+02
#            9.92139996e+02 5.35683808e+00 4.10968627e+02 3.01427109e+01
#            1.36497191e+02 8.34737015e+00 1.50591496e+03 3.16349736e+02
#            2.66721225e+03 3.89497395e+03 1.73841140e+05 1.27682979e+02
#            7.41583034e+01 4.65746394e+02 6.25270604e+03 1.81862473e+01
#            2.54701330e+02 8.01225721e+01 6.83326114e+00 1.00362796e+03
#            1.50769594e+03 6.60930724e+03 2.38061392e+04 1.74133062e+02
#            3.06165766e+03 4.69186790e+04 2.13093732e+04 1.62303217e+03
#            3.45357852e+04 1.75245790e+03 9.46997974e+03 8.54775657e+03
#            1.73249452e+03 1.16449002e+02 9.46329039e+02 2.37342194e+02
#            5.74511740e+02 1.27185320e+03 1.97865894e+02 1.81920143e+03
#            1.68984770e+03 1.15305877e+04 4.36061272e+02 3.06561592e+01
#            1.37106572e+04 1.60527619e+03 3.26225704e+03 2.45404166e+03
#            9.00439598e+03 2.58041013e+02 1.56854736e+04 1.29608073e+03
#            4.93336436e+01 5.59807046e+01]
