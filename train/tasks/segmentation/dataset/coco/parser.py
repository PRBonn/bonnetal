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
means(rgb): [0.47037394 0.44669544 0.40731883]
stds(rgb):  [0.27876515 0.27429348 0.28861644]

Num of pixels:  20354514743
Frequency:  [1.03595582e-01 8.69684146e-02 1.56773018e-03 5.82611153e-03
 4.81058803e-03 3.16223407e-03 7.10428246e-03 7.05528129e-03
 5.76380800e-03 2.61561927e-03 6.43652977e-04 1.06484818e-03
 1.07875453e-03 6.98690299e-04 3.30846713e-03 1.65507630e-03
 6.01311471e-03 4.48253614e-03 3.37169861e-03 1.84147500e-03
 2.59677750e-03 4.60398424e-03 1.72642509e-03 3.25079452e-03
 3.17922092e-03 9.28004241e-04 3.00187903e-03 1.02122941e-03
 7.74191387e-04 3.01174387e-03 3.52895713e-04 3.00067384e-04
 3.31869518e-04 1.49010479e-04 6.79291802e-04 1.38228842e-04
 1.80938973e-04 5.82766927e-04 1.16591352e-03 5.55644934e-04
 1.83246594e-03 9.64564533e-04 2.68603416e-03 3.53508157e-04
 4.86584039e-04 3.04124273e-04 6.10763335e-03 2.51745687e-03
 1.19416608e-03 3.49547734e-03 1.43915212e-03 1.98661498e-03
 8.55161482e-04 1.22814719e-03 8.29490195e-03 2.09027995e-03
 3.95652007e-03 6.19389573e-03 5.21590882e-03 2.07798941e-03
 9.07128538e-03 2.41144264e-02 3.08866224e-03 3.29269545e-03
 3.44996375e-03 2.17966680e-04 5.69893272e-04 1.33344903e-03
 1.06328032e-03 9.01832455e-04 3.21914572e-03 5.66035602e-05
 1.64377842e-03 3.49153060e-03 2.07557215e-03 1.33823711e-03
 1.73024557e-03 3.61442810e-04 3.16915293e-03 3.26746183e-05
 1.69843597e-04 2.24706580e-03 1.08037029e-03 1.15556594e-03
 2.19738081e-03 2.83867548e-03 4.58330597e-03 6.13085488e-03
 5.53305060e-03 1.95223391e-03 1.24932391e-03 2.50343202e-03
 4.28674371e-03 1.36921250e-03 3.32965639e-03 1.77840698e-03
 5.10465080e-04 2.04364749e-03 1.78148449e-02 2.76140555e-03
 5.15718043e-03 2.26026582e-02 1.41155564e-03 9.53189813e-03
 2.24532113e-02 2.74807151e-03 1.89481003e-02 1.06579298e-03
 7.92184791e-04 7.43852368e-04 5.30637362e-03 2.23005552e-03
 8.45400979e-03 6.19471526e-03 4.12920107e-03 1.70490166e-03
 9.71786370e-03 6.47590623e-02 1.39815155e-02 8.92733677e-03
 8.67340285e-02 8.37997595e-03 1.41617307e-02 1.35923816e-02
 2.34834311e-02 7.09260706e-03 4.15174260e-02 1.33029928e-02
 4.80344372e-03 7.12591456e-03 3.01482646e-02 4.35955532e-03
 6.39422134e-02 6.29973913e-03]
********************************************************************************
Log strategy
Weights:  [3.30289772 3.44347075 4.45638856 4.38993873 4.40558454 4.43124634
 4.3704214  4.37116607 4.39089505 4.43982977 4.47110532 4.46438403
 4.4641625  4.47022578 4.42895632 4.45500307 4.38707112 4.4106653
 4.42796692 4.45204959 4.4401263  4.40878283 4.45387203 4.42985916
 4.43098019 4.46656527 4.43376055 4.46507904 4.46901983 4.43360579
 4.47575828 4.47660484 4.47609518 4.47902746 4.47053574 4.47920049
 4.47851516 4.47207879 4.4627746  4.47251257 4.45219224 4.46598228
 4.43872198 4.47574847 4.47361754 4.47653982 4.3856233  4.44137513
 4.46232492 4.42603155 4.45842983 4.44975287 4.46772733 4.46178419
 4.35241406 4.44811407 4.41883936 4.38430288 4.39932503 4.44830829
 4.34076057 4.12794364 4.43239948 4.42920318 4.42674297 4.4779212
 4.47228467 4.4601095  4.464409   4.46698271 4.43035479 4.4805109
 4.45518222 4.42609323 4.44834649 4.46003338 4.45381149 4.47562135
 4.43113793 4.48089522 4.47869317 4.44563805 4.46413676 4.46293932
 4.44642236 4.43632268 4.40910322 4.38526776 4.3944411  4.45029668
 4.46144729 4.44159602 4.41370389 4.45954104 4.42862471 4.45304841
 4.47323538 4.4488511  4.21416693 4.43753689 4.40023077 4.14827356
 4.45886822 4.33387961 4.15029549 4.4377465  4.19835921 4.46436897
 4.46873253 4.46950434 4.39793066 4.44590653 4.35002018 4.38429034
 4.41615226 4.45421316 4.33110842 3.65425719 4.26863963 4.34291598
 3.44555095 4.3511337  4.26604345 4.27425755 4.13640191 4.37059881
 3.90903173 4.27844617 4.40569505 4.37009275 4.04897801 4.41257335
 3.66257514 4.38268395]
Linear strategy
Weights:  [0.89640442 0.91303159 0.99843227 0.99417389 0.99518941 0.99683777
 0.99289572 0.99294472 0.99423619 0.99738438 0.99935635 0.99893515
 0.99892125 0.99930131 0.99669153 0.99834492 0.99398689 0.99551746
 0.9966283  0.99815853 0.99740322 0.99539602 0.99827357 0.99674921
 0.99682078 0.999072   0.99699812 0.99897877 0.99922581 0.99698826
 0.9996471  0.99969993 0.99966813 0.99985099 0.99932071 0.99986177
 0.99981906 0.99941723 0.99883409 0.99944436 0.99816753 0.99903544
 0.99731397 0.99964649 0.99951342 0.99969588 0.99389237 0.99748254
 0.99880583 0.99650452 0.99856085 0.99801339 0.99914484 0.99877185
 0.9917051  0.99790972 0.99604348 0.9938061  0.99478409 0.99792201
 0.99092871 0.97588557 0.99691134 0.9967073  0.99655004 0.99978203
 0.99943011 0.99866655 0.99893672 0.99909817 0.99678085 0.9999434
 0.99835622 0.99650847 0.99792443 0.99866176 0.99826975 0.99963856
 0.99683085 0.99996733 0.99983016 0.99775293 0.99891963 0.99884443
 0.99780262 0.99716132 0.99541669 0.99386915 0.99446695 0.99804777
 0.99875068 0.99749657 0.99571326 0.99863079 0.99667034 0.99822159
 0.99948953 0.99795635 0.98218516 0.99723859 0.99484282 0.97739734
 0.99858844 0.9904681  0.97754679 0.99725193 0.9810519  0.99893421
 0.99920782 0.99925615 0.99469363 0.99776994 0.99154599 0.99380528
 0.9958708  0.9982951  0.99028214 0.93524094 0.98601848 0.99107266
 0.91326597 0.99162002 0.98583827 0.98640762 0.97651657 0.99290739
 0.95848257 0.98669701 0.99519656 0.99287409 0.96985174 0.99564044
 0.93605779 0.99370026]
Squared strategy
Weights:  [0.80354088 0.83362668 0.996867   0.98838172 0.99040197 0.99368553
 0.98584191 0.98593921 0.98850561 0.9947756  0.99871311 0.99787144
 0.99784365 0.99860311 0.99339401 0.99669259 0.98800993 0.99105502
 0.99326797 0.99632044 0.99481319 0.99081323 0.99655013 0.99350898
 0.99365167 0.99814485 0.99400525 0.99795858 0.99845222 0.99398558
 0.99929433 0.99939996 0.99933637 0.999702   0.99864188 0.99972356
 0.99963815 0.99883481 0.99766953 0.99888902 0.99633843 0.9980718
 0.99463515 0.99929311 0.99902707 0.99939184 0.98782204 0.99497142
 0.99761309 0.99302126 0.99712377 0.99603072 0.99829041 0.99754521
 0.983479   0.99582381 0.99210261 0.98765057 0.98959539 0.99584834
 0.98193972 0.95235265 0.99383222 0.99342545 0.99311197 0.99956411
 0.99886054 0.99733488 0.99787457 0.99819715 0.99357207 0.9998868
 0.99671515 0.99302913 0.99585316 0.99732532 0.9965425  0.99927725
 0.99367174 0.99993465 0.99966034 0.99551092 0.99784043 0.9976902
 0.99561007 0.99433071 0.99085439 0.98777588 0.98896451 0.99609934
 0.99750291 0.9949994  0.99144489 0.99726345 0.99335177 0.99644635
 0.99897933 0.99591688 0.96468768 0.99448481 0.98971224 0.95530556
 0.99717888 0.98102706 0.95559772 0.99451141 0.96246283 0.99786955
 0.99841626 0.99851285 0.98941541 0.99554486 0.98316345 0.98764894
 0.99175865 0.9965931  0.98065871 0.87467561 0.97223245 0.98222502
 0.83405473 0.98331027 0.97187709 0.97299999 0.95358461 0.98586509
 0.91868884 0.97357098 0.99041619 0.98579895 0.94061239 0.9912999
 0.87620418 0.98744021]
1/w strategy
Weights:  [9.65292034e+00 1.14984261e+01 6.37860798e+02 1.71640773e+02
 2.07874363e+02 3.16231125e+02 1.40759971e+02 1.41737592e+02
 1.73496110e+02 3.82317177e+02 1.55360808e+03 9.39092189e+02
 9.26986355e+02 1.43122881e+03 3.02253865e+02 6.04198100e+02
 1.66302887e+02 2.23087497e+02 2.96585535e+02 5.43039993e+02
 3.85091194e+02 2.17202705e+02 5.79228263e+02 3.07616159e+02
 3.14541481e+02 1.07756967e+03 3.33123573e+02 9.79202324e+02
 1.29165359e+03 3.32032445e+02 2.83361805e+03 3.33247373e+03
 3.01314165e+03 6.71048707e+03 1.47209973e+03 7.23385690e+03
 5.52641986e+03 1.71592243e+03 8.57689188e+02 1.79967807e+03
 5.45709757e+02 1.03672652e+03 3.72294698e+02 2.82870902e+03
 2.05510121e+03 3.28802141e+03 1.63729272e+02 3.97224691e+02
 8.37397449e+02 2.86083142e+02 6.94848751e+02 5.03366266e+02
 1.16935611e+03 8.14228025e+02 1.20555831e+02 4.78402530e+02
 2.52746721e+02 1.61449018e+02 1.91720775e+02 4.81232091e+02
 1.10237839e+02 4.14689352e+01 3.23763716e+02 3.03701627e+02
 2.89857278e+02 4.58764672e+03 1.75468373e+03 7.49929301e+02
 9.40476913e+02 1.10884112e+03 3.10640456e+02 1.76636127e+04
 6.08350801e+02 2.86406522e+02 4.81792541e+02 7.47246149e+02
 5.77949302e+02 2.76661288e+03 3.15540735e+02 3.05954315e+04
 5.88742315e+03 4.45022815e+02 9.25600003e+02 8.65369349e+02
 4.55085184e+02 3.52275730e+02 2.18182645e+02 1.63109124e+02
 1.80731800e+02 5.12231075e+02 8.00426523e+02 3.99450034e+02
 2.33276756e+02 7.30341490e+02 3.00330389e+02 5.62297827e+02
 1.95895948e+03 4.89318785e+02 5.61329299e+01 3.62133109e+02
 1.93904029e+02 4.42425642e+01 7.08433228e+02 1.04910789e+02
 4.45370393e+01 3.63890226e+02 5.27757115e+01 9.38259711e+02
 1.26231580e+03 1.34433471e+03 1.88452263e+02 4.48417318e+02
 1.18286924e+02 1.61427660e+02 2.42177012e+02 5.86540654e+02
 1.02903169e+02 1.54418518e+01 7.15229535e+01 1.12015364e+02
 1.15294989e+01 1.19331942e+02 7.06127883e+01 7.35705703e+01
 4.25831970e+01 1.40991681e+02 2.40862658e+01 7.51709983e+01
 2.08183540e+02 1.40332667e+02 3.31693940e+01 2.29380667e+02
 1.56391184e+01 1.58736480e+02]


'''
IMG_EXT = ['.jpg']

LBL_EXT = ['.png']
SCALES = [1.0]


class ToLabel:
  def __call__(self, label):
    label = np.array(label)
    return torch.from_numpy(label).long()


def load_image(file):
  return Image.open(file)


def load_label(file):
  return Image.open(file)


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


class MS_COCO(Dataset):
  def __init__(self, root, subset, h, w, means, stds, crop_h=None, crop_w=None):
    self.images_root = os.path.join(root, subset + "2017")
    self.labels_root = os.path.join(root,
                                    "annotations/panoptic_"+subset+"2017_remap")

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
      label = load_label(f).convert('L')

    # resize (resizing is different if we are in train or valid mode)
    # generate resizer
    if self.subset == 'train':
      new_h = self.crop_h
      new_w = self.crop_w
    else:
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
      self.train_dataset = MS_COCO(root=self.location,
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

      self.valid_dataset = MS_COCO(root=self.location,
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
