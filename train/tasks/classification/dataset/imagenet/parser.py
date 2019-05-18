import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Parser():
  # standard conv, BN, relu
  def __init__(self, img_prop, img_means, img_stds, classes, train, location=None, batch_size=None, workers=2):
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
      self.workers = workers

      # Data loading code
      self.normalize = transforms.Normalize(mean=self.img_means,
                                            std=self.img_stds)

      self.jitter = transforms.ColorJitter(brightness=0.05,
                                           contrast=0.05,
                                           saturation=0.05,
                                           hue=0.05)

      self.affine = transforms.RandomAffine(degrees=7,
                                            fillcolor=0)

      # before cropping, resize smallest edge to allow only one dimension of sliding
      size = max([self.img_prop["height"], self.img_prop["width"]])
      self.resize = transforms.Resize(size)

      self.train_dataset = datasets.ImageFolder(self.location + '/train',
                                                transforms.Compose([
                                                    self.jitter,
                                                    self.affine,
                                                    self.resize,
                                                    transforms.transforms.CenterCrop(
                                                        size=(self.img_prop["height"],
                                                              self.img_prop["width"])),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    self.normalize]))

      self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     num_workers=self.workers,
                                                     pin_memory=True)

      # print("classes", self.train_dataset.classes)
      # print("class_to_idx", self.train_dataset.class_to_idx)
      # print("imgs", self.train_dataset.imgs)

      self.trainiter = iter(self.trainloader)

      self.valid_dataset = datasets.ImageFolder(self.location + '/val',
                                                transforms.Compose([
                                                    self.resize,
                                                    transforms.transforms.CenterCrop(
                                                        size=(self.img_prop["height"],
                                                              self.img_prop["width"])),
                                                    transforms.ToTensor(),
                                                    self.normalize]))

      self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=False,
                                                     num_workers=self.workers,
                                                     pin_memory=True)

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
