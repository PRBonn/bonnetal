import torch
import torchvision
import torchvision.transforms as transforms


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
      self.location = location
      self.batch_size = batch_size
      self.workers = workers

      # data transforms
      self.transform = transforms.Compose([transforms.Resize((self.img_prop["height"],
                                                              self.img_prop["width"])),
                                           transforms.ToTensor(),
                                           transforms.Normalize(self.img_means,
                                                                self.img_stds), ])

      # train set loader
      trainset = torchvision.datasets.CIFAR10(root=self.location,
                                              train=True,
                                              download=True,
                                              transform=self.transform)
      self.trainloader = torch.utils.data.DataLoader(trainset,
                                                     batch_size=self.batch_size,
                                                     shuffle=True,
                                                     num_workers=workers,
                                                     pin_memory=True)
      self.trainiter = iter(self.trainloader)

      # valid set loader
      validset = torchvision.datasets.CIFAR10(root=self.location,
                                              train=False,
                                              download=True,
                                              transform=self.transform)
      self.validloader = torch.utils.data.DataLoader(validset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=workers,
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
