# This file is covered by the LICENSE file in the root of this project.

import torch.optim.lr_scheduler as toptim


class OneShot_LR(toptim._LRScheduler):
  """ One shot learning rate scheduler.
      Initially, increases the learning rate from 0 to the final value, in a
      certain number of steps. After this number of epochs, each step decreases
      LR linearly to the initial value. In a last stage, the training experiences
      exponential decay with base "post_decay", as is the case with most cnn training.
  """

  def __init__(self, optimizer, base_lr, max_lr, step_size_up, step_size_down, cycle_momentum, base_momentum, max_momentum, post_decay):
    # cyclic params
    self.optimizer = optimizer
    self.initial_lr = base_lr
    self.max_lr = max_lr
    self.step_size_up = step_size_up
    self.step_size_down = step_size_down
    self.cycle_momentum = cycle_momentum
    self.base_momentum = base_momentum
    self.max_momentum = max_momentum
    self.post_decay = post_decay

    # cap to one
    if self.step_size_up < 1:
      self.step_size_up = 1
    if self.step_size_down < 1:
      self.step_size_down = 1

    # cyclic lr
    self.initial_scheduler = toptim.CyclicLR(self.optimizer,
                                             base_lr=self.initial_lr,
                                             max_lr=self.max_lr,
                                             step_size_up=self.step_size_up,
                                             step_size_down=self.step_size_down,
                                             cycle_momentum=self.cycle_momentum,
                                             base_momentum=self.base_momentum,
                                             max_momentum=self.max_momentum)

    # our params
    self.oneshot_n = self.step_size_up + self.step_size_down   # steps to warm up for
    self.finished = False  # am i done
    super().__init__(optimizer)

  def get_lr(self):
    return [self.initial_lr * (self.post_decay ** self.last_epoch) for lr in self.base_lrs]

  def step(self, epoch=None):
    if self.finished or self.initial_scheduler.last_epoch >= self.oneshot_n:
      if not self.finished:
        self.base_lrs = [self.initial_lr for lr in self.base_lrs]
        self.finished = True
      return super(OneShot_LR, self).step(epoch)
    else:
      return self.initial_scheduler.step(epoch)
