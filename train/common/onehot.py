# This file is covered by the LICENSE file in the root of this project.
import torch


def to_one_hot(tensor, nClasses):
  if len(tensor.size()) == 1:
    b = tensor.size(0)
    if tensor.is_cuda:
      one_hot = torch.zeros(b, nClasses, device=torch.device(
          'cuda')).scatter_(1, tensor.unsqueeze(1), 1)
    else:
      one_hot = torch.zeros(b, nClasses).scatter_(1, tensor.unsqueeze(1), 1)
  elif len(tensor.size()) == 2:
    n, b = tensor.size()
    if tensor.is_cuda:
      one_hot = torch.zeros(n, nClasses, b, device=torch.device(
          'cuda')).scatter_(1, tensor.unsqueeze(1), 1)
    else:
      one_hot = torch.zeros(n, nClasses, b).scatter_(
          1, tensor.unsqueeze(1), 1)
  elif len(tensor.size()) == 3:
    n, h, w = tensor.size()
    if tensor.is_cuda:
      one_hot = torch.zeros(n, nClasses, h, w, device=torch.device(
          'cuda')).scatter_(1, tensor.unsqueeze(1), 1)
    else:
      one_hot = torch.zeros(n, nClasses, h, w).scatter_(
          1, tensor.unsqueeze(1), 1)
  return one_hot
