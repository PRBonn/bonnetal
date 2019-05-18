# This file is covered by the LICENSE file in the root of this project.


import torch
import numpy as np


def random_projection(mat, dim=3):
  """
   Projects a matrix of dimensions m x oldim to m x dim using a random
   projection
  """
  # include has to be here for multiprocessing problems
  from sklearn import random_projection as rp

  if mat.is_cuda:
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")

  # project
  m, oldim = mat.shape
  t = rp.GaussianRandomProjection()
  proj_mat = torch.tensor(t._make_random_matrix(dim, oldim))
  proj_mat = proj_mat.to(device)
  output = torch.matmul(mat.double(), proj_mat.t())

  # check new dim
  assert(output.shape[0] == m)
  assert(output.shape[1] == dim)

  return output
