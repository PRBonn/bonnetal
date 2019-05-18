# This file is covered by the LICENSE file in the root of this project.


def jaccard(box_a, box_b):
  """
  Compute the jaccard overlap (IoU of two boxes) of two sets of boxes
  Args:
    box_a: (tensor) Shape(num_objects, 4) with tl_x, tl_y, br_x, br_y
      tl = topleft
      br = bottomright
    box_b: (tensor) Shape(num_priors, 4) with tl_x, tl_y, br_x, br_y
  Return:
    jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
  """

  # make sure, boxes are given in float values
  box_a = box_a.float()
  box_b = box_b.float()

  inter = intersect(box_a, box_b)
  area_a = ((box_a[:, 2] - box_a[:, 0]) *
            (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)
  area_b = ((box_b[:, 2] - box_b[:, 0]) *
            (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)
  union = area_a + area_b - inter
  return inter / union


def intersect(box_a, box_b):
  """
  Compute the intersection of two sets of boxes
  Args:
    box_a: (tensor) Shape(num_objects, 4) with tl_x, tl_y, br_x, br_y
      tl = topleft
      br = bottomright
    box_b: (tensor) Shape(num_priors, 4) with tl_x, tl_y, br_x, br_y
  Return:
    intersection: (tensor) Shape: [box_a.size(0), box_b.size(0)]
  """
  A = box_a.shape[0]
  B = box_b.shape[0]
  max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                     box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
  min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                     box_b[:, :2].unsqueeze(0).expand(A, B, 2))
  inter = torch.clamp((max_xy - min_xy), min=0)
  return inter[:, :, 0] * inter[:, :, 1]
