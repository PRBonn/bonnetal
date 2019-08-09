#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import yaml
import os
import cv2
import numpy as np
import json
import shutil

LBL_EXT = ['.png']
IMG_EXT = ['.jpg', ]
SUBSETS = ['panoptic_train2017', 'panoptic_val2017']


def is_label(filename):
  return any(filename.endswith(ext) for ext in LBL_EXT)


def is_image(filename):
  return any(filename.endswith(ext) for ext in IMG_EXT)


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./generate_gt.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      default=None,
      help='Directory to get the dataset from.'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset dir", FLAGS.dataset)
  print("----------\n")
  print("Commit hash (training version): ", str(
      subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  print("----------\n")

  # try to open data yaml
  try:
    print("Opening config file")
    with open('cfg.yaml', 'r') as file:
      CFG = yaml.safe_load(file)
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # make lut for remap
  max_val = 0
  for key, val in CFG["int_mapping"].items():
    if val > max_val:
      max_val = val
  assert(max_val < 256)
  LUT = np.zeros(max_val + 1, dtype=np.uint8)
  for key, val in CFG["int_mapping"].items():
    LUT[val] = key  # inverse mapping for cross entropy

  for subset in SUBSETS:
    SUBSET_JSON = os.path.join(FLAGS.dataset, "annotations", subset + ".json")
    SUBSET_DIR = os.path.join(FLAGS.dataset, "annotations", subset)
    print("Getting labels from: ")
    print(SUBSET_JSON)
    print(SUBSET_DIR)

    # now make the remap directory
    REMAP_DIR = os.path.join(FLAGS.dataset, "annotations", subset + "_remap")
    print("putting labels in")
    print(REMAP_DIR)
    # create folder
    try:
      if os.path.isdir(REMAP_DIR):
        shutil.rmtree(REMAP_DIR)
      os.makedirs(REMAP_DIR)
    except Exception as e:
      print(e)
      print("Error creating", REMAP_DIR, ". Check permissions!")
      quit()

    # open json file
    with open(SUBSET_JSON) as file:
      subset_json = json.load(file)
      subset_labels = subset_json["annotations"]
      for label in subset_labels:
        label_file = os.path.join(SUBSET_DIR, label["file_name"])
        # open file BGR
        print("Getting instance image from ", label_file)
        lbl = cv2.imread(label_file, cv2.IMREAD_COLOR)
        B, G, R = cv2.split(lbl)
        B = B.astype(np.uint32)
        G = G.astype(np.uint32)
        R = R.astype(np.uint32)

        # show for debugging purposes
        # cv2.imshow("label", lbl)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # turn to instance number
        inst = R + G * 256 + B * (256**2)
        inst_remapped = np.zeros_like(inst)
        # cv2.imshow("inst", inst.astype(np.float32) / inst.max())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # make remap from instances in label
        for segment in label["segments_info"]:
          # get categories
          idx = segment["id"]
          cat_idx = segment["category_id"]
          # mask for segment
          mask = inst == idx
          # put in new label
          inst_remapped[mask] = cat_idx

        # conver to uint8
        inst_remapped = inst_remapped.astype(np.uint8)

        # pass to my class defition
        inst_remapped = LUT[inst_remapped]

        # show
        # cv2.imshow("inst_remapped", inst_remapped)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # now save in remapped directory
        path_remapped = os.path.join(REMAP_DIR, label["file_name"])
        cv2.imwrite(path_remapped, inst_remapped)
