#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import yaml
import os
import cv2
import numpy as np


def is_image(filename):
  return any(filename.endswith(ext) for ext in ['.jpg', '.png'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./train.py")
  parser.add_argument(
      '--cfg', '-c',
      type=str,
      required=False,
      help='Yaml cfg file. See /config for sample. No default!',
  )
  parser.add_argument(
      '--image', '-i',
      type=str,
      required=True,
      default=None,
      help='Directory to get the images from. If not passed, do from scratch!'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("config yaml: ", FLAGS.cfg)
  print("image dir", FLAGS.image)
  print("----------\n")
  print("Commit hash (training version): ", str(
      subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  print("----------\n")

  # try to open data yaml
  try:
    if(FLAGS.cfg):
      print("Opening config file %s" % FLAGS.cfg)
      with open(FLAGS.cfg, 'r') as file:
        CFG = yaml.safe_load(file)
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # create list of images and examine their pixel values
  filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(FLAGS.image)) for f in fn if is_image(f)]

  # examine individually pixel values
  counter = 0.0
  pix_val = np.zeros(3, dtype=np.float)
  for filename in filenames:
    # analize
    print("Accumulating mean", filename)

    # open as rgb
    cv_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    
    # normalize to 1
    cv_img = cv_img.astype(np.float) / 255.0

    # count pixels and add them to counter
    h, w, d = cv_img.shape
    counter += h * w

    # sum to moving pix value counter in each channel
    pix_val += np.sum(cv_img, (0, 1))

  # calculate means
  means = (pix_val / counter)

  # means
  print("means(rgb): ", means)

  # pass again and calculate variance
  pix_var = np.zeros(3, dtype=np.float)
  for filename in filenames:
    # analizel
    print("Accumulating variance", filename)

    # open as rgb
    cv_img = cv2.imread(filename, cv2.IMREAD_COLOR)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    # normalize to 1
    cv_img = cv_img.astype(np.float) / 255.0

    # sum to moving pix value counter in each channel
    pix_var += np.sum(np.square(cv_img - means), (0, 1))

  # calculate the standard deviations
  stds = np.sqrt(pix_var / counter)
  print("stds(rgb): ", stds)

  if FLAGS.cfg:
    # save in cfg file
    CFG["dataset"]["img_means"] = means.tolist()
    CFG["dataset"]["img_stds"] = stds.tolist()

    try:
      with open(FLAGS.cfg, 'w') as file:
        yaml.dump(CFG, file)
    except Exception as e:
      print(e)
      print("Error saving to yaml file.")
      quit()

  # finalize by printing both
  print("*" * 80)
  print("means(rgb): ", means)
  print("stds(rgb): ", stds)
  print("*" * 80)
