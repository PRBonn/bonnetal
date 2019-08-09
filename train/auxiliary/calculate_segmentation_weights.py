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
      '--labels', '-l',
      type=str,
      required=True,
      default=None,
      help='Directory to get the images from. No default!'
  )
  parser.add_argument(
      '--numclasses', '-n',
      type=int,
      required=True,
      help='Number of classes. No default!',
  )
  parser.add_argument(
      '--extrafilter', '-e',
      type=str,
      required=False,
      default=None,
      help="Extra filter for label name (for example, cityscapes contains TrainIds"
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("config yaml: ", FLAGS.cfg)
  print("label dir", FLAGS.labels)
  print("Number of classes", FLAGS.numclasses)
  print("Extra filter ", FLAGS.extrafilter)
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
        FLAGS.numclasses = len(CFG["dataset"]["labels"])
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  if FLAGS.numclasses is None:
    print("Either --numclasses option or cfg file must be provided to get number of classes")
    quit()

  # create list of images and examine their pixel values
  filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(FLAGS.labels)) for f in fn if is_image(f)]

  # examine individually pixel values
  counter = 0
  frequencies = np.zeros(FLAGS.numclasses, dtype=np.uint32)

  # filter names
  if FLAGS.extrafilter is not None:
    filenames = [name for name in filenames if FLAGS.extrafilter in name]

  for filename in filenames:
    # analize
    print("Accumulating class value count for ", filename)

    # open
    cv_img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

    # count pixels and add them to counter
    h, w = cv_img.shape
    counter += h * w

    # sum to moving pix count
    hist = cv2.calcHist([cv_img], [0], None,
                        [FLAGS.numclasses], [0, FLAGS.numclasses]).squeeze(1).astype(np.uint32)
    # print(hist)

    frequencies += hist

  # make frequencies
  print("*" * 80)
  print("Num of pixels: ", counter)
  frequencies = frequencies / float(counter)
  print("Frequency: ", frequencies)
  print("*" * 80)

  # make weights from pix numbers
  # log
  e = 1.02
  weights = 1 / np.log((frequencies + e))
  print("Log strategy")
  print("Weights: ", weights)

  # linear (1-freq)
  weights = (1 - frequencies)
  print("Linear strategy")
  print("Weights: ", weights)

  # (1-freq)^2
  weights = (1 - frequencies) ** 2
  print("Squared strategy")
  print("Weights: ", weights)

  # 1/freq
  e = 1e-8
  weights = 1 / (frequencies + e)
  print("1/w strategy")
  print("Weights: ", weights)
