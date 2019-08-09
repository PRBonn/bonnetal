#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

from tasks.segmentation.modules.trainer import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./train.py")
  parser.add_argument(
      '--cfg', '-c',
      type=str,
      required=False,
      help='Classification yaml cfg file. See /config for sample. No default!',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default=os.path.expanduser("~") + '/logs/' +
      datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
      help='Directory to put the log data. Default: ~/logs/date+time'
  )
  parser.add_argument(
      '--path', '-p',
      type=str,
      required=False,
      default=None,
      help='Directory to get the pretrained model. If not passed, do from scratch!'
  )
  parser.add_argument(
      '--eval',
      dest='eval',
      default=False,
      action='store_true',
      help='Only evaluate, no training. Defaults to %(default)s',
  )
  parser.add_argument(
      '--no_batchnorm',
      dest='no_batchnorm',
      default=False,
      action='store_true',
      help='Halt batchnorm training, for smaller batch-size fine-tuning. Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("config yaml: ", FLAGS.cfg)
  print("log dir", FLAGS.log)
  print("model path", FLAGS.path)
  print("eval only", FLAGS.eval)
  print("No batchnorm", FLAGS.no_batchnorm)
  print("----------\n")
  print("Commit hash (training version): ", str(
      subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  print("----------\n")

  if FLAGS.path is None and FLAGS.cfg is None:
    print("If no pretrained model is provided, then a cfg file MUST")
    quit()

  # try to open data yaml
  try:
    if(FLAGS.cfg):
      print("Opening config file %s" % FLAGS.cfg)
      f = open(FLAGS.cfg, 'r')
      configfile = FLAGS.cfg
    else:
      print("Opening default data file cfg.yaml from log folder")
      f = open(FLAGS.path + '/cfg.yaml', 'r')
      configfile = FLAGS.path + '/cfg.yaml'
    CFG = yaml.safe_load(f)
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # create log folder
  try:
    if os.path.isdir(FLAGS.log):
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

  # does model folder exist?
  if FLAGS.path is not None:
    if os.path.isdir(FLAGS.path):
      print("model folder exists! Using model from %s" % (FLAGS.path))
    else:
      print("model folder doesnt exist! Start with random weights...")
  else:
    print("No pretrained directory found.")

  # copy all files to log folder (to remember what we did, and make inference
  # easier). Also, standardize name to be able to open it later
  try:
    print("Copying files to %s for further reference." % FLAGS.log)
    copyfile(configfile, FLAGS.log + "/cfg.yaml")
  except Exception as e:
    print(e)
    print("Error copying files, check permissions. Exiting...")
    quit()

  # create trainer and start the training
  trainer = Trainer(CFG, FLAGS.log, FLAGS.path, FLAGS.eval, FLAGS.no_batchnorm)
  trainer.train()
