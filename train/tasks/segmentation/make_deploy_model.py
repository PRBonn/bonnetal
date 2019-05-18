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

from tasks.segmentation.modules.traceSaver import *

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./make_deploy_model.py")
  parser.add_argument(
      '--path', '-p',
      type=str,
      required=True,
      help='Directory to get the pretrained model. No default!'
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      required=True,
      help='Directory to put the new model. No default!'
  )
  parser.add_argument(
      '--new_h',
      type=int,
      dest='new_h',
      default=None,
      help='Force Height to. Defaults to %(default)s',
  )
  parser.add_argument(
      '--new_w',
      type=int,
      dest='new_w',
      default=None,
      help='Force Width to. Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("model path", FLAGS.path)
  print("log dir", FLAGS.log)
  print("Height force", FLAGS.new_h)
  print("Width force", FLAGS.new_w)
  print("----------\n")
  print("Commit hash (training version): ", str(
      subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  print("----------\n")

  # does model folder exist?
  if FLAGS.path is not None:
    if os.path.isdir(FLAGS.path):
      print("model folder exists! Using model from %s" % (FLAGS.path))
    else:
      print("model folder doesnt exist!")
      quit()
  else:
    print("No pretrained directory found.")

  # create log folder
  try:
    if os.path.isdir(FLAGS.log):
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

  # create saver and start the exporting
  onnx_maker = TraceSaver(FLAGS.path,
                          FLAGS.log,
                          (FLAGS.new_h, FLAGS.new_w))   # force image properties
  onnx_maker.export()
