#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import os
import shutil
import numpy as np
import cv2
import __init__ as booger

# choice of backends implemented
backend_choices = ["native", "caffe2", "tensorrt", "pytorch"]

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer_img.py")
  parser.add_argument(
      '--image', '-i',
      nargs='+',
      type=str,
      required=True,
      help='Image to infer. No Default',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default=None,
      help='Directory to put the log data. Default: No saving.'
  )
  parser.add_argument(
      '--path', '-p',
      type=str,
      required=True,
      default=None,
      help='Directory to get the trained model.'
  )
  parser.add_argument(
      '--backend', '-b',
      type=str,
      required=False,
      default=backend_choices[0],
      help='Backend to use to infer. Defaults to %(default)s, and choices:{choices}'.format(
          choices=backend_choices),
  )
  parser.add_argument(
      '--workspace', '-w',
      type=int,
      required=False,
      default=8000000000,
      help='Workspace size for tensorRT. Defaults to %(default)s'
  )
  parser.add_argument(
      '--verbose', '-v',
      dest='verbose',
      default=False,
      action='store_true',
      help='Verbose mode. Defaults to %(default)s',
  )
  parser.add_argument(
      '--calib_images', '-ci',
      nargs='+',
      type=str,
      required=False,
      default=None,
      help='Images to calibrate int8 inference. No Default',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("Image", FLAGS.image)
  print("log dir", FLAGS.log)
  print("model path", FLAGS.path)
  print("backend", FLAGS.backend)
  print("workspace", FLAGS.workspace)
  print("Verbose", FLAGS.verbose)
  print("INT8 Calibration Images", FLAGS.calib_images)
  print("----------\n")
  print("Commit hash: ", str(
      subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  print("----------\n")

  if(FLAGS.calib_images is not None and not isinstance(FLAGS.calib_images, list)):
    FLAGS.calib_images = [FLAGS.calib_images]

  # create log folder
  if FLAGS.log is not None:
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
      print("model folder doesnt exist! Exiting...")
      quit()
  else:
    print("No pretrained directory found")
    quit()

  # check that backend makes sense
  assert(FLAGS.backend in backend_choices)

  # create inference context for the desired backend
  if FLAGS.backend == "tensorrt":
    # import and use tensorRT
    from tasks.segmentation.modules.userTensorRT import UserTensorRT
    user = UserTensorRT(FLAGS.path, FLAGS.workspace, FLAGS.calib_images)
  elif FLAGS.backend == "caffe2":
    # import and use caffe2
    from tasks.segmentation.modules.userCaffe2 import UserCaffe2
    user = UserCaffe2(FLAGS.path)
  elif FLAGS.backend == "pytorch":
    # import and use caffe2
    from tasks.segmentation.modules.userPytorch import UserPytorch
    user = UserPytorch(FLAGS.path)
  else:
    # default to native pytorch
    from tasks.segmentation.modules.user import User
    user = User(FLAGS.path)

  if FLAGS.verbose:
    cv2.namedWindow('predictions', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('predictions', 960, 540)

  # open images
  if type(FLAGS.image) is not list:
    images = [FLAGS.image]
  else:
    images = FLAGS.image

  for img in images:
    # open
    cv_img = cv2.imread(img, cv2.IMREAD_COLOR)

    if cv_img is None:
      print("Can't open ", img)
      continue

    # infer
    print("Inferring ", img)
    mask, color_mask = user.infer(cv_img, FLAGS.verbose)

    # show?
    if FLAGS.verbose:
      # if I want to show, I open a resizeable window
      stack = np.concatenate([cv_img, color_mask], axis=1)
      cv2.imshow("predictions", stack)
      cv2.waitKey(1)

    # save to log
    if FLAGS.log is not None:
      # save
      basename = os.path.basename(img)
      cv2.imwrite(os.path.join(FLAGS.log, "mask_" + basename), mask)
      cv2.imwrite(os.path.join(FLAGS.log, "color_" + basename), color_mask)

  cv2.destroyAllWindows()
