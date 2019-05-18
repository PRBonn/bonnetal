#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


import argparse
import subprocess
import datetime
import os
import shutil
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
      default=1000000000,
      help='Workspace size for tensorRT. Defaults to %(default)s'
  )
  parser.add_argument(
      '--topk', '-k',
      type=int,
      required=False,
      default=1,
      help='Top predictions. Defaults to %(default)s'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("Image", FLAGS.image)
  print("model path", FLAGS.path)
  print("backend", FLAGS.backend)
  print("workspace", FLAGS.workspace)
  print("topk", FLAGS.topk)
  print("----------\n")
  print("Commit hash: ", str(
      subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
  print("----------\n")

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
    from tasks.classification.modules.userTensorRT import UserTensorRT
    user = UserTensorRT(FLAGS.path, FLAGS.workspace)
  elif FLAGS.backend == "caffe2":
    # import and use caffe2
    from tasks.classification.modules.userCaffe2 import UserCaffe2
    user = UserCaffe2(FLAGS.path)
  elif FLAGS.backend == "pytorch":
    # import and use caffe2
    from tasks.classification.modules.userPytorch import UserPytorch
    user = UserPytorch(FLAGS.path)
  else:
    # default to native pytorch
    from tasks.classification.modules.user import User
    user = User(FLAGS.path)

  # cv2 window that can be resized
  cv2.namedWindow('predictions', cv2.WINDOW_NORMAL)
  cv2.resizeWindow('predictions', 960, 540)

  # open images
  if type(FLAGS.image) is not list:
    images = [FLAGS.image]
  else:
    images = FLAGS.image

  for img in images:
    # order
    print("*" * 80)

    # open
    cv_img = cv2.imread(img, cv2.IMREAD_COLOR)

    if cv_img is None:
      print("Can't open ", img)
      continue

    # infer
    print("Inferring ", img)
    max_class, max_class_str = user.infer(cv_img, FLAGS.topk)

    # make string from classes
    h, w, d = cv_img.shape
    for i, c in enumerate(max_class_str, 1):
      # put text in frame to show
      watermark = "[" + str(i) + "]: " + c
      font_size, _ = cv2.getTextSize(watermark, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                     fontScale=0.75, thickness=1)
      cv2.putText(cv_img, watermark,
                  org=(10, h - 10 -
                       (2 * (len(max_class_str) - i) * font_size[1])),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                  fontScale=0.75,
                  color=(255, 255, 255),
                  thickness=1,
                  lineType=cv2.LINE_AA,
                  bottomLeftOrigin=False)

    # Display the resulting frame
    cv2.imshow('predictions', cv_img)
    ret = cv2.waitKey(0)
    if ret == ord('q') or ret == 27:
      break
    else:
      continue
