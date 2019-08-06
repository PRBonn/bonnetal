#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import os
import cv2
import __init__ as booger
from queue import Queue
from threading import Thread, Event
import numpy as np
import shutil

# choice of backends implemented
backend_choices = ["native", "caffe2", "tensorrt", "pytorch"]


class CaptureRunner(Thread):
  def __init__(self, queue, cap, blocking):
    Thread.__init__(self)
    self.queue = queue
    self.cap = cap
    self.blocking = blocking
    self.stopping = False
    self._stop_event = Event()

  def run(self):
    self.stopping = False
    while not self._stop_event.is_set():
      ret = None
      if self.cap.isOpened():
        ret, cv_img = self.cap.read()
      else:
        self.stop()
      if ret:
        if not self.blocking and not self.queue.full():
          self.queue.put_nowait(cv_img)
        elif self.blocking:
          while not self._stop_event.is_set():
            try:
              self.queue.put(cv_img, block=self.blocking, timeout=0.1)
              break
            except:
              pass

  def stop(self):
    self._stop_event.set()
    while(not self.queue.empty()):
      self.queue.get()


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./infer_video.py")
  parser.add_argument(
      '--video',
      type=str,
      required=False,
      default=None,
      help='Path to video. Defaults to Webcam!'
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
      default=1000000000,
      help='workspace size. Defaults to %(default)s'
  )
  parser.add_argument(
      '--verbose', '-v',
      dest='verbose',
      default=False,
      action='store_true',
      help='Verbose mode. Defaults to %(default)s',
  )
  parser.add_argument(
      '--mask', '-m',
      type=int,
      nargs='+',
      required=False,
      default=None,
      help='Optional mask numbers. If given, it prints only sectors containing classes listed.'
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
  print("Video", FLAGS.video)
  print("log dir", FLAGS.log)
  print("model path", FLAGS.path)
  print("backend", FLAGS.backend)
  print("workspace", FLAGS.workspace)
  print("Verbose", FLAGS.verbose)
  print("Mask", FLAGS.mask)
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
  else:
    # check that if log is None, at least I'm verbose:
    if not FLAGS.verbose:
      print("Program needs to either save to log, or be verbose, otherwise I'm just wasting electricity")
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
    cv2.namedWindow("press q to exit", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("press q to exit", 960, 540)

  # open images
  if FLAGS.video is None:
    print("No video specified! Inferring from webcam.")
    cap = cv2.VideoCapture(0)
    blocking = False
  else:
    print("Trying to open video: ", FLAGS.video)
    cap = cv2.VideoCapture(FLAGS.video)
    blocking = True
  if cap is None:
    print("Can't open webcam/video")

  # queue for images
  queue = Queue(1)

  # Create worker thread
  worker = CaptureRunner(queue, cap, blocking)

  # Setting daemon to True will let the main thread exit even though the workers are blocking
  # worker.daemon = True
  worker.start()

  # infer images
  idx = 0
  while not worker.stopping:
    # get image
    try:
      cv_img = queue.get(timeout=1)
    except:
      print("No more frames")
      break

    # order
    if(FLAGS.verbose):
      print("*" * 80)
      print("New frame ", idx)

    # flip!
    if(FLAGS.video is None):
      # flip image if I am in webcam mode
      cv_img = np.ascontiguousarray(np.flip(cv_img, axis=1))

    # infer
    argmax, color = user.infer(cv_img, FLAGS.verbose)

    # alpha colors
    alpha = cv2.addWeighted(cv_img, 1, color, 0.5, 0, dtype=-1)

    # mask?
    if FLAGS.mask is not None:
      # get classes to mask
      if type(FLAGS.mask) is not list:
        mask_categories = [FLAGS.mask]
      else:
        mask_categories = FLAGS.mask

      # make a mask container
      masked_alpha = np.zeros(alpha.shape, dtype=np.uint8)
      masked_color = np.zeros(color.shape, dtype=np.uint8)

      # for each class, get mask, and add to alpha masked
      for cat in mask_categories:
        # get positions where this category is
        positions = argmax == cat

        # put in corresponding image
        masked_alpha[positions] = alpha[positions]
        masked_color[positions] = color[positions]

      # image to show
      stack = np.concatenate([cv_img, masked_alpha, masked_color], axis=1)
    else:
      # image to show
      stack = np.concatenate([cv_img, alpha, color], axis=1)

    # show
    if FLAGS.verbose:
      cv2.imshow("press q to exit", stack)
      ret = cv2.waitKey(1)
      if ret == ord('q') and FLAGS.log is None:
        break

    # save to log
    if FLAGS.log is not None:
      # save
      basename = str(idx) + ".png"
      cv2.imwrite(os.path.join(FLAGS.log, "alpha_" + basename), alpha)
      cv2.imwrite(os.path.join(FLAGS.log, "color_" + basename), color)
      cv2.imwrite(os.path.join(FLAGS.log, "argmax_" + basename), argmax)

    # increment counter
    idx += 1

  worker.stop()
  worker.join()
