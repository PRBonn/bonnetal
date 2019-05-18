# Bonnetal Classification

This readme contains simple usage explanations of how the inference works with the C++ interface, both with PyTorch and TensorRT.
Before you start with the inference, make sure you've read the [readme](../../../train/tasks/classification) in the training part, so that you know how to generate an [inference model](../../../train/tasks/classification#make-inference-model).

![](https://i.ibb.co/J2ryMKD/im-a-tennis-ball.png)
Example of a PhD student being ignored by an algorithm (and why full image classification is a bad idea). This is from a ResNet50 trained on ImageNet (person is not a class, and neither is PhD student :stuck_out_tongue:).

---
## Build

#### Standalone

  ```sh
  $ cd bonnetal/deploy
  $ git clone https://github.com/ros/catkin.git src/catkin
  $ CMAKE_PREFIX_PATH=""
  $ catkin init
  $ catkin build bonnetal_classification_standalone
  ```
  If you get an error, and your system default is python3 (3.5/3.6) you may need to change the default to avoid breaking the catkin-pkg package:

  ```sh
  $ sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
  $ sudp update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
  ```


#### ROS

Extending an existing catkin workspace (ROS installed and workspace sourced):
  
  ```sh
  $ cd bonnetal/deploy
  $ catkin init
  $ catkin build
  $ source devel/setup.bash
  ```
---
## Perform Inference

#### Standalone

- Infer images:

  ```sh
  # use --verbose or -v to get verbose mode
  $ ./devel/lib/bonnetal_classification_standalone/infer_img -h # help
  $ ./devel/lib/bonnetal_classification_standalone/infer_img -p /path/to/save/inference/ready -i /path/to/images # can be multiple images
  $ ./devel/lib/bonnetal_classification_standalone/infer_img -p /path/to/save/inference/ready -i /path/to/images -b pytorch
  $ ./devel/lib/bonnetal_classification_standalone/infer_img -p /path/to/save/inference/ready -i /path/to/images -b tensorrt
  ```
  
- Infer video/webcam:

  ```sh
  # use --verbose or -v to get verbose mode
  $ ./devel/lib/bonnetal_classification_standalone/infer_video -h # help
  $ ./devel/lib/bonnetal_classification_standalone/infer_video -p /path/to/save/inference/ready --video /path/to/video
  $ ./devel/lib/bonnetal_classification_standalone/infer_video -p /path/to/save/inference/ready --video /path/to/video -b pytorch
  $ ./devel/lib/bonnetal_classification_standalone/infer_video -p /path/to/save/inference/ready --video /path/to/video -b tensorrt
  ```


- Infer Webcam instead of video:

  ```sh
    # use --verbose or -v to get verbose mode
    $ ./devel/lib/bonnetal_classification_standalone/infer_video -p /path/to/pretrained/ # simply drop the video and it will look for /dev/video0
  ```

#### ROS

- Example with webcam contains all necessary stuff (see the config file inside the `_ros` package):

  ```sh
  $ roslaunch bonnetal_classification_ros bonnetal_classification_webcam_sample.launch
  ```

