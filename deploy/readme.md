# Bonnetal! Deployment/Inference (C++/ROS)

## Description

- Contains C++ libraries and inference nodes to all tasks in order to allow for:
  - Deploy on a robot using ROS
  - Build libraries against existing C++ applications that can do DL inference for all tasks.

---
## Tasks

- [Full-image classification](src/classification).
- [Semantic Segmentation](src/segmentation).
- [Instance Segmentation](src/instances).
- [Object Detection](src/detection).
- [CNN Keypoint/Feature Extraction](src/features)
- [Counting](src/counting)

---
## Dependencies

If your system is still not running after installing all these dependencies, have a look at the [docker files](../docker) we include. We put an effort into making sure the pipeline works there, and constantly update the installed dependencies, but if we forget one, please submit a PR to this readme to help the rest.

#### System dependencies

First you need to install the nvidia driver and CUDA, so have fun!

- CUDA Installation guide: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- Then you can do the other dependencies:

  ```sh
  $ sudo apt-get update 
  $ sudo apt-get install -yqq  build-essential ninja-build \
    python3-dev python3-pip apt-utils curl git cmake unzip autoconf autogen \
    libtool mlocate zlib1g-dev python3-numpy python3-wheel wget \
    software-properties-common openjdk-8-jdk libpng-dev  \
    libxft-dev ffmpeg libboost-all-dev \
    libyaml-cpp-dev
  $ sudo updatedb
  ```

- Then follow the steps to install ROS: [Link](http://wiki.ros.org/ROS/Installation).

#### Python dependencies

- Then install the Python packages needed:

  ```sh
  $ sudo apt install python-empy
  $ sudo pip install catkin_tools trollius # for build
  ```

#### Pytorch 

- `From source`: When building the Bonnetal C++ deployment interface, we had some problems between the provided `libtorch` binaries and the [Dual ABI](https://gcc.gnu.org/onlinedocs/gcc-5.2.0/libstdc++/manual/manual/using_dual_abi.html) introduced in GCC 5.1, so to avoid installing twice with 2 different methods, we recommend building PyTorch from source here too (the version on my GitHub is in sync with the last one supported by our framework).

  ```sh
  $ git clone --recursive https://github.com/tano297/pytorch
  $ cd pytorch
  $ sudo python3 setup.py install
  $ sudo cp -r torch/include/* /usr/local/include
  $ sudo cp -r torch/lib/* /usr/local/lib
  $ sudo cp -r torch/share/* /usr/local/share
  ```

  The last `cp` steps are optional, the same behavior can be achieved in a similar fashion by modifiying a flag before each build:

  ```
  $ CMAKE_PREFIX_PATH=/absolute/path/to/libtorch:$CMAKE_PREFIX_PATH
  ```

#### TensorRT

In order to infer with TensorRT during inference with the C++ libraries:

- Install TensorRT: [Link](https://developer.nvidia.com/tensorrt).

---
## Building

- For standalone use (if no ROS is installed):
  
  ```sh
  $ cd bonnetal/deploy
  $ git clone https://github.com/ros/catkin.git src/catkin
  $ CMAKE_PREFIX_PATH=""
  $ catkin init
  $ catkin build bonnetal_classification_standalone #example package for classification task
  ```
  If you get an error, and your system default is python3 (3.5/3.6) you may need to change the default to avoid breaking the catkin-pkg package:

  ```sh
  $ sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1
  $ sudp update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2
  ```

- Extending an existing catkin workspace (ROS installed and workspace sourced):
  
  ```sh
  $ cd bonnetal/deploy
  $ catkin init
  $ catkin build
  $ source devel/setup.bash
  ```