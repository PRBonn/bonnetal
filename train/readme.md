# Bonnetal! Training (Python)

## Description

- Contains Python inferfaces to all tasks in order to allow for:
  - Training of included/novel architectures on included/novel datasets.
  - Evaluate the accuracy on such datasets.
  - Generate nice paper images :smile:
  - Generate inference-ready models that can be picked up by our [deployment interface](../deploy)

---
## Tasks

- [Full-image classification](tasks/classification).
- [Semantic Segmentation](tasks/segmentation).

---
## Dependencies

These are the requirements for running the entirety of the training interface (all tasks). If your system is still not running after installing all these dependencies, have a look at the [docker files](../docker) we include. We put an effort into making sure the pipeline works there, and constantly update the installed dependencies, but if we forget one, please submit a PR to this readme to help the rest.

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
    libxft-dev ffmpeg
  $ sudo updatedb
  ```

#### Python dependencies

- Then install the Python packages needed:

  ```sh
  $ sudo pip3 install -U pip
  $ sudo pip3 install numpy==1.14.0 \
    onnx==1.5.0 \
    torchvision==0.2.2.post3 \
    opencv_python==3.4.0.12 \
    scipy==0.19.1 \
    Pillow==6.0.0 \
    genpy==2016.1.3 \
    scikit_learn==0.20.3 \
    tensorflow==1.13.1 \
    PyYAML==5.1
  $ export PYTHONPATH=/usr/local/lib/python3.6/dist-packages/cv2/:$PYTHONPATH # Needed if you have ROS installed (you may want to put it in your .bashrc)
  ```

#### Pytorch 

- `From source`: When building the Bonnetal C++ deployment interface, we had some problems between the provided `libtorch` binaries and the [Dual ABI](https://gcc.gnu.org/onlinedocs/gcc-5.2.0/libstdc++/manual/manual/using_dual_abi.html) introduced in GCC 5.1, so to avoid installing twice with 2 different methods, we recommend building PyTorch from source here too (the version on my GitHub is in sync with the last one supported by our framework).

  ```sh
  $ git clone --recursive https://github.com/tano297/pytorch
  $ cd pytorch
  $ sudo python3 setup.py install
  ```

- `(optionally) From pip:` [Link](https://pytorch.org/)

#### TensorRT (optional)

In order to test if the models will work with TensorRT during inference with the C++ libraries, the Python interface also provides a way to infer using TensorRT with an homologous pipeline. If you want to use this interface you need to:

- Install TensorRT: [Link](https://developer.nvidia.com/tensorrt). 
- Install PyCUDA: [Link](https://pypi.org/project/pycuda/), [Help!](http://alisonrowland.com/articles/installing-pycuda-via-pip).