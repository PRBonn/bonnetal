# Use an official nvidia runtime as a parent image
FROM nvidia/cuda:10.1-devel-ubuntu18.04

# who am I
MAINTAINER Andres Milioto <amilioto@uni-bonn.de>

CMD ["bash"]

# ENVIRONMENT STUFF FOR CUDA
RUN ls /usr/local/cuda/bin
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
ENV PATH=/usr/local/cuda/bin:$PATH
ENV CUDA_ROOT /usr/local/cuda

# set working directory
WORKDIR /bonnetal-docker-base

# apt packages
RUN apt-get update && apt-get install -yqq  build-essential ninja-build \
  python3-dev python3-pip tig apt-utils curl git cmake unzip autoconf autogen \
  libtool mlocate zlib1g-dev python python3-numpy python3-wheel wget \
  software-properties-common openjdk-8-jdk libpng-dev  \
  libxft-dev vim meld sudo ffmpeg python3-pip libboost-all-dev \
  libyaml-cpp-dev -y && updatedb


# python packages
RUN pip3 install -U pip
RUN pip3 install numpy==1.14.0 \
  onnx==1.5.0 \
  torchvision==0.2.2.post3 \
  pycuda==2018.1.1 \
  opencv_python==3.4.0.12 \
  scipy==0.19.1 \
  Pillow==6.0.0 \
  genpy==2016.1.3 \
  scikit_learn==0.20.3 \
  tensorflow==1.13.1 \
  PyYAML==5.1

# tensorrt (Deb local)
RUN wget http://ipb.uni-bonn.de/html/projects/bonnetal/external/nv-tensorrt-repo-ubuntu1804-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb && \
  dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb && \
  rm nv-tensorrt-repo-ubuntu1804-cuda10.1-trt5.1.5.0-ga-20190427_1-1_amd64.deb && \
  apt-get update && \
  apt install tensorrt python3-libnvinfer-dev -yqq


# build pytorch from source and install in the system
RUN git clone --recursive https://github.com/tano297/pytorch
RUN cd pytorch && python3 setup.py install && cd ..
RUN cd pytorch && \
  cp -r torch/include/* /usr/local/include && \
  cp -r torch/lib/* /usr/local/lib && \
  cp -r torch/share/* /usr/local/share && \
  cd .. && rm -r pytorch

# install ros
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
  apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116 && \
  apt update && \ 
  DEBIAN_FRONTEND=noninteractive apt install -yqq ros-melodic-desktop-full && \
  rosdep init && \
  rosdep update

# install catkin tools
RUN apt-get install -y python-pip python-empy 
RUN pip install -U pip catkin-tools trollius

# recommended from nvidia to use the cuda devices
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# clean the cache
RUN apt update && \
  apt autoremove --purge -y && \
  apt clean -y

RUN rm -rf /var/lib/apt/lists/*