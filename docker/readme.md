# Bonnetal Docker

This directory contains the docker files used to generate both the `base` image, and the `runtime` image. The `base` image is also pre-built in my dockerhub "tano297/bonnetal:base", because building pytorch from source takes a while and it makes no sense for everybody to have to do it. I will now explain how to use the `runtime` to use bonnetal. These images are also a good point to check for missing dependencies in your system, that I may have forgotten to mention. If it works on the docker container, and it doesn't work in your system, please make sure you try to solve your system issue before generating a github issue.

Once you have a working image, you should be able to use the entirety of bonnetal inside of it, both the training part in Python, and the inference part in C++. The base image has a PyTorch compiled from scratch, ROS Melodic, TensorRT 5.1.2, CUDA 10.1, and Ubuntu 18.04 as a base. CUDA 10.1 requires the 418 driver, so if you don't have this driver, and want to install it, you can build the `base` image from its own Dockerfile.

---
## Known caveats

- If your user ID is not 1000, you may have problems between your host account and the `developer` account that I create inside the container, especially in sharing files between host computer and docker. You can check this with `echo $UID` and change the `runtime/Dockerfile` accordingly in every place the number 1000 appears. 
- The `-v /home/$USER:/home/$USER` part of the `docker run` commands allows you to access your home inside the container in the same path that you have it in your own system. This location, `/home/$USER` is parallel to that of the docker's use `/home/developer`. If your user name is `developer` you will need to modify this in order to get the `bonnetal` folder structure in the home, and from a security standpoint making your entire home accessible inside may not be optimal, so feel free to change this accordingly. If you have permissions problems with `/home/$USER` inside the image, refer to point 1 (check the user and group id for the permissions with `ll -al`, it should say developer if you did it right).
- Docker images are quite large in storage, so you may want to change the default location of the storage to a larger disk, or even an external one (not recommended). This is a nice tutorial: [link](https://linuxconfig.org/how-to-move-docker-s-default-var-lib-docker-to-another-directory-on-ubuntu-debian-linux)

From our experience with Bonnet, as useful as Docker is, it is also sometimes quite nitpicky, so if you find other caveats, submit them as a PR so I add them here.

---
## IF YOU HAVE AN NVIDIA GPU (RECOMMENDED)

#### Installing nvidia-docker

To use this docker images with the GPU you will need to download and install `nvidia-docker` from [here](https://github.com/NVIDIA/nvidia-docker). First you will need [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce). Then, if you have 14.04/16.04/18.04, or Debian Jessie/Stretch, you can use the provided `install-nvidia-docker.sh`:

```sh
$ ./install-nvidia-docker.sh
```

#### Using dockerhub pre-built image

This procedure downloads the `base` image from my dockerhub, and then builds the `runtime` image on top of that.

```sh
# THIS SHOULD BE DONE STANDING IN THE REPO's ROOT DIRECTORY
$ nvidia-docker build -t tano297/bonnetal:runtime -f docker/runtime/Dockerfile .
$ nvidia-docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/developer/.Xauthority -v /home/$USER:/home/$USER --net=host --pid=host --ipc=host tano297/bonnetal:runtime /bin/bash
$ nvidia-smi # check that everything went well
```

#### Using only this repository

This procedure builds both the `base` and `runtime` images. It can be useful if any of the dependencies are not compatible with your system (such as CUDA10.1), but **IT TAKES A WHILE** to build.

```sh
# THIS SHOULD BE DONE STANDING IN THE REPO's ROOT DIRECTORY
$ nvidia-docker build -t tano297/bonnetal:base -f docker/base/Dockerfile .
$ nvidia-docker build -t tano297/bonnetal:runtime -f docker/runtime/Dockerfile .
$ nvidia-docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/developer/.Xauthority -v /home/$USER:/home/$USER --net=host --pid=host --ipc=host tano297/bonnetal:runtime /bin/bash
$ nvidia-smi # check that everything went well
```

---
## If you DON'T have an NVIDIA GPU (Good luck)

If you want to try the framework out in a computer with no GPU, then you can basically do all the same but using the good 'ol docker, which you can just install with:

```sh
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```

#### Using dockerhub pre-built image

This procedure downloads the `base` image from my dockerhub, and then builds the `runtime` image on top of that.

```sh
# THIS SHOULD BE DONE STANDING IN THE REPO's ROOT DIRECTORY
$ docker build -t tano297/bonnetal:runtime -f docker/runtime/Dockerfile .
$ docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/developer/.Xauthority -v /home/$USER:/home/$USER --net=host --pid=host --ipc=host tano297/bonnetal:runtime /bin/bash
$ nvidia-smi # check that everything went well
```

#### Using only this repository

This procedure builds both the `base` and `runtime` images. It can be useful if any of the dependencies are not compatible with your system (such as CUDA10.1), but **IT TAKES A WHILE** to build.

```sh
# THIS SHOULD BE DONE STANDING IN THE REPO's ROOT DIRECTORY
$ docker build -t tano297/bonnetal:base -f docker/base/Dockerfile .
$ docker build -t tano297/bonnetal:runtime -f docker/runtime/Dockerfile .
$ docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/home/developer/.Xauthority -v /home/$USER:/home/$USER --net=host --pid=host --ipc=host tano297/bonnetal:runtime /bin/bash
$ nvidia-smi # check that everything went well
```