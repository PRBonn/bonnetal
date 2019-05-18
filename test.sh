#!/bin/bash

# env vars
export PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

# source workspace
source /opt/ros/melodic/setup.bash

# build
cd /bonnetal/deploy/
pwd
catkin init
catkin build