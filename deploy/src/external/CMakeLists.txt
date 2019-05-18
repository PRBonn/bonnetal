cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(bonnetal_external)

# handle debug vs release mode
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

## Use C++11
set (CMAKE_CXX_STANDARD 11)

# check flags
message("Build type: ${CMAKE_BUILD_TYPE}")

## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
)

####################################
## pytorch specific configuration ##
####################################
option(TORCH_FOUND "Torch installed on system?" OFF)

# try to find pytorch c++ libs
find_package(PyTorch PATHS cmake)
if (TORCH_FOUND)
  message("Pytorch SUCCESS!\n")
  set(TORCH_FOUND ON)
else()
  message("Torch NOT FOUND\n")
  set(TORCH_FOUND OFF)
endif (TORCH_FOUND)

####################################
## pytorch specific configuration ##
####################################
# this package libs and includes for TensorRT
option(TENSORRT_FOUND "TensorRT installed on system?" OFF)

# first try to find cuda and tensorrt-related stuff (nvinfer, onnx, etc)
find_package(TensorRT PATHS cmake)
if (TENSORRT_FOUND)
  message("TensorRT SUCCESS!\n")
  set(TENSORRT_FOUND ON)
else()
  message("TensorRT NOT FOUND\n")
  set(TENSORRT_FOUND OFF)
endif (TENSORRT_FOUND)

# complain if no backend is installed
if(NOT TORCH_FOUND AND NOT TENSORRT_FOUND)
  message(FATAL_ERROR
  "At least one of Pytorch, or TensorRT must be installed
  TORCH_FOUND ${TORCH_FOUND}
  TENSORRT_FOUND ${TENSORRT_FOUND}\n")
endif()

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    bonnetal_external
  CATKIN_DEPENDS
  DEPENDS
  CFG_EXTRAS
    pytorch-config.cmake
    tensorrt-config.cmake
)

#########################
## Install cmake finds ##
#########################
install(DIRECTORY cmake
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

###########################
## Install header config ##
###########################
include_directories(
  include
	${catkin_INCLUDE_DIRS}
)

# turn config file into flags (sets TENSORRT_FOUND and TORCH_FOUND)
configure_file(include/external.hpp.in ${PROJECT_SOURCE_DIR}/include/external.hpp @ONLY)


# Mark cpp header files for installation
install(
  DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)

# make wrapper lib
add_library(bonnetal_external src/external.cpp)


#################
## Install lib ##
#################

# Mark executables and/or libraries for installation
install(
  TARGETS bonnetal_external
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)