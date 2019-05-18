cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(bonnetal_classification_lib)

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
  bonnetal_external
)

# get opencv (required)
find_package(OpenCV)
if(OpenCV_FOUND)
  message("OpenCV Libs: ${OpenCV_LIBS}")
  set(OpenCV_LIBRARIES ${OpenCV_LIBS})
  message("OpenCV Headers: ${OpenCV_INCLUDE_DIRS}\n")
else()
  message(FATAL_ERROR "OpenCV NOT FOUND\n")
endif()

# libyaml-cpp (required)
find_package (yaml-cpp)
if (yaml-cpp_FOUND)
  message("YAML Libs: ${YAML_CPP_LIBRARIES}")
  message("YAML Headers: ${YAML_CPP_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "yaml-cpp NOT FOUND\n")
endif (yaml-cpp_FOUND)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
    bonnetal_classification_lib
  CATKIN_DEPENDS
    bonnetal_external
  DEPENDS
    OpenCV
    YAML_CPP
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${bonnetal_external_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${TORCH_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIRS}
)

## Declare a cpp library
# define the network library (compiling tf and trt conditionally)
set(bonnetal_classification_lib_SOURCES src/net.cpp src/selector.cpp)

if(TORCH_FOUND)
  message("Building Pytorch")
  list(APPEND bonnetal_classification_lib_SOURCES src/netPytorch.cpp)
endif()
if(TENSORRT_FOUND)
  message("Building TensorRT")
  list(APPEND bonnetal_classification_lib_SOURCES src/netTensorRT.cpp)
endif()

# make the library and link stuff to it
add_library(bonnetal_classification_lib ${bonnetal_classification_lib_SOURCES})
target_link_libraries(bonnetal_classification_lib ${catkin_LIBRARIES}
                                                  ${bonnetal_external_LIBRARIES}
                                                  ${OpenCV_LIBS}
                                                  ${YAML_CPP_LIBRARIES}
                                                  ${TORCH_LIBRARIES}
                                                  ${TENSORRT_LIBRARIES})

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS bonnetal_classification_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(
  DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)