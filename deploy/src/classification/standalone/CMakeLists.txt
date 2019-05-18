cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(bonnetal_classification_standalone)

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
    bonnetal_classification_lib
)

# get opencv and yaml-cpp
find_package(OpenCV)
find_package(yaml-cpp)

# get boost (required)
find_package(Boost COMPONENTS program_options filesystem)
if(Boost_FOUND)
  message("Boost Libs: ${Boost_LIBRARIES}")
  message("Boost Headers: ${Boost_INCLUDE_DIRS}\n")
else()
  message(FATAL_ERROR "Boost NOT FOUND\n")
endif()

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES
  CATKIN_DEPENDS
    bonnetal_classification_lib
  DEPENDS
)

###########
## Build ##
###########

# explicit include dirs
include_directories(include
                    ${catkin_INCLUDE_DIRS}
                    ${bonnetal_classification_lib_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${YAML_CPP_INCLUDE_DIR}
                    ${TORCH_INCLUDE_DIRS}
                    ${TENSORRT_INCLUDE_DIRS}
)

# apps
add_executable(infer_img src/infer_img.cpp)
target_link_libraries(infer_img   ${catkin_LIBRARIES}
                                  ${bonnetal_classification_lib_LIBRARIES}
                                  ${OpenCV_LIBS}
                                  ${Boost_LIBRARIES}
                                  ${YAML_CPP_LIBRARIES}
                                  ${TORCH_LIBRARIES}
                                  ${TENSORRT_LIBRARIES}
                                  )

add_executable(infer_video src/infer_video.cpp)
target_link_libraries(infer_video   ${catkin_LIBRARIES}
                                    ${bonnetal_classification_lib_LIBRARIES}
                                    ${OpenCV_LIBS}
                                    ${Boost_LIBRARIES}
                                    ${YAML_CPP_LIBRARIES}
                                    ${TORCH_LIBRARIES}
                                    ${TENSORRT_LIBRARIES}
                                    )


#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS infer_img infer_video
  DESTINATION bin/
)