cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(bonnetal_classification_ros)

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
    bonnetal_classification_msg
    bonnetal_classification_lib
    roscpp
    sensor_msgs
    std_msgs
    cv_bridge
    image_transport
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
    bonnetal_classification_msg
    roscpp
    sensor_msgs
    std_msgs
    cv_bridge
    image_transport
  DEPENDS
)

###########
## Build ##
###########

# explicit include dirs
include_directories(include
                    ${catkin_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${YAML_CPP_INCLUDE_DIR}
                    ${TORCH_INCLUDE_DIRS}
                    ${TENSORRT_INCLUDE_DIRS}
)

# apps
add_executable(bonnetal_classification_node src/node.cpp
                                            src/handler.cpp)
target_link_libraries(bonnetal_classification_node   
                                        ${catkin_LIBRARIES}
                                        ${OpenCV_LIBS}
                                        ${Boost_LIBRARIES}
                                        ${YAML_CPP_LIBRARIES}
                                        ${TORCH_LIBRARIES}
                                        ${TENSORRT_LIBRARIES}
                                        )
add_dependencies(bonnetal_classification_node ${catkin_EXPORTED_TARGETS})

#############
## Install ##
#############

# Mark executables and/or libraries for installation
install(
  TARGETS bonnetal_classification_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

#############
## Testing ##
#############

if(CATKIN_ENABLE_TESTING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  ## Add gtest based cpp test target and link libraries!
endif()