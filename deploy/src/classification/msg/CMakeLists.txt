cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(bonnetal_classification_msg)

## Find catkin macros and libraries
find_package(catkin REQUIRED
  COMPONENTS
    message_generation
    sensor_msgs
    std_msgs
)

###########
## Build ##
###########

# make mesage
add_message_files(FILES classification.msg)
generate_messages(DEPENDENCIES std_msgs sensor_msgs)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS
  LIBRARIES
  CATKIN_DEPENDS
  DEPENDS
    message_runtime
    sensor_msgs
    std_msgs
)