/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */

#include <ros/ros.h>
#include "handler.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "bonnetal_classification_node");
  ros::NodeHandle nodeHandle("~");

  // init handler
  Handler h(nodeHandle);

  ros::spin();
  return 0;
}
