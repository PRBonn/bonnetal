/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */
#pragma once

// ROS
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>

// Network
#include <selector.hpp>

/*!
 * Main class for the node to handle the ROS interfacing.
 */
class Handler {
 public:
  /*!
   * Constructor.
   *
   * @param      nodeHandle  the ROS node handle.
   */
  Handler(ros::NodeHandle& nodeHandle);

  /*!
   * Destructor.
   */
  virtual ~Handler();

 private:
  /*!
   * Reads and verifies the ROS parameters.
   *
   * @return     true if successful.
   */
  bool readParameters();

  /*!
   * ROS topic callback method.
   *
   * @param[in]  img_msg  The image message (to infer)
   */
  void imageCallback(const sensor_msgs::ImageConstPtr& img_msg);

  //! ROS node handle.
  ros::NodeHandle& _node_handle;

  //! ROS topic subscribers and publishers.
  image_transport::ImageTransport _it;
  image_transport::Subscriber _img_subscriber;
  image_transport::Publisher _argmax_publisher;
  image_transport::Publisher _color_publisher;
  image_transport::Publisher _blend_publisher;

  //! ROS topic names to subscribe to.
  std::string _input_image_topic;
  std::string _argmax_topic;
  std::string _color_topic;
  std::string _blend_topic;

  //! CNN related stuff
  std::unique_ptr<bonnetal::segmentation::Net> _net;
  std::string _model_path;
  bool _verbose;
  std::string _backend;
};
