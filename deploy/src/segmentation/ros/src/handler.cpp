/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */
// STD
#include <unistd.h>
#include <string>

// net stuff
#include "handler.hpp"

/*!
 * Constructor.
 *
 * @param      nodeHandle  the ROS node handle.
 */
Handler::Handler(ros::NodeHandle& nodeHandle)
    : _node_handle(nodeHandle), _it(nodeHandle) {
  // Try to read the necessary parameters
  if (!readParameters()) {
    ROS_ERROR("Could not read parameters.");
    ros::requestShutdown();
  }

  // Subscribe to images to infer
  ROS_INFO("Subscribing to image topics.");
  _img_subscriber =
      _it.subscribe(_input_image_topic, 1, &Handler::imageCallback, this);

  // Advertise our topics
  ROS_INFO("Advertising our outputs.");
  // Advertise our topics
  _argmax_publisher = _it.advertise(_argmax_topic, 1);
  _color_publisher = _it.advertise(_color_topic, 1);
  _blend_publisher = _it.advertise(_blend_topic, 1);

  ROS_INFO("Generating CNN and setting verbosity");
  // create a network
  _net = bonnetal::segmentation::make_net(_model_path, _backend);
  // set verbosity
  _net->verbosity(_verbose);

  ROS_INFO("Successfully launched node.");
}

/*!
 * Destructor.
 */
Handler::~Handler() {}

bool Handler::readParameters() {
  if (!_node_handle.getParam("model_path", _model_path) ||
      !_node_handle.getParam("verbose", _verbose) ||
      !_node_handle.getParam("backend", _backend) ||
      !_node_handle.getParam("input_image", _input_image_topic) ||
      !_node_handle.getParam("output_argmax", _argmax_topic) ||
      !_node_handle.getParam("output_color", _color_topic) ||
      !_node_handle.getParam("output_blend", _blend_topic))
    return false;
  return true;
}

void Handler::imageCallback(const sensor_msgs::ImageConstPtr& img_msg) {
  if (_verbose) {
    // report that we got something
    ROS_INFO("Image received.");
    ROS_INFO("Image encoding: %s", img_msg->encoding.c_str());
  }

  // Only do inference if somebody is subscribed (save energy and temp)
  uint32_t argmax_subs = _argmax_publisher.getNumSubscribers();
  uint32_t color_subs = _color_publisher.getNumSubscribers();
  uint32_t blend_subs = _blend_publisher.getNumSubscribers();
  uint32_t total_subs = argmax_subs + color_subs + blend_subs;

  if (_verbose) {
    std::cout << "Subscribers:  " << std::endl
              << "Argmax: " << argmax_subs << std::endl
              << "Color: " << color_subs << std::endl
              << "Blend: " << blend_subs << std::endl;
  }

  if (total_subs > 0) {
    // Get the image
    cv_bridge::CvImageConstPtr cv_img;
    cv_img = cv_bridge::toCvShare(img_msg);

    // change to bgr according to encoding
    cv::Mat cv_img_bgr(cv_img->image.rows, cv_img->image.cols, CV_8UC3);
    ;
    if (img_msg->encoding == "bayer_rggb8") {
      if (_verbose) ROS_INFO("Converting BAYER_RGGB8 to BGR for CNN");
      cv::cvtColor(cv_img->image, cv_img_bgr, cv::COLOR_BayerBG2BGR);
    } else if (img_msg->encoding == "bgr8") {
      if (_verbose) ROS_INFO("Converting BGR8 to BGR for CNN");
      cv_img_bgr = cv_img->image;
    } else if (img_msg->encoding == "rgb8") {
      if (_verbose) ROS_INFO("Converting RGB8 to BGR for CNN");
      cv::cvtColor(cv_img->image, cv_img_bgr, cv::COLOR_RGB2BGR);
    } else {
      if (_verbose) ROS_ERROR("Colorspace conversion non implemented. Skip...");
      return;
    }

    // infer
    cv::Mat argmax = _net->infer(cv_img_bgr);

    if (argmax_subs > 0) {
      // Send the argmax (changing it to depth 16, since 32 can't be done)
      cv::Mat argmax_16;
      argmax.convertTo(argmax_16, CV_16UC1);
      sensor_msgs::ImagePtr argmax_msg =
          cv_bridge::CvImage(img_msg->header, "mono16", argmax_16).toImageMsg();
      _argmax_publisher.publish(argmax_msg);
    }

    if (color_subs > 0 || blend_subs > 0) {
      // get color
      cv::Mat color_mask = _net->color(argmax);

      if (color_subs > 0) {
        // Send the color mask
        sensor_msgs::ImagePtr color_msg =
            cv_bridge::CvImage(img_msg->header, "bgr8", color_mask)
                .toImageMsg();
        _color_publisher.publish(color_msg);
      }

      if (blend_subs > 0) {
        // get color
        cv::Mat blend_mask = _net->blend(cv_img_bgr, color_mask);

        // Send the alpha blend
        sensor_msgs::ImagePtr blend_msg =
            cv_bridge::CvImage(img_msg->header, "bgr8", blend_mask)
                .toImageMsg();
        _blend_publisher.publish(blend_msg);
      }
    }
  }
}