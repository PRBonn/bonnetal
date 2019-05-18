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

// classification message
#include <bonnetal_classification_msg/classification.h>

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
  _output_publisher =
      _node_handle.advertise<bonnetal_classification_msg::classification>(
          _output_topic, 1);

  ROS_INFO("Generating CNN and setting verbosity");
  // create a network
  _net = bonnetal::classification::make_net(_model_path, _backend);
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
      !_node_handle.getParam("output_msg", _output_topic))
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
  uint32_t argmax_subs = _output_publisher.getNumSubscribers();

  if (_verbose) {
    std::cout << "Subscribers:  " << std::endl
              << "Argmax: " << argmax_subs << std::endl;
  }

  if (argmax_subs > 0) {
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

    // Infer with net
    std::vector<float> logits = _net->infer(cv_img_bgr);

    // get argmax
    int argmax = _net->argmax(logits);

    // get string
    std::string argmax_string(_net->idx_to_string(argmax));

    // make classification message (and publish with proper timestamp)
    bonnetal_classification_msg::classification out_msg;
    out_msg.header = img_msg->header;
    out_msg.argmax = argmax;
    out_msg.argmax_string = argmax_string;
    _output_publisher.publish(out_msg);
  }
}
