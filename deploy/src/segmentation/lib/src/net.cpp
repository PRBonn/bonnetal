/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */
#include "net.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace bonnetal {
namespace segmentation {

/**
 * @brief      Constructs the object.
 *
 * @param[in]  model_path  The model path for the inference model directory
 */
Net::Net(const std::string& model_path)
    : _model_path(model_path), _verbose(true) {
  // set default verbosity level
  verbosity(_verbose);

  // Try to get the config file as well
  std::string cfg_path = _model_path + "/cfg.yaml";
  try {
    _cfg = YAML::LoadFile(cfg_path);
  } catch (YAML::Exception& ex) {
    throw std::runtime_error("Can't open cfg.yaml from " + cfg_path);
  }

  // Get label dictionary from yaml cfg
  YAML::Node color_map;
  try {
    color_map = _cfg["dataset"]["color_map"];
  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one the label dictionary from cfg in " + cfg_path
              << std::endl;
    throw ex;
  }

  // Generate string map from xentropy indexes (that we'll get from argmax)
  YAML::const_iterator it;
  _argmax_to_bgr.resize(color_map.size());
  for (it = color_map.begin(); it != color_map.end(); ++it) {
    // Get label and key
    int key = it->first.as<int>();  // <- key
    cv::Vec3b color = {
        static_cast<uint8_t>(color_map[key][0].as<unsigned int>()),
        static_cast<uint8_t>(color_map[key][1].as<unsigned int>()),
        static_cast<uint8_t>(color_map[key][2].as<unsigned int>())};
    _argmax_to_bgr[key] = color;
  }

  // get the number of classes
  _n_classes = color_map.size();

  // get image size
  _img_h = _cfg["dataset"]["img_prop"]["height"].as<int>();
  _img_w = _cfg["dataset"]["img_prop"]["width"].as<int>();
  _img_d = _cfg["dataset"]["img_prop"]["depth"].as<int>();

  // check that depth is 3, rest is not implemented
  if (_img_d != 3) {
    throw std::runtime_error("Only depth 3 implemented, and net is " + _img_d);
  }

  // get normalization parameters
  YAML::Node img_means, img_stds;
  try {
    img_means = _cfg["dataset"]["img_means"];
    img_stds = _cfg["dataset"]["img_stds"];
  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one the mean or std dictionary from cfg"
              << std::endl;
    throw ex;
  }
  // fill in means from yaml node
  for (it = img_means.begin(); it != img_means.end(); ++it) {
    // Get value
    float mean = it->as<float>();
    // Put in indexing vector
    _img_means.push_back(mean);
  }
  // fill in stds from yaml node
  for (it = img_stds.begin(); it != img_stds.end(); ++it) {
    // Get value
    float std = it->as<float>();
    // Put in indexing vector
    _img_stds.push_back(std);
  }
}

/**
 * @brief      Resizing if necessary, pass to float
 *
 * @param image Input image
 *
 * @return cv::Mat containing pre-processed image somewhat ready for CNN
 */
cv::Mat Net::preprocess(const cv::Mat& image) {
  // create temp mat to fill in with processed data
  cv::Mat preprocessed;

  // resize if image is not as desired
  if (image.rows != _img_h || image.cols != _img_w) {
    // Complain that I am resizing just to make sure the user knows
    if (_verbose) {
      std::cout << "Watch out, I'm resizing internally. Input should be "
                << _img_h << "x" << _img_w << ", but is " << image.rows << "x"
                << image.cols << std::endl;
    }

    // resize
    cv::resize(image, preprocessed, cv::Size(_img_w, _img_h), 0, 0,
               cv::INTER_LINEAR);
  } else {
    // just put in preprocessed
    preprocessed = image.clone();
  }

  // Make float
  preprocessed.convertTo(preprocessed, CV_32F);

  // return the vector organized as CHW, normalized, and as float
  // RVO should move this
  return preprocessed;
}

/**
 * @brief      Resizing if necessary and other postproc steps for argmax
 *
 * @param image Input image
 *
 * @return cv::Mat containing pre-processed image somewhat ready for CNN
 */
cv::Mat Net::postprocess(const cv::Mat& img, const cv::Mat& argmax) {
  // create temp mat to fill in with processed data
  cv::Mat postprocessed;

  // resize if image is not as desired
  if (img.rows != argmax.rows || img.cols != argmax.cols) {
    // Complain that I am resizing just to make sure the user knows
    if (_verbose) {
      std::cout << "Watch out, I'm resizing output internally (NN)."
                << std::endl;
    }

    // resize
    cv::resize(argmax, postprocessed, cv::Size(img.cols, img.rows), 0, 0,
               cv::INTER_NEAREST);
  } else {
    // just put in preprocessed
    postprocessed = argmax.clone();
  }

  // RVO should move this
  return postprocessed;
}

/**
 * @brief      Convert mask to color using dictionary as lut
 *
 * @param[in]  argmax      The mask from argmax
 *
 * @return     the colored segmentation mask :)
 */
cv::Mat Net::color(const cv::Mat& argmax) {
  cv::Mat colored(argmax.rows, argmax.cols, CV_8UC3);
  colored.forEach<cv::Vec3b>(
      [&](cv::Vec3b& pixel, const int* position) -> void {
        // argmax ist int32, ojo!
        int c = argmax.at<int32_t>(cv::Point(position[1], position[0]));
        pixel = _argmax_to_bgr[c];
      });
  return colored;
}

/**
 * @brief      Blend image with color mask
 *
 * @param[in]  img         Image being inferred
 * @param[in]  color_mask  Color mask from CNN
 *
 * @return     Blent mask with input
 */
cv::Mat Net::blend(const cv::Mat& img, const cv::Mat& color_mask) {
  cv::Mat blend;
  cv::addWeighted(img, 1.0, color_mask, 0.5, 0.0, blend);
  return blend;
}

}  // namespace segmentation
}  // namespace bonnetal
