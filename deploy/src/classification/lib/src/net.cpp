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
namespace classification {

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
  YAML::Node label_dict;
  try {
    label_dict = _cfg["dataset"]["labels"];
  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one the label dictionary from cfg in " + cfg_path
              << std::endl;
    throw ex;
  }

  // Generate string map from xentropy indexes (that we'll get from argmax)
  YAML::const_iterator it;
  _argmax_to_string.resize(label_dict.size());
  for (it = label_dict.begin(); it != label_dict.end(); ++it) {
    // Get label and key
    int key = it->first.as<int>();                     // <- key
    std::string label = it->second.as<std::string>();  // <- argmax label

    // Put in indexing vector
    _argmax_to_string[key] = label;
  }

  // get the number of classes
  _n_classes = label_dict.size();

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
 * @brief      String with class name from argmax index
 *
 * @param idx
 *
 * @return std::string Class as a string
 */
std::string Net::idx_to_string(const uint32_t& idx) {
  if (_verbose) {
    std::cout << "Argmax String:" << _argmax_to_string[idx] << std::endl;
  }
  return _argmax_to_string[idx];
}

/**
 * @brief      Get argmax from a vector of floats (index of max)
 *
 * @param logits unbounded logits (max is most probable)
 *
 * @return index of max element
 */

int Net::argmax(std::vector<float> logits) {
  // get argmax
  std::vector<float>::iterator result =
      std::max_element(logits.begin(), logits.end());
  int argmax = std::distance(logits.begin(), result);

  if (_verbose) {
    std::cout << "Argmax: " << argmax << std::endl;
  }

  return argmax;
}

}  // namespace classification
}  // namespace bonnetal
