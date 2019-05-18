/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */
#pragma once

// standard stuff
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// opencv
#include <opencv2/core/core.hpp>

// yamlcpp
#include "yaml-cpp/yaml.h"

namespace bonnetal {
namespace classification {

/**
 * @brief      Class for classification network inference.
 */
class Net {
 public:
  /**
   * @brief      Constructs the object.
   *
   * @param[in]  model_path  The model path for the inference model directory
   */
  Net(const std::string& model_path);

  /**
   * @brief      Resizing if necessary, pass to float
   *
   * @param image Input image
   *
   * @return cv::Mat containing pre-processed image somewhat ready for CNN
   */
  cv::Mat preprocess(const cv::Mat& image);

  /**
   *
   * @brief      Infer logits from image
   *
   * @param[in]  image    The image to process
   *
   * @return     vector containing logits (unbounded)
   */
  virtual std::vector<float> infer(const cv::Mat& image) = 0;

  /**
   * @brief      String with class name from argmax index
   *
   * @param idx
   *
   * @return std::string Class as a string
   */
  std::string idx_to_string(const uint32_t& idx);

  /**
   * @brief      Set verbosity level for backend execution
   *
   * @param[in]  verbose  True is max verbosity, False is no verbosity.
   *
   * @return     Exit code.
   */
  void verbosity(const bool verbose) { _verbose = verbose; }

  /**
   * @brief      Get argmax from a vector of floats (index of max)
   *
   * @param logits unbounded logits (max is most probable)
   *
   * @return index of max element
   */

  int argmax(std::vector<float> logits);

 protected:
  // general
  std::string _model_path;  // Where to get model weights and cfg
  bool _verbose;            // verbose mode?

  // image properties
  int _img_h, _img_w, _img_d;  // height, width, and depth for inference
  std::vector<float> _img_means, _img_stds;  // mean and std per channel (RGB)

  // problem properties
  uint32_t _n_classes;  // number of classes to differ from

  // config
  YAML::Node _cfg;  // yaml nodes with configuration from training
  std::vector<std::string> _argmax_to_string;  // string from class
};

}  // namespace classification
}  // namespace bonnetal
