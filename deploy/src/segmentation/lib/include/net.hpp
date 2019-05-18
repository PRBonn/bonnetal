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
namespace segmentation {

/**
 * @brief      Class for segmentation network inference.
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
   * @brief      Resizing if necessary and other postproc steps for argmax
   *
   * @param image Input image
   *
   * @return cv::Mat containing pre-processed image somewhat ready for CNN
   */
  cv::Mat postprocess(const cv::Mat& img, const cv::Mat& argmax);

  /**
   * @brief      Infer logits from image
   *
   * @param[in]  image    The image to process
   *
   * @return     Image with argmax
   */
  virtual cv::Mat infer(const cv::Mat& image) = 0;

  /**
   * @brief      Convert mask to color using dictionary as lut
   *
   * @param[in]  argmax      The mask from argmax
   *
   * @return     the colored segmentation mask :)
   */
  cv::Mat color(const cv::Mat& argmax);

  /**
   * @brief      Blend image with color mask
   *
   * @param[in]  img         Image being inferred
   * @param[in]  color_mask  Color mask from CNN
   *
   * @return     Blent mask with input
   */
  cv::Mat blend(const cv::Mat& img, const cv::Mat& color_mask);

  /**
   * @brief      Set verbosity level for backend execution
   *
   * @param[in]  verbose  True is max verbosity, False is no verbosity.
   *
   * @return     Exit code.
   */
  void verbosity(const bool verbose) { _verbose = verbose; }

 protected:
  // general
  std::string _model_path;  // Where to get model weights and cfg
  bool _verbose;            // verbose mode?

  // image properties
  int _img_h, _img_w, _img_d;  // height, width, and depth for inference
  std::vector<float> _img_means, _img_stds;  // mean and std per channel (RGB)

  // problem properties
  int32_t _n_classes;  // number of classes to differ from

  // config
  YAML::Node _cfg;  // yaml nodes with configuration from training
  std::vector<cv::Vec3b> _argmax_to_bgr;  // for color conversion
};

}  // namespace segmentation
}  // namespace bonnetal
