/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */
#pragma once
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/utils.h>
#include "net.hpp"

namespace bonnetal {
namespace segmentation {

/**
 * @brief      Class for segmentation network inference with Pytorch.
 */
class NetPytorch : public Net {
 public:
  /**
   * @brief      Constructs the object.
   *
   * @param[in]  model_path  The model path for the inference model directory
   *                         containing the "model.pytorch" file and the cfg
   */
  NetPytorch(const std::string& model_path);

  /**
   * @brief      Destroys the object.
   */
  ~NetPytorch();

  /**
   * @brief      Infer argmax from image
   *
   * @param[in]  image    The image to process
   *
   * @return     Image    Containing argmax of each pixel
   */
  cv::Mat infer(const cv::Mat& image);

  /**
   * @brief      Set verbosity level for backend execution
   *
   * @param[in]  verbose  True is max verbosity, False is no verbosity.
   *
   * @return     Exit code.
   */
  void verbosity(const bool verbose);

 protected:
  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> _module;

  // device for inference
  std::unique_ptr<torch::Device> _device;
};

}  // namespace segmentation
}  // namespace bonnetal
