/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */
#include "netPytorch.hpp"

namespace bonnetal {
namespace segmentation {

void torch_jit_module_compat(const torch::jit::script::Module& src,
                             std::shared_ptr<torch::jit::script::Module>& module) {
  // Works for pytorch >= 1.2
  *module = src;
}

void torch_jit_module_compat(const std::shared_ptr<torch::jit::script::Module>& src,
                             std::shared_ptr<torch::jit::script::Module>& module) {
  // Works for pytorch = 1.1
  module = src;
}

/**
 * @brief      Constructs the object.
 *
 * @param[in]  model_path  The model path for the inference model directory
 *                         containing the "model.pytorch" file and the cfg
 */
NetPytorch::NetPytorch(const std::string& model_path) : Net(model_path) {
  // Try to open the model
  std::cout << "Trying to open model" << std::endl;
  try {
    torch_jit_module_compat(torch::jit::load(_model_path + "/model.pytorch",
                                             torch::kCUDA),
                            _module);
    _device = std::unique_ptr<torch::Device>(new torch::Device(torch::kCUDA));
  } catch (...) {
    std::cout << "Could not send model to GPU, using CPU" << std::endl;
    torch_jit_module_compat(torch::jit::load(_model_path + "/model.pytorch",
                                             torch::kCPU),
                            _module);
    _device = std::unique_ptr<torch::Device>(new torch::Device(torch::kCPU));
  }

  // Check that it opened and is healthy
  if (_module != nullptr) {
    std::cout << "Successfully opened model" << std::endl;
  } else {
    throw std::runtime_error(
        "Could not open model.pytorch. "
        "I Can't infer without it, so I'm giving up");
  }
}

/**
 * @brief      Destroys the object.
 */
NetPytorch::~NetPytorch() {}

/**
 * @brief      Infer argmax from image
 *
 * @param[in]  image    The image to process
 *
 * @return     Image    Containing argmax of each pixel
 */
cv::Mat NetPytorch::infer(const cv::Mat& image) {
  // take all with no grad in this scope (same as py with torch.no_grad():)
  torch::NoGradGuard no_grad;

  // start inference
  if (_verbose) {
    std::cout << "Inferring image with Pytorch" << std::endl;
  }

  auto begin = std::chrono::high_resolution_clock::now();

  // preprocess input (resize, normalize, convert to float, etc)
  cv::Mat input_preproc = preprocess(image);

  // put in tensor
  at::Tensor image_tensor = torch::from_blob(
      input_preproc.data, {1, _img_h, _img_w, _img_d}, at::kFloat);

  // send to gpu
  image_tensor = image_tensor.to(*_device);

  // reshape for cnn
  image_tensor = image_tensor.permute({0, 3, 1, 2});

  // Normalize mean and std (and pass to RGB)
  image_tensor[0][2] =
      image_tensor[0][0].div(255).sub(_img_means[0]).div(_img_stds[0]);
  image_tensor[0][1] =
      image_tensor[0][1].div(255).sub(_img_means[1]).div(_img_stds[1]);
  image_tensor[0][0] =
      image_tensor[0][2].div(255).sub(_img_means[2]).div(_img_stds[2]);

  // infer
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(image_tensor);
  torch::Tensor logits_tensor = _module->forward(inputs).toTensor();

  // do argmax
  torch::Tensor argmax_tensor = logits_tensor.argmax(1, false);

  // make int32 to work with opencv
  argmax_tensor = argmax_tensor.toType(at::kInt);

  // get data in cpu
  argmax_tensor = argmax_tensor.to(torch::kCPU);

  // just in case type is wrong
  assert(logits_tensor.numel() == _img_h * _img_w * _n_classes);
  assert(argmax_tensor.numel() == _img_h * _img_w);
  // std::cout << logits_tensor.numel() << std::endl;
  // std::cout << argmax_tensor.numel() << std::endl;

  // fill cpu logits
  cv::Mat argmax(_img_h, _img_w, CV_32SC1);
  memcpy(argmax.data, argmax_tensor.data<int32_t>(),
         argmax_tensor.numel() * sizeof(int32_t));

  // post process
  argmax = postprocess(image, argmax);

  auto end = std::chrono::high_resolution_clock::now();
  if (_verbose) {
    std::cout << "Time to infer: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                      begin)
                         .count() /
                     1000000.0
              << "ms" << std::endl;
  }

  return argmax;
}

/**
 * @brief      Set verbosity level for backend execution
 *
 * @param[in]  verbose  True is max verbosity, False is no verbosity.
 *
 * @return     Exit code.
 */
void NetPytorch::verbosity(const bool verbose) {
  // call parent class verbosity
  this->Net::verbosity(verbose);
}
}  // namespace segmentation
}  // namespace bonnetal
