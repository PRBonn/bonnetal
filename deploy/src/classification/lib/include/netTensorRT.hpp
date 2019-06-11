/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */
#pragma once

// For plugin factory
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvOnnxParserRuntime.h>
#include <cuda_runtime.h>
#include <fstream>
#include <ios>
#include "net.hpp"

#define MAX_WORKSPACE_SIZE (1UL << 33)  // gpu workspace size (8gb is pretty good)
#define MIN_WORKSPACE_SIZE (1UL << 20)  // gpu workspace size (pretty bad)

#define DEVICE_DLA_0 0                // jetson DLA 0 enabled
#define DEVICE_DLA_1 0                // jetson DLA 1 enabled

using namespace nvinfer1;  // I'm taking a liberty because the code is
                           // unreadable otherwise

#define CUDA_CHECK(status) {                                                      \
    if(status != cudaSuccess) {                                                   \
      printf("%s in %s at %d\n", cudaGetErrorString(status), __FILE__, __LINE__); \
      exit(-1);                                                                   \
    }                                                                             \
}

namespace bonnetal {
namespace classification {

// Logger for GIE info/warning/errors
class Logger : public ILogger {
 public:
  void set_verbosity(bool verbose) { _verbose = verbose; }
  void log(Severity severity, const char* msg) override {
    if (_verbose) {
      switch (severity) {
        case Severity::kINTERNAL_ERROR:
          std::cerr << "INTERNAL_ERROR: ";
          break;
        case Severity::kERROR:
          std::cerr << "ERROR: ";
          break;
        case Severity::kWARNING:
          std::cerr << "WARNING: ";
          break;
        case Severity::kINFO:
          std::cerr << "INFO: ";
          break;
        default:
          std::cerr << "UNKNOWN: ";
          break;
      }
      std::cout << msg << std::endl;
    }
  }

 private:
  bool _verbose = false;
};

/**
 * @brief      Class for classification network inference with TensorRT.
 */
class NetTensorRT : public Net {
 public:
  /**
   * @brief      Constructs the object.
   *
   * @param[in]  model_path  The model path for the inference model directory
   *                         containing the "model.trt" file and the cfg
   */
  NetTensorRT(const std::string& model_path);

  /**
   * @brief      Destroys the object.
   */
  ~NetTensorRT();

  /**
   * @brief      Infer logits from image
   *
   * @param[in]  image    The image to process
   *
   * @return     vector containing logits (unbounded)
   */
  std::vector<float> infer(const cv::Mat& image);

  /**
   * @brief      Set verbosity level for backend execution
   *
   * @param[in]  verbose  True is max verbosity, False is no verbosity.
   *
   * @return     Exit code.
   */
  void verbosity(const bool verbose);

  /**
   * @brief Get the Buffer Size object
   *
   * @param d dimension
   * @param t data type
   * @return int size of data
   */
  int getBufferSize(Dims d, DataType t);

  /**
   * @brief Deserialize an engine that comes from a previous run
   *
   * @param engine_path
   */
  void deserializeEngine(const std::string& engine_path);

  /**
   * @brief Serialize an engine that we generated in this run
   *
   * @param engine_path
   */
  void serializeEngine(const std::string& engine_path);

  /**
   * @brief Generate an engine from ONNX model
   *
   * @param onnx_path path to onnx file
   */
  void generateEngine(const std::string& onnx_path);

  /**
   * @brief Prepare io buffers for inference with engine
   */
  void prepareBuffer();

 protected:
  ICudaEngine* _engine;  // tensorrt engine (smart pointer doesn't work, must
                         // destroy myself)
  IExecutionContext*
      _context;     // execution context (must destroy in destructor too)
  Logger _gLogger;  // trt logger
  std::vector<void*> _deviceBuffers;  // device mem
  cudaStream_t _cudaStream; // cuda stream for async ops
  std::vector<float*> _hostBuffers;
  uint _inBindIdx;
  uint _outBindIdx;

};

}  // namespace classification
}  // namespace bonnetal
