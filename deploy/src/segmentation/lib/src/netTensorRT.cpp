/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */
#include "netTensorRT.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>

namespace bonnetal {
namespace segmentation {

/**
 * @brief      Constructs the object.
 *
 * @param[in]  model_path  The model path for the inference model directory
 *                         containing the "model.trt" file and the cfg
 */
NetTensorRT::NetTensorRT(const std::string& model_path)
    : Net(model_path), _engine(0), _context(0) {
  // Try to open the model
  std::cout << "Trying to open model" << std::endl;

  // generate trt path form model path
  std::string engine_path = model_path + "/model.trt";

  // try to deserialize the engine
  try {
    deserializeEngine(engine_path);
  } catch (std::exception e) {
    std::cout << "Could not deserialize TensorRT engine. " << std::endl
              << "Generating from sratch... This may take a while..."
              << std::endl;

    // destroy crap from engine
    if (_engine) _engine->destroy();

  } catch (...) {
    throw std::runtime_error("Unknown TensorRT exception. Giving up.");
  }

  // if there is no engine, try to generate one from onnx
  if (!_engine) {
    // generate path
    std::string onnx_path = model_path + "/model.onnx";
    // generate engine
    generateEngine(onnx_path);
    // save engine
    serializeEngine(engine_path);
  }

  // prepare buffers for io :)
  prepareBuffer();

  CUDA_CHECK(cudaStreamCreate(&_cudaStream));

}  // namespace segmentation

/**
 * @brief      Destroys the object.
 */
NetTensorRT::~NetTensorRT() {
  // free cuda buffers
  int n_bindings = _engine->getNbBindings();
  for (int i = 0; i < n_bindings; i++) {
    CUDA_CHECK(cudaFree(_deviceBuffers[i]));
  }

  // free cuda pinned mem
  for (auto& buffer : _hostBuffers) CUDA_CHECK(cudaFreeHost(buffer));

  // destroy cuda stream
  CUDA_CHECK(cudaStreamDestroy(_cudaStream));

  // destroy the execution context
  if (_context) {
    _context->destroy();
  }

  // destroy the engine
  if (_engine) {
    _engine->destroy();
  }
}

/**
 * @brief      Infer argmax from image
 *
 * @param[in]  image    The image to process
 *
 * @return     Image    Containing argmax of each pixel
 */
cv::Mat NetTensorRT::infer(const cv::Mat& image) {
  // check if engine is valid
  if (!_engine) {
    throw std::runtime_error("Invaild engine on inference.");
  }

  // start inference
  if (_verbose) {
    std::cout << "Inferring image with TensorRT" << std::endl;
  }

  // clock now
  auto begin = std::chrono::high_resolution_clock::now();

  // preprocess input (resize, normalize, convert to float, etc)
  cv::Mat input_preproc = preprocess(image);

  // normalize using means and stds, and put in buffer using position
  // (transposing to CHW, and from BGR to RGB on the way)
  int channel_offset = _img_h * _img_w;
  input_preproc.forEach<cv::Vec3f>([&](cv::Vec3f& pixel,
                                       const int* position) -> void {
    for (int i = 0; i < _img_d; i++) {
      // normalization
      pixel[i] = (pixel[i] / 255.0 - this->_img_means[i]) / this->_img_stds[i];

      // put in buffer
      int buffer_idx = channel_offset * (_img_d - 1 - i) +
                       position[0] * _img_w + position[1];
      ((float*)_hostBuffers[_inBindIdx])[buffer_idx] = pixel[i];
    }
  });

  // execute inference
  CUDA_CHECK(
      cudaMemcpyAsync(_deviceBuffers[_inBindIdx], _hostBuffers[_inBindIdx],
                      getBufferSize(_engine->getBindingDimensions(_inBindIdx),
                                    _engine->getBindingDataType(_inBindIdx)),
                      cudaMemcpyHostToDevice, _cudaStream));
  _context->enqueue(1, &_deviceBuffers[0], _cudaStream, nullptr);
  CUDA_CHECK(
      cudaMemcpyAsync(_hostBuffers[_outBindIdx], _deviceBuffers[_outBindIdx],
                      getBufferSize(_engine->getBindingDimensions(_outBindIdx),
                                    _engine->getBindingDataType(_outBindIdx)),
                      cudaMemcpyDeviceToHost, _cudaStream));
  CUDA_CHECK(cudaStreamSynchronize(_cudaStream));

  // take the data out
  cv::Mat argmax(_img_h, _img_w, CV_32SC1);

  // argmax of channel dimension
  argmax.forEach<int32_t>([&](int32_t& pixel, const int* position) -> void {
    // "n_classes"dimension array index from pose
    int32_t pix_idx = position[0] * _img_w + position[1];
    pixel = ((int*)_hostBuffers[_outBindIdx])[pix_idx];
  });

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
void NetTensorRT::verbosity(const bool verbose) {
  // call parent class verbosity
  this->Net::verbosity(verbose);

  // set verbosity for tensorRT logger
  _gLogger.set_verbosity(verbose);
}

/**
 * @brief Get the Buffer Size object
 *
 * @param d dimension
 * @param t data type
 * @return int size of data
 */
int NetTensorRT::getBufferSize(Dims d, DataType t) {
  int size = 1;
  for (int i = 0; i < d.nbDims; i++) size *= d.d[i];

  switch (t) {
    case DataType::kINT32:
      return size * 4;
    case DataType::kFLOAT:
      return size * 4;
    case DataType::kHALF:
      return size * 2;
    case DataType::kINT8:
      return size * 1;
    default:
      throw std::runtime_error("Data type not handled");
  }
  return 0;
}

/**
 * @brief Deserialize an engine that comes from a previous run
 *
 * @param engine_path
 */
void NetTensorRT::deserializeEngine(const std::string& engine_path) {
  // feedback to user where I am
  std::cout << "Trying to deserialize previously stored: " << engine_path
            << std::endl;

  // open model if it exists, otherwise complain
  std::stringstream gieModelStream;
  gieModelStream.seekg(0, gieModelStream.beg);
  std::ifstream file_ifstream(engine_path.c_str());
  if (file_ifstream) {
    std::cout << "Successfully found TensorRT engine file " << engine_path
              << std::endl;
  } else {
    throw std::runtime_error("TensorRT engine file not found" + engine_path);
  }

  // create inference runtime
  IRuntime* infer = createInferRuntime(_gLogger);
  if (infer) {
    std::cout << "Successfully created inference runtime" << std::endl;
  } else {
    throw std::runtime_error("Couldn't created inference runtime.");
  }

// if using DLA, set the desired core before deserialization occurs
#if NV_TENSORRT_MAJOR >= 5 &&                             \
    !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && \
      NV_TENSORRT_PATCH == 0)
  if (DEVICE_DLA_0) {
    infer->setDLACore(0);
  }
  if (DEVICE_DLA_1) {
    infer->setDLACore(1);
  }
#endif
  std::cout << "Successfully selected DLA core." << std::endl;

  // read file
  gieModelStream << file_ifstream.rdbuf();
  file_ifstream.close();
  // read the stringstream into a memory buffer and pass that to TRT.
  gieModelStream.seekg(0, std::ios::end);
  const int modelSize = gieModelStream.tellg();
  gieModelStream.seekg(0, std::ios::beg);
  void* modelMem = malloc(modelSize);
  if (modelMem) {
    std::cout << "Successfully allocated " << modelSize << " for model."
              << std::endl;
  } else {
    throw std::runtime_error("failed to allocate " + std::to_string(modelSize) +
                             " bytes to deserialize model");
  }
  gieModelStream.read((char*)modelMem, modelSize);
  std::cout << "Successfully read " << modelSize << " to modelmem."
            << std::endl;

  // because I use onnx-tensorRT i have to use their plugin factory
  nvonnxparser::IPluginFactory* plug_fact =
      nvonnxparser::createPluginFactory(_gLogger);

  // Now deserialize
  _engine = infer->deserializeCudaEngine(modelMem, modelSize, plug_fact);

  free(modelMem);
  if (_engine) {
    std::cerr << "Created engine!" << std::endl;
  } else {
    throw std::runtime_error("Device failed to create CUDA engine");
  }

  std::cout << "Successfully deserialized Engine from trt file" << std::endl;
}

/**
 * @brief Serialize an engine that we generated in this run
 *
 * @param engine_path
 */
void NetTensorRT::serializeEngine(const std::string& engine_path) {
  // feedback to user where I am
  std::cout << "Trying to serialize engine and save to : " << engine_path
            << " for next run" << std::endl;

  // do only if engine is healthy
  if (_engine) {
    // do the serialization
    IHostMemory* engine_plan = _engine->serialize();
    // Try to save engine for future uses.
    std::ofstream stream(engine_path.c_str(), std::ofstream::binary);
    if (stream)
      stream.write(static_cast<char*>(engine_plan->data()),
                   engine_plan->size());
  }
}

/**
 * @brief Generate an engine from ONNX model
 *
 * @param onnx_path path to onnx file
 */
void NetTensorRT::generateEngine(const std::string& onnx_path) {
  // feedback to user where I am
  std::cout << "Trying to generate trt engine from : " << onnx_path
            << std::endl;

  // create inference builder
  IBuilder* builder = createInferBuilder(_gLogger);

  // set optimization parameters here
  // CAN I DO HALF PRECISION (and report to user)
  std::cout << "Platform ";
  if (builder->platformHasFastFp16()) {
    std::cout << "HAS ";
    builder->setFp16Mode(true);
  } else {
    std::cout << "DOESN'T HAVE ";
    builder->setFp16Mode(false);
  }
  std::cout << "fp16 support." << std::endl;
  // BATCH SIZE IS ALWAYS ONE
  builder->setMaxBatchSize(1);

  // create a network builder
  INetworkDefinition* network = builder->createNetwork();

  // generate a parser to get weights from onnx file
  nvonnxparser::IParser* parser =
      nvonnxparser::createParser(*network, _gLogger);

  // finally get from file
  if (!parser->parseFromFile(onnx_path.c_str(),
                             static_cast<int>(ILogger::Severity::kVERBOSE))) {
    throw std::runtime_error("ERROR: could not parse input ONNX.");
  } else {
    std::cout << "Success picking up ONNX model" << std::endl;
  }

  // make the argmax part now, so that I don't have to copy back big volumes
  // and waste cpu and time
  auto pred = network->addTopK(*network->getOutput(0),
                               nvinfer1::TopKOperation::kMAX, 1, 1);
  if (pred == nullptr) {
    throw std::runtime_error("ERROR: could not add argmax layer.");
  } else {
    std::cout << "Success adding argmax to trt model" << std::endl;
  }
  pred->getOutput(1)->setName("putoelquelee");
  pred->setOutputType(1, nvinfer1::DataType::kINT32);
  network->unmarkOutput(*network->getOutput(0));
  network->markOutput(*pred->getOutput(1));

  // put in engine
  // iterate until I find a size that fits
  for (unsigned long ws_size = MAX_WORKSPACE_SIZE;
       ws_size >= MIN_WORKSPACE_SIZE; ws_size /= 2) {
    // set size
    builder->setMaxWorkspaceSize(ws_size);

    // try to build
    _engine = builder->buildCudaEngine(*network);
    if (!_engine) {
      std::cerr << "Failure creating engine from ONNX model" << std::endl
                << "Current trial size is " << ws_size << std::endl;
      continue;
    } else {
      std::cout << "Success creating engine from ONNX model" << std::endl
                << "Final size is " << ws_size << std::endl;
      break;
    }
  }

  // final check
  if (!_engine) {
    throw std::runtime_error("ERROR: could not create engine from ONNX.");
  } else {
    std::cout << "Success creating engine from ONNX model" << std::endl;
  }
}

/**
 * @brief Prepare io buffers for inference with engine
 */
void NetTensorRT::prepareBuffer() {
  // check if engine is ok
  if (!_engine) {
    throw std::runtime_error(
        "Invalid engine. Please remember to create engine first.");
  }

  // get execution context from engine
  _context = _engine->createExecutionContext();
  if (!_context) {
    throw std::runtime_error("Invalid execution context. Can't infer.");
  }

  int n_bindings = _engine->getNbBindings();
  if (n_bindings != 2) {
    throw std::runtime_error("Invalid number of bindings: " +
                             std::to_string(n_bindings));
  }

  // clear buffers and reserve memory
  _deviceBuffers.clear();
  _deviceBuffers.reserve(n_bindings);
  _hostBuffers.clear();
  _hostBuffers.reserve(n_bindings);

  // allocate memory
  for (int i = 0; i < n_bindings; i++) {
    nvinfer1::Dims dims = _engine->getBindingDimensions(i);
    nvinfer1::DataType dtype = _engine->getBindingDataType(i);
    CUDA_CHECK(cudaMalloc(&_deviceBuffers[i],
                          getBufferSize(_engine->getBindingDimensions(i),
                                        _engine->getBindingDataType(i))));

    CUDA_CHECK(cudaMallocHost(&_hostBuffers[i],
                              getBufferSize(_engine->getBindingDimensions(i),
                                            _engine->getBindingDataType(i))));

    if (_engine->bindingIsInput(i))
      _inBindIdx = i;
    else
      _outBindIdx = i;
    // print for puny human
    std::cout << "Binding: " << i << ", type: " << (int)dtype << std::endl;
    for (int d = 0; d < dims.nbDims; d++) {
      std::cout << "[Dim " << dims.d[d] << "]";
    }
    std::cout << std::endl;
  }

  // exit
  std::cout << "Successfully create binding buffer" << std::endl;
}

}  // namespace segmentation
}  // namespace bonnetal
