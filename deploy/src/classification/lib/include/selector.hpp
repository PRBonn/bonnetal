/* Copyright (c) 2019 Andres Milioto, Cyrill Stachniss, University of Bonn.
 *
 *  This file is part of Bonnetal, and covered by the provided LICENSE file.
 *
 */
#pragma once

// standard stuff
#include <algorithm>
#include <iostream>
#include <string>

// selective network library (conditional build)
#include "external.hpp"  //this one contains the flags for external lib build
#ifdef TORCH_FOUND
#include "netPytorch.hpp"
#endif
#ifdef TENSORRT_FOUND
#include "netTensorRT.hpp"
#endif

// Only to be used with classification
namespace bonnetal {
namespace classification {

/**
 * @brief Makes a network with the desired backend, checking that it exists,
 *        it is implemented, and that it was compiled.
 *
 * @param backend "pytorch, tensorrt"
 * @return std::unique_ptr<Net>
 */
std::unique_ptr<Net> make_net(const std::string& path,
                              const std::string& backend);

}  // namespace classification
}  // namespace bonnetal
