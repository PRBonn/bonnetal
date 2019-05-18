#include "external.hpp"
#include <iostream>

namespace bonnetal {
namespace external {

void print_flags(void) {
#ifdef TENSORRT_FOUND
  std::cout << "[TENSORRT_FOUND] Defined" << std::endl;
#else
  std::cerr << "[TENSORRT_FOUND] NOT defined" << std::endl;
#endif
#ifdef TORCH_FOUND
  std::cout << "[TORCH_FOUND] Defined" << std::endl;
#else
  std::cerr << "[TORCH_FOUND] NOT defined" << std::endl;
#endif
}
}  // namespace external
}  // namespace bonnetal