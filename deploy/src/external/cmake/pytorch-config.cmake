####################################
## pytorch specific configuration ##
####################################
# try to find pytorch c++ libs
find_package(Torch)
if (TORCH_FOUND)
  message("Pytorch available!")
  message("Pytorch Libs: ${TORCH_LIBRARIES}")
  message("Pytorch Headers: ${TORCH_INCLUDE_DIRS}")
  set(TORCH_FOUND ON)
else()
  message("Torch NOT FOUND")
  set(TORCH_FOUND OFF)
endif (TORCH_FOUND)

