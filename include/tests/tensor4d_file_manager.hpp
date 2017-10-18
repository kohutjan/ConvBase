#ifndef TENSOR4D_FILE_MANAGER
#define TENSOR4D_FILE_MANAGER

#include "tensor4d.hpp"
#include <fstream>
#include <iostream>

Tensor4D loadTensor4D(std::string tensorFile, bool gradients);
void storeTensor4D(Tensor4D tensor, std::string tensorFile, bool gradients);

#endif
