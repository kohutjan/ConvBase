#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "operator.hpp"

class Convolution : public Operator
{
  public:
    Convolution(std::vector<std::vector<std::string>> IO, int _numberOfKernels,
         int _kernelSize, int _stride, int _pad, bool _bias) : Operator(IO),
         numberOfKernels(_numberOfKernels), kernelSize(_kernelSize),
         stride(_stride), pad(_pad), bias(_bias){}
    std::vector<Tensor4D> Forward(std::vector<Tensor4D> input);
    int GetNumberOfKernels() const { return numberOfKernels; }
    int GetKernelSize() const { return kernelSize; }
    int GetStride() const { return stride; }
    int GetPad() const { return pad; }
    bool GetBias() const { return bias; }
    ~Convolution(){}

  private:
    const int numberOfKernels;
    const int kernelSize;
    const int stride;
    const int pad;
    const bool bias;
};

#endif
