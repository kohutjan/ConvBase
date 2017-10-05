#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "operator.hpp"
#include "operators/im2col.hpp"
#include <Eigen/Dense>

class Convolution : public Operator
{
  public:
    Convolution(std::vector<std::vector<std::string>> IO, int _numberOfKernels,
                int _kernelSize, int _stride, int _pad, bool _bias,
                Tensor4D _kernels) : Operator(IO),numberOfKernels(_numberOfKernels),
                kernelSize(_kernelSize), stride(_stride), pad(_pad), bias(_bias),
                kernels(_kernels){}
    std::vector<Tensor4D> Forward(std::vector<Tensor4D> input);
    std::vector<Tensor4D> Backward(std::vector<Tensor4D> input);
    void ComputeOutputShape(int inputN, int inputH, int inputW, int inputC);
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
    Tensor4D kernels;
};

#endif
