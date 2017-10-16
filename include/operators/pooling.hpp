#ifndef POOLING_HPP
#define POOLING_HPP

#include "operator.hpp"

class Pooling : public Operator
{
  public:
    Pooling(std::vector<std::vector<std::string>> IO, int _kernelSize,
            int _stride) : Operator("Pooling", IO), kernelSize(_kernelSize),
            stride(_stride){}
    void Forward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void Backward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void ComputeTopShape();
    int GetKernelSize() const { return kernelSize; }
    int GetStride() const { return stride; }
    ~Pooling(){}
  private:
    const int kernelSize;
    const int stride;
};

#endif
