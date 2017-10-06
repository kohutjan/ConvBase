#ifndef IM2COL_HPP
#define IM2COL_HPP

#include "operator.hpp"
#include <cstring>

class Im2Col : public Operator
{
  public:
    Im2Col(int _kernelSize, int _stride, int _pad) : kernelSize(_kernelSize),
           stride(_stride), pad(_pad){}
    void Forward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void Backward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void ComputeTopShape();
    int GetKernelSize() const { return kernelSize; }
    int GetStride() const { return stride; }
    int GetPad() const { return pad; }
    ~Im2Col(){}

  private:
    const int kernelSize;
    const int stride;
    const int pad;
};

#endif
