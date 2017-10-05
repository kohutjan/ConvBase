#ifndef IM2COL_HPP
#define IM2COL_HPP

#include "operator.hpp"
#include <cstring>

class Im2Col : public Operator
{
  public:
    Im2Col(std::vector<std::vector<std::string>> IO, int _kernelSize,
           int _stride, int _pad) : Operator(IO), kernelSize(_kernelSize),
           stride(_stride), pad(_pad){}
    std::vector<Tensor4D> Forward(std::vector<Tensor4D> input);
    std::vector<Tensor4D> Backward(std::vector<Tensor4D> input);
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
