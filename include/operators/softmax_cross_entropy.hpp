#ifndef SOFTMAX_CROSS_ENTROPY_HPP
#define SOFTMAX_CROSS_ENTROPY_HPP

#include "operator.hpp"
#include <math.h>

class SoftmaxCrossEntropy : public Operator
{
  public:
    SoftmaxCrossEntropy(std::vector<std::vector<std::string>> IO) :
                        Operator("SoftmaxCrossEntropy", IO){}
    void Forward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void Backward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void ComputeTopShape();
    Tensor4D GetSoftmaxTop() const { return softmaxTop; }
    ~SoftmaxCrossEntropy(){}

  private:
    Tensor4D softmaxTop;
};

#endif
