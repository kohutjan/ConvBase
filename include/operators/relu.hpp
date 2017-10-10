#ifndef RELU_HPP
#define RELU_HPP

#include "operator.hpp"

class ReLU : public Operator
{
  public:
    ReLU(std::vector<std::vector<std::string>> IO) : Operator("ReLU", IO){}
    void Forward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void Backward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void ComputeTopShape();
    ~ReLU(){}
};

#endif
