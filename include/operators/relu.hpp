#ifndef RELU_HPP
#define RELU_HPP

#include "operator.hpp"

class ReLU : public Operator
{
  public:
    ReLU(std::vector<std::vector<std::string>> IO) : Operator(IO){}
    std::vector<Tensor4D> Forward(std::vector<Tensor4D> input);
    ~ReLU(){}
};

#endif
