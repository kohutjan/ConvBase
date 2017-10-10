#ifndef RESHAPE_HPP
#define RESHAPE_HPP

#include "operator.hpp"
#include <cstring>

class Reshape : public Operator
{
  public:
    Reshape(std::vector<std::vector<std::string>> IO, std::vector<int> _shape)
            : Operator("Reshape", IO), shape(_shape){}
    void Forward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void Backward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void ComputeTopShape();
    std::vector<int> GetShape() const { return shape; }
    ~Reshape(){}
  private:
    std::vector<int> shape;
};

#endif
