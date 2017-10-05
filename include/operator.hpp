#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include "tensor4d.hpp"
#include <vector>

class Operator
{
  public:
    std::vector<std::string> GetBottoms() { return bottoms; }
    std::vector<std::string> GetTops() { return tops; }
    virtual std::vector<Tensor4D> Forward(std::vector<Tensor4D> inputs) = 0;
    virtual std::vector<Tensor4D> Backward(std::vector<Tensor4D> inputs) = 0;
    virtual ~Operator(){};

  protected:
    Operator(std::vector<std::vector<std::string>> IO) :
    bottoms(IO[0]), tops(IO[1]){}

  private:
    const std::vector<std::string> bottoms;
    const std::vector<std::string> tops;
};

#endif
