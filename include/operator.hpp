#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include "tensor4d.hpp"
#include <vector>

class Operator
{
  public:
    std::vector<std::string> GetBottoms() const { return bottoms; }
    std::vector<std::string> GetTops() const { return tops; }
    int GetOutputN() const { return outputN; }
    int GetOutputH() const { return outputH; }
    int GetOutputW() const { return outputW; }
    int GetOutputC() const { return outputC; }
    virtual std::vector<Tensor4D> Forward(std::vector<Tensor4D> inputs) = 0;
    virtual std::vector<Tensor4D> Backward(std::vector<Tensor4D> inputs) = 0;
    virtual void ComputeOutputShape(int inputN, int inputH, int inputW, int inputC) = 0;
    virtual ~Operator(){};

  protected:
    Operator(std::vector<std::vector<std::string>> IO) :bottoms(IO[0]),
             tops(IO[1]){}
    Operator(){}
    int outputN;
    int outputH;
    int outputW;
    int outputC;

  private:
    const std::vector<std::string> bottoms;
    const std::vector<std::string> tops;

};

#endif
