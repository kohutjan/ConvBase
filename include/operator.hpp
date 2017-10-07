#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include "tensor4d.hpp"
#include <vector>

class Operator
{
  public:
    std::vector<std::string> GetBottomName() const { return bottomName; }
    std::vector<std::string> GetTopName() const { return topName; }
    std::vector<std::vector<int> > GetBottomShape() const { return bottomShape; }
    std::vector<std::vector<int> > GetTopShape() const { return topShape; }
    virtual void Forward(std::vector<Tensor4D> bottoms, std::vector<Tensor4D> tops) = 0;
    virtual void Backward(std::vector<Tensor4D> bottoms, std::vector<Tensor4D> tops) = 0;
    virtual void SetBottomShape(std::vector<std::vector<int>> _bottomShape){ bottomShape = _bottomShape; }
    virtual void ComputeTopShape() = 0;
    virtual ~Operator(){};

  protected:
    Operator(std::vector<std::vector<std::string>> IO) : bottomName(IO[0]), topName(IO[1])
    {
      bottomShape = std::vector<std::vector<int>>(bottomName.size(), std::vector<int>(4));
      topShape = std::vector<std::vector<int>>(topName.size(), std::vector<int>(4));
    }
    Operator(){}
    std::vector<std::vector<int>> bottomShape;
    std::vector<std::vector<int>> topShape;

  private:
    const std::vector<std::string> bottomName;
    const std::vector<std::string> topName;

};

#endif
