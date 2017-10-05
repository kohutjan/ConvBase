#ifndef RESHAPE_HPP
#define RESHAPE_HPP

#include "operator.hpp"

class Reshape : public Operator
{
  public:
    Reshape(std::vector<std::vector<std::string>> IO, int _h, int _w, int _c)
            : Operator(IO), h(_h), w(_w), c(_c){}
    std::vector<Tensor4D> Forward(std::vector<Tensor4D> input);
    std::vector<Tensor4D> Backward(std::vector<Tensor4D> input);
    int GetH() const { return h; }
    int GetW() const { return w; }
    int GetC() const { return c; }
    ~Reshape(){}
  private:
    const int h;
    const int w;
    const int c;
};

#endif
