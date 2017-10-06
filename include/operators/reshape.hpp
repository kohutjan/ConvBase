#ifndef RESHAPE_HPP
#define RESHAPE_HPP

#include "operator.hpp"

class Reshape : public Operator
{
  public:
    Reshape(std::vector<std::vector<std::string>> IO, int _h, int _w, int _c)
            : Operator(IO), h(_h), w(_w), c(_c){}
    void Forward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void Backward(std::vector<Tensor4D> bottom, std::vector<Tensor4D> top);
    void ComputeTopShape();
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
