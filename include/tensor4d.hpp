#ifndef TENSOR4D_HPP
#define TENSOR4D_HPP

#include <string>
#include <memory>

class Tensor4D
{
  public:
    Tensor4D()
    {
      n = 0;
      h = 0;
      w = 0;
      c = 0;
      data = NULL;
    }
    Tensor4D(std::string _name, int _n, int _h, int _w, int _c)
             : name(_name), n(_n), h( _h), w( _w), c(_c), data(NULL){}
    std::string GetName() { return name; }
    int GetH() const { return h; }
    int GetW() const { return w; }
    int GetN() const { return n; }
    int GetC() const { return c; }
    std::shared_ptr<float> GetData() const { return data; }

  private:
    std::string name;
    int n;
    int h;
    int w;
    int c;
    std::shared_ptr<float> data;
};

#endif
