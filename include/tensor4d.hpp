#ifndef TENSOR4D_HPP
#define TENSOR4D_HPP

#include <string>
#include <memory>

class Tensor4D
{
  public:
    /*
    Tensor4D()
    {
      n = 0;
      h = 0;
      w = 0;
      c = 0;
      data = NULL;
    }
    */
    Tensor4D(std::string _name, int _N, int _H, int _W, int _C) : name(_name),
             N(_N), H( _H), W( _W), C(_C)
    {
      data = std::make_shared<float>(N * H * W * C);
    }
    Tensor4D(int _N, int _H, int _W, int _C) : N(_N), H( _H), W( _W), C(_C)
    {
      data = std::make_shared<float>(N * H * W * C);
    }
    std::string GetName() { return name; }
    int GetN() const { return N; }
    int GetH() const { return H; }
    int GetW() const { return W; }
    int GetC() const { return C; }
    float * GetPixel(int n, int h, int w)
    {
      return data.get() + n * H * W * C + h * W * C + w * C;
    }
    float * GetData()
    {
      return data.get();
    }

  private:
    std::string name;
    int N;
    int H;
    int W;
    int C;
    std::shared_ptr<float> data;
};

#endif
