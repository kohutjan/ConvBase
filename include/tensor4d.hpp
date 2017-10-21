#ifndef TENSOR4D_HPP
#define TENSOR4D_HPP

#define Nd 0
#define Hd 1
#define Wd 2
#define Cd 3

#include <string>
#include <memory>
#include <vector>

class Tensor4D
{
  public:
    Tensor4D()
    {
      shape = {0, 0, 0, 0};
    }
    Tensor4D(std::string _name, std::vector<int> _shape) : name(_name), shape(_shape)
    {
      data = std::shared_ptr<float>(new float[shape[Nd] * shape[Hd] * shape[Wd] *
                                              shape[Cd]], std::default_delete<float[]>());
      gradients = std::shared_ptr<float>(new float[shape[Nd] * shape[Hd] * shape[Wd] *
                                                   shape[Cd]], std::default_delete<float[]>());
    }
    Tensor4D(std::vector<int> _shape) : shape(_shape)
    {
      data = std::shared_ptr<float>(new float[shape[Nd] * shape[Hd] * shape[Wd] *
                                              shape[Cd]], std::default_delete<float[]>());
      gradients = std::shared_ptr<float>(new float[shape[Nd] * shape[Hd] * shape[Wd] *
                                                   shape[Cd]], std::default_delete<float[]>());
    }
    Tensor4D(std::string _name, int _N, int _H, int _W, int _C) : name(_name), shape(4)
    {
      shape[Nd] = _N;
      shape[Hd] = _H;
      shape[Wd] = _W;
      shape[Cd] = _C;
      data = std::shared_ptr<float>(new float[shape[Nd] * shape[Hd] * shape[Wd] *
                                              shape[Cd]], std::default_delete<float[]>());
      gradients = std::shared_ptr<float>(new float[shape[Nd] * shape[Hd] * shape[Wd] *
                                                   shape[Cd]], std::default_delete<float[]>());
    }
    Tensor4D(int _N, int _H, int _W, int _C) : shape(4)
    {
      shape[Nd] = _N;
      shape[Hd] = _H;
      shape[Wd] = _W;
      shape[Cd] = _C;
      data = std::shared_ptr<float>(new float[shape[Nd] * shape[Hd] * shape[Wd] *
                                              shape[Cd]], std::default_delete<float[]>());
      gradients = std::shared_ptr<float>(new float[shape[Nd] * shape[Hd] * shape[Wd] *
                                                   shape[Cd]], std::default_delete<float[]>());
    }
    std::string GetName() const { return name; }
    void SetName(std::string _name) { name = _name; }
    std::vector<int> GetShape() const { return shape; }
    void SetShape(std::vector<int> _shape) { shape = _shape; }
    int GetSize() { return shape[Nd] * shape[Hd] * shape[Wd] * shape[Cd]; }
    float * GetPixel(int n, int h, int w)
    {
      return (data.get() +
              n * shape[Hd] * shape[Wd] * shape[Cd] +
              h * shape[Wd] * shape[Cd] + w * shape[Cd]);
    }
    int GetActualIndex(int n, int h, int w, int c)
    {
      return (n * shape[Hd] * shape[Wd] * shape[Cd] +
              h * shape[Wd] * shape[Cd] +
              w * shape[Cd] + c);
    }
    int GetActualIndex(int n, int h, int w, int c, int pad)
    {
      int imagePadOffset = (pad * (shape[Wd] + 2 * pad + shape[Hd]) * 2) * shape[Cd];
      int widthPadOffset = (pad * (shape[Wd] + 2 * pad)) * shape[Cd];
      int heightPadOffset = ((h - pad) * 2 * pad + pad) * shape[Cd];
      int padOffset = n * imagePadOffset + widthPadOffset + heightPadOffset;
      return ((n * (shape[Hd] + 2 * pad) * (shape[Wd] + 2 * pad) * shape[Cd] +
               h * ((shape[Wd] + 2 * pad) * shape[Cd]) +
               w * shape[Cd] + c) - padOffset);
    }
    float * GetData()
    {
      return data.get();
    }
    float * GetGradients()
    {
      return gradients.get();
    }

  private:
    std::string name;
    std::vector<int> shape;
    std::shared_ptr<float> data;
    std::shared_ptr<float> gradients;
};

#endif
