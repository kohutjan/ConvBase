#include "operators/pooling.hpp"

using namespace std;

void Pooling::Forward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  for (int n = 0; n < this->bottomShape[0][Nd]; ++n)
  {
    for (int h = 0; h <= this->bottomShape[0][Hd] - this->kernelSize; h += this->stride)
    {
      for (int w = 0; w <= this->bottomShape[0][Wd] - this->kernelSize; w += this->stride)
      {
        for (int hK = 0; hK < this->kernelSize; ++hK)
        {
          for (int wK = 0; wK < this->kernelSize; ++wK)
          {
            
          }
        }
      }
    }
  }
}

void Pooling::Backward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{

}

void Pooling::ComputeTopShape()
{
  this->topShape[0][Nd] = this->bottomShape[0][Nd];
  this->topShape[0][Hd] = (this->bottomShape[0][Hd] - this->kernelSize) / this->stride + 1;
  this->topShape[0][Wd] = (this->bottomShape[0][Wd] - this->kernelSize) / this->stride + 1;
  this->topShape[0][Cd] = this->bottomShape[0][Cd];
}
