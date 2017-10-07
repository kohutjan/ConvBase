#include "operators/im2col.hpp"

using namespace std;

void Im2Col::Forward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  float * topDataVal = top[0].GetData();
  int topOffset = 0;

  for (int n = 0; n < this->bottomShape[0][Nd]; ++n)
  {
    for (int h = 0; h <= this->bottomShape[0][Hd] + 2 * this->pad - this->kernelSize; h += this->stride)
    {
      for (int w = 0; w <= this->bottomShape[0][Wd] + 2 * this->pad - this->kernelSize; w += this->stride)
      {
        for (int hK = 0; hK < this->kernelSize; ++hK)
        {
          if (this->pad)
          {
            for (int wK = 0; wK < this->kernelSize; ++wK)
            {
              bool insidePad = ((h + hK) < this->pad) || ((w + wK) < this->pad) ||
                               ((h + hK) >= this->bottomShape[0][Hd] + this->pad) ||
                               ((w + wK) >= this->bottomShape[0][Wd] + this->pad);
              if (insidePad)
              {
                fill(topDataVal + topOffset, topDataVal + topOffset + this->bottomShape[0][Cd], 0.0);
              }
              else
              {
                float * inputPixel = bottom[0].GetPixel(n, h - this->pad + hK, w - this->pad + wK);
                memcpy(topDataVal + topOffset, inputPixel, this->bottomShape[0][Cd] * sizeof(float));
              }
              topOffset += this->bottomShape[0][Cd];
            }
          }
          else
          {
            float * inputPixel = bottom[0].GetPixel(n, h - this->pad + hK, w - this->pad);
            memcpy(topDataVal + topOffset, inputPixel, this->kernelSize * this->bottomShape[0][Cd] * sizeof(float));
            topOffset += this->kernelSize * this->bottomShape[0][Cd];
          }
        }
      }
    }
  }
}

void Im2Col::Backward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  float * bottomGradientsVal = bottom[0].GetGradients();
  float * topGradientsVal = top[0].GetGradients();
  for (int i = 0; i < this->map.size(); ++i)
  {
    bottomGradientsVal[this->map[i]] += topGradientsVal[i];
  }
}

void Im2Col::ComputeMap()
{
  this->map = vector<int>(this->topShape[0][Hd] * this->topShape[0][Wd]);
  for (int n = 0; n < this->bottomShape[0][Nd]; ++n)
  {
    for (int h = 0; h <= this->bottomShape[0][Hd] + 2 * this->pad - this->kernelSize; h += this->stride)
    {
      for (int w = 0; w <= this->bottomShape[0][Wd] + 2 * this->pad - this->kernelSize; w += this->stride)
      {
        for (int hK = 0; hK < this->kernelSize; ++hK)
        {
          for (int wK = 0; wK < this->kernelSize; ++wK)
          {
            this->map.push_back((n) * ((this->bottomShape[0][Wd] + 2 * this->pad) * (this->bottomShape[0][Hd] + 2 * this->pad)) +
                                (h + hK) * (this->bottomShape[0][Wd] + 2 * this->pad) +
                                (w + wK));
          }
        }
      }
    }
  }
}

void Im2Col::ComputeTopShape()
{
  this->topShape = vector<vector<int> >(1, vector<int>(4));
  this->topShape[0][Nd] = 1;
  this->topShape[0][Hd] = ((this->bottomShape[0][Hd] + 2 * this->pad -
                           this->kernelSize) / this->stride + 1) *
                         ((this->bottomShape[0][Wd] + 2 * this->pad -
                           this->kernelSize) / this->stride + 1) *
                           this->bottomShape[0][Nd];
  this->topShape[0][Wd] = this->kernelSize * this->kernelSize * this->bottomShape[0][Cd];
  this->topShape[0][Cd] = 1;
  this->ComputeMap();
}
