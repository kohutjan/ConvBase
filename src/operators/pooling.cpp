#include "operators/pooling.hpp"

using namespace std;

void Pooling::Forward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  float * topDataVal = top[0].GetData();
  int topOffset = 0;

  for (int n = 0; n < this->bottomShape[0][Nd]; ++n)
  {
    for (int h = 0; h <= this->bottomShape[0][Hd] - this->kernelSize; h += this->stride)
    {
      for (int w = 0; w <= this->bottomShape[0][Wd] - this->kernelSize; w += this->stride)
      {
        vector<float> tmpMaxKernelVals(this->bottomShape[0][Cd]);
        vector<int> tmpMaxKernelIndexes(this->bottomShape[0][Cd]);
        memcpy(&tmpMaxKernelVals[0], bottom[0].GetPixel(n, h, w),
               this->bottomShape[0][Cd] * sizeof(float));
        for (size_t c = 0; c < tmpMaxKernelIndexes.size(); ++c)
        {
          tmpMaxKernelIndexes[c] = bottom[0].GetActualIndex(n, h, w, c);
        }
        for (int hK = 0; hK < this->kernelSize; ++hK)
        {
          for (int wK = 0; wK < this->kernelSize; ++wK)
          {
            float * pixelVal = bottom[0].GetPixel(n, h + hK, w + wK);
            for (size_t c = 0; c < tmpMaxKernelVals.size(); ++c)
            {
              if (pixelVal[c] > tmpMaxKernelVals[c])
              {
                tmpMaxKernelVals[c] = pixelVal[c];
                tmpMaxKernelIndexes[c] = bottom[0].GetActualIndex(n, h + hK, w + wK, c);
              }
            }
          }
        }
        memcpy(topDataVal + topOffset, &tmpMaxKernelVals[0],
               this->bottomShape[0][Cd] * sizeof(float));
        memcpy(&this->maxPoolingMap[0] + topOffset, &tmpMaxKernelIndexes[0],
               this->bottomShape[0][Cd] * sizeof(float));
        topOffset += this->bottomShape[0][Cd];
      }
    }
  }
}

void Pooling::Backward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  float * topGradientsVal = top[0].GetGradients();
  float * bottomGradientsVal = bottom[0].GetGradients();
  fill(bottomGradientsVal, bottomGradientsVal + bottom[0].GetSize(), 0.0);
  for (int i = 0; i < top[0].GetSize(); ++i)
  {
    bottomGradientsVal[this->maxPoolingMap[i]] += topGradientsVal[i];
  }
}

void Pooling::ComputeTopShape()
{
  this->topShape[0][Nd] = this->bottomShape[0][Nd];
  this->topShape[0][Hd] = (this->bottomShape[0][Hd] - this->kernelSize) / this->stride + 1;
  this->topShape[0][Wd] = (this->bottomShape[0][Wd] - this->kernelSize) / this->stride + 1;
  this->topShape[0][Cd] = this->bottomShape[0][Cd];
  this->maxPoolingMap = vector<int>(this->topShape[0][Nd] * this->topShape[0][Hd] *
                                    this->topShape[0][Wd] * this->topShape[0][Cd]);
}
