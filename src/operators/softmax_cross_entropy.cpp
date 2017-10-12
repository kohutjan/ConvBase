#include "operators/softmax_cross_entropy.hpp"
#include <iostream>

using namespace std;

void SoftmaxCrossEntropy::Forward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  vector<float> expSum(this->bottomShape[0][Nd], 0.0);
  int bottomImageSize = this->bottomShape[0][Hd] * this->bottomShape[0][Wd] * this->bottomShape[0][Cd];
  float * bottomDataVal = bottom[0].GetData();
  for (int n = 0; n < this->bottomShape[0][Nd]; ++n)
  {
    for (int i = 0; i < bottomImageSize; ++i)
    {
      expSum[n] += exp(bottomDataVal[n * bottomImageSize + i]);
    }
  }
  float * softmaxTopDataVal = this->softmaxTop.GetData();
  float * topDataVal = top[0].GetData();
  float * topLabelVal = top[0].GetGradients();
  for (int n = 0; n < this->bottomShape[0][Nd]; ++n)
  {
    for (int i = 0; i < bottomImageSize; ++i)
    {
      float softmaxOutput = exp(bottomDataVal[n * bottomImageSize + i]) / expSum[n];
      softmaxTopDataVal[n * bottomImageSize + i] = softmaxOutput;
      if (i == topLabelVal[n])
      {
        topDataVal[n] = -log(softmaxOutput);
      }
    }
  }
}

void SoftmaxCrossEntropy::Backward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  int topSoftmaxImageSize = this->softmaxTop.GetShape()[Hd] * this->softmaxTop.GetShape()[Wd] * this->softmaxTop.GetShape()[Cd];
  float * bottomGradientsVal = bottom[0].GetGradients();
  float * softmaxTopDataVal = this->softmaxTop.GetData();
  float * topLabelVal = top[0].GetGradients();
  for (int n = 0; n < this->bottomShape[0][Nd]; ++n)
  {
    for (int i = 0; i < topSoftmaxImageSize; ++i)
    {
      bottomGradientsVal[n * topSoftmaxImageSize + i] = softmaxTopDataVal[n * topSoftmaxImageSize + i];
      if (i == int(topLabelVal[n]))
      {
        bottomGradientsVal[n * topSoftmaxImageSize + i] -= 1.0;
      }
    }
  }
}

void SoftmaxCrossEntropy::ComputeTopShape()
{
  this->topShape[0][Nd] = this->bottomShape[0][Nd];
  this->topShape[0][Hd] = 1;
  this->topShape[0][Wd] = 1;
  this->topShape[0][Cd] = 1;
  this->softmaxTop = Tensor4D(this->bottomShape[0]);
}
