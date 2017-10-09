#include "operators/relu.hpp"

using namespace std;

void ReLU::Forward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  float * bottomDataVal = bottom[0].GetData();
  float * topDataVal = top[0].GetData();
  int bottomSize = bottom[0].GetSize();
  for (int i = 0; i < bottomSize; ++i)
  {
    if (bottomDataVal[i] < 0.0)
    {
      topDataVal[i] = 0.0;
    }
  }
}

void ReLU::Backward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  float * topGradientsVal = top[0].GetGradients();
  float * bottomDataVal = bottom[0].GetData();
  float * bottomGradientsVal = bottom[0].GetGradients();
  int topSize = top[0].GetSize();
  for (int i = 0; i < topSize; ++i)
  {
    bottomGradientsVal[i] = (bottomDataVal[i] > 0 ? topGradientsVal[i] : 0.0);
  }
}

void ReLU::ComputeTopShape()
{
  this->topShape = this->bottomShape;
}
