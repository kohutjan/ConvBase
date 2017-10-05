#include "operators/relu.hpp"

using namespace std;

vector<Tensor4D> ReLU::Forward(vector<Tensor4D> input)
{
  vector<Tensor4D> output;
  return output;
}

vector<Tensor4D> ReLU::Backward(vector<Tensor4D> input)
{
  vector<Tensor4D> output;
  return output;
}

void ReLU::ComputeOutputShape(int inputN, int inputH, int inputW, int inputC)
{
  this->outputN = inputN;
  this->outputH = inputH;
  this->outputW = inputW;
  this->outputC = inputC;
}
