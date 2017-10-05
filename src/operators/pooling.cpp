#include "operators/pooling.hpp"

using namespace std;

vector<Tensor4D> Pooling::Forward(vector<Tensor4D> input)
{
  vector<Tensor4D> output;
  return output;
}

vector<Tensor4D> Pooling::Backward(vector<Tensor4D> input)
{
  vector<Tensor4D> output;
  return output;
}

void Pooling::ComputeOutputShape(int inputN, int inputH, int inputW, int inputC)
{
  this->outputN = inputN;
  this->outputC = inputC;  
}
