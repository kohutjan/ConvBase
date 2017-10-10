#include "operators/reshape.hpp"

using namespace std;

void Reshape::Forward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  if (this->bottomName[0].compare(this->topName[0]) != 0)
  {
    memcpy(top[0].GetData(), bottom[0].GetData(), bottom[0].GetSize() * sizeof(float));
  }
  top[0].SetShape({this->topShape[0][Nd], this->topShape[0][Hd],
                   this->topShape[0][Wd], this->topShape[0][Cd]});
}

void Reshape::Backward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  if (this->bottomName[0].compare(this->topName[0]) != 0)
  {
    memcpy(bottom[0].GetGradients(), top[0].GetGradients(), top[0].GetSize() * sizeof(float));
  }
  bottom[0].SetShape({this->bottomShape[0][Nd], this->bottomShape[0][Hd],
                      this->bottomShape[0][Wd], this->bottomShape[0][Cd]});
}

void Reshape::ComputeTopShape()
{
  this->topShape[0][Nd] = this->bottomShape[0][Nd];
  this->topShape[0][Hd] = this->shape[0];
  this->topShape[0][Wd] = this->shape[1];
  this->topShape[0][Cd] = this->shape[2];
}
