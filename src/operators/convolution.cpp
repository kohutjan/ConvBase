#include "operators/convolution.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;

void Convolution::Forward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  vector<Tensor4D> col(1, Tensor4D(im2col.GetTopShape()[0]));
  this->im2col.Forward(bottom, col);
  Map<Matrix<float,
             Dynamic,
             Dynamic,
             RowMajor> >
             eigenCol(col[0].GetData(), col[0].GetShape()[Hd], col[0].GetShape()[Wd]);
  Map<Matrix<float,
             Dynamic,
             Dynamic,
             ColMajor> >
             eigenKernels(this->kernels.GetData(), this->kernelSize *
                          this->kernelSize * this->kernels.GetShape()[Cd],
                          this->numberOfKernels);

  Matrix<float, Dynamic, Dynamic, RowMajor> eigenOutput = eigenCol * eigenKernels;

  /*
  cout << eigenCol << endl;
  cout << endl;
  cout << eigenKernels << endl;
  cout << endl;
  cout << eigenOutput << endl;
  */

  memcpy(top[0].GetData(), eigenOutput.data(),
         this->topShape[0][Nd] * this->topShape[0][Hd] * this->topShape[0][Wd] *
         this->topShape[0][Cd] * sizeof(float));
  if (this->bias)
  {
    float * outputPixel = top[0].GetData();
    float * biasesVal = this->biases.GetData();
    for (int i = 0; i < this->topShape[0][Nd] * this->topShape[0][Hd] *
         this->topShape[0][Wd]; ++i)
    {
      for (int c = 0; c < this->topShape[0][Cd]; ++c)
      {
        outputPixel[i * this->topShape[0][Cd] + c] += biasesVal[c];
      }
    }
  }
}

void Convolution::Backward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{

}

void Convolution::ComputeTopShape()
{
  this->topShape[0][Nd] = this->bottomShape[0][Nd];
  this->topShape[0][Hd] = (this->bottomShape[0][Hd] + 2 * this->pad - this->kernelSize) / this->stride + 1;
  this->topShape[0][Wd] = (this->bottomShape[0][Wd] + 2 * this->pad - this->kernelSize) / this->stride + 1;
  this->topShape[0][Cd] = this->numberOfKernels;
  im2col.SetBottomShape(this->bottomShape);
  im2col.ComputeTopShape();
}
