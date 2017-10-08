#include "operators/convolution.hpp"
#include <iostream>

using namespace std;
using namespace Eigen;

void Convolution::Forward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  this->im2col.Forward(bottom, vector<Tensor4D>(1, this->col));
  Map<Matrix<float,
             Dynamic,
             Dynamic,
             RowMajor>>
             eigenColData(this->col.GetData(), this->col.GetShape()[Hd],
                          this->col.GetShape()[Wd]);
  Map<Matrix<float,
             Dynamic,
             Dynamic,
             ColMajor>>
             eigenKernels(this->kernels.GetData(), this->kernelSize *
                          this->kernelSize * this->kernels.GetShape()[Cd],
                          this->numberOfKernels);

  Map<Matrix<float,
             Dynamic,
             Dynamic,
             RowMajor>>
             eigenTopData(top[0].GetData(), eigenColData.rows(),
                          eigenKernels.cols());

  eigenTopData = eigenColData * eigenKernels;

  /*
  cout << eigenCol << endl;
  cout << endl;
  cout << eigenKernels << endl;
  cout << endl;
  cout << eigenOutput << endl;
  */

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
  Map<Matrix<float,
             Dynamic,
             Dynamic,
             RowMajor>>
             eigenTopGradients(top[0].GetGradients(), this->topShape[0][Nd] *
                               this->topShape[0][Hd] * this->topShape[0][Wd],
                               this->topShape[0][Cd]);
  Map<Matrix<float,
             Dynamic,
             Dynamic,
             RowMajor>>
             eigenTransposeKernels(this->kernels.GetData(),
                                   this->numberOfKernels, this->kernelSize *
                                   this->kernelSize * this->kernels.GetShape()[Cd]);

  Map<Matrix<float,
             Dynamic,
             Dynamic,
             RowMajor>>
             eigenColGradients(this->col.GetGradients(), this->col.GetShape()[Hd],
                               this->col.GetShape()[Wd]);

  eigenColGradients = eigenTopGradients * eigenTransposeKernels;

  /*
  cout << eigenTopGradients << endl;
  cout << endl;
  cout << eigenTransposeKernels << endl;
  cout << endl;
  cout << eigenColGradients << endl;
  */

  this->im2col.Backward(bottom, vector<Tensor4D>(1, this->col));

  Map<Matrix<float,
             Dynamic,
             Dynamic,
             ColMajor>>
             eigenTransposeColData(this->col.GetData(), this->col.GetShape()[Wd],
                                   this->col.GetShape()[Hd]);

  Map<Matrix<float,
             Dynamic,
             Dynamic,
             ColMajor>>
             eigenKernelsGradients(this->kernels.GetGradients(), this->kernelSize *
                                   this->kernelSize * this->kernels.GetShape()[Cd],
                                   this->numberOfKernels);

  eigenKernelsGradients = eigenTransposeColData * eigenTopGradients;
}

void Convolution::ComputeTopShape()
{
  this->topShape[0][Nd] = this->bottomShape[0][Nd];
  this->topShape[0][Hd] = (this->bottomShape[0][Hd] + 2 * this->pad - this->kernelSize) / this->stride + 1;
  this->topShape[0][Wd] = (this->bottomShape[0][Wd] + 2 * this->pad - this->kernelSize) / this->stride + 1;
  this->topShape[0][Cd] = this->numberOfKernels;
  im2col.SetBottomShape(this->bottomShape);
  im2col.ComputeTopShape();
  this->col = Tensor4D(im2col.GetTopShape()[0]);
}
