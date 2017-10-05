#include "operators/convolution.hpp"

using namespace std;
using namespace Eigen;

vector<Tensor4D> Convolution::Forward(vector<Tensor4D> input)
{
  Im2Col im2col(this->kernelSize, this->stride, this->pad);
  Tensor4D col = im2col.Forward(input)[0];

  Map<Matrix<float,
             Dynamic,
             Dynamic,
             RowMajor> >
             eigenCol(col.GetData(), col.GetH(), col.GetW());
  Map<Matrix<float,
             Dynamic,
             Dynamic,
             ColMajor> >
             eigenKernels(this->kernels.GetData(), this->kernelSize *
                          this->kernelSize * this->kernels.GetC(),
                          this->numberOfKernels);

  MatrixXf eigenOutput = eigenCol * eigenKernels;

  vector<Tensor4D> output(1, Tensor4D(this->outputN, this->outputH,
                                      this->outputW, this->outputC));
  memcpy(output[0].GetData(), eigenOutput.data(),
         this->outputN * this->outputH * this->outputW * this->outputC * sizeof(float));

  return output;
}

vector<Tensor4D> Convolution::Backward(vector<Tensor4D> input)
{
  vector<Tensor4D> output;
  return output;
}

void Convolution::ComputeOutputShape(int inputN, int inputH, int inputW, int inputC)
{
  this->outputN = inputN;
  this->outputH = (inputH + 2 * this->pad - this->kernelSize) / this->stride + 1;
  this->outputW = (inputW + 2 * this->pad - this->kernelSize) / this->stride + 1;
  this->outputC = this->numberOfKernels;
}
