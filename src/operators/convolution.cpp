#include "operators/convolution.hpp"

using namespace std;
using namespace Eigen;

void Convolution::Forward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  this->im2col.Forward(bottom, vector<Tensor4D>(1, this->col));
  RowMajorMap eigenColData(this->col.GetData(), this->col.GetShape()[Hd],
                           this->col.GetShape()[Wd]);
  ColMajorMap eigenKernelsData(this->kernels.GetData(), this->kernelSize *
                               this->kernelSize * this->kernels.GetShape()[Cd],
                               this->numberOfKernels);

  RowMajorMap eigenTopData(top[0].GetData(), eigenColData.rows(),
                           eigenKernelsData.cols());

  eigenTopData = eigenColData * eigenKernelsData;

  if (this->bias)
  {
    Map<RowVectorXf> eigenBiasesData(this->biases.GetData(), this->biases.GetSize());
    eigenTopData = eigenTopData.array().rowwise() + eigenBiasesData.array();
  }
}

void Convolution::Backward(vector<Tensor4D> bottom, vector<Tensor4D> top)
{
  RowMajorMap eigenTopGradients(top[0].GetGradients(), this->topShape[0][Nd] *
                                this->topShape[0][Hd] * this->topShape[0][Wd],
                                this->topShape[0][Cd]);
  RowMajorMap eigenTransposeKernelsData(this->kernels.GetData(),
                                        this->numberOfKernels, this->kernelSize *
                                        this->kernelSize * this->kernels.GetShape()[Cd]);

  RowMajorMap eigenColGradients(this->col.GetGradients(), this->col.GetShape()[Hd],
                                this->col.GetShape()[Wd]);

  eigenColGradients = eigenTopGradients * eigenTransposeKernelsData;

  this->im2col.Backward(bottom, vector<Tensor4D>(1, this->col));

  ColMajorMap eigenTransposeColData(this->col.GetData(), this->col.GetShape()[Wd],
                                    this->col.GetShape()[Hd]);

  ColMajorMap eigenKernelsGradients(this->kernels.GetGradients(), this->kernelSize *
                                    this->kernelSize * this->kernels.GetShape()[Cd],
                                    this->numberOfKernels);

  eigenKernelsGradients = eigenTransposeColData * eigenTopGradients;

  if (this->bias)
  {
    Map<RowVectorXf> eigenBiasesGradients(this->biases.GetGradients(), this->biases.GetSize());
    eigenBiasesGradients = eigenTopGradients.colwise().sum();
  }
}

void Convolution::UpdateWeights(float learningRate, float momentum, float weightDecay)
{
  float * kernelsDataVal = this->kernels.GetData();
  float * kernelsGradientsVal = this->kernels.GetGradients();
  float * kernelsMomentumVal = this->kernelsMomentum.GetData();
  for (int i = 0; i < this->kernels.GetSize(); ++i)
  {
    kernelsMomentumVal[i] = momentum * kernelsMomentumVal[i] -
                            (kernelsGradientsVal[i] * (1.0 / this->topShape[0][Nd]) * learningRate) -
                            weightDecay * learningRate * kernelsDataVal[i];
    kernelsDataVal[i] += kernelsMomentumVal[i];
  }
  if (this->bias)
  {
    float * biasesDataVal = this->biases.GetData();
    float * biasesGradientsVal = this->biases.GetGradients();
    float * biasesMomentumVal = this->biasesMomentum.GetData();
    for (int i = 0; i < this->biases.GetSize(); ++i)
    {
      biasesMomentumVal[i] = momentum * biasesMomentumVal[i] -
                             (biasesGradientsVal[i] * (1.0 / this->topShape[0][Nd]) * learningRate) -
                             weightDecay * learningRate * biasesDataVal[i];
      biasesDataVal[i] += biasesMomentumVal[i];
    }
  }
}

void Convolution::InitWeights()
{
  this->kernels = Tensor4D(this->numberOfKernels, this->kernelSize,
                           this->kernelSize, this->bottomShape[0][Cd]);
  float scale = sqrt(3.0 / (this->kernels.GetShape()[Hd] * this->kernels.GetShape()[Wd]
                            * this->kernels.GetShape()[Cd]));
  random_device rd;
  mt19937 mt(rd());
  uniform_real_distribution<float> dist(-scale, scale);
  float * kernelsVal = this->kernels.GetData();
  for (int i = 0; i < this->kernels.GetSize(); ++i)
  {
    kernelsVal[i] = dist(mt);
  }
  if (this->bias)
  {
    this->biases = Tensor4D(1, 1, 1, this->numberOfKernels);
    float * biasesVal = this->biases.GetData();
    fill(biasesVal, biasesVal + this->biases.GetSize(), 0.0);
  }
}

void Convolution::InitMomentums()
{
  this->kernelsMomentum = Tensor4D(this->kernels.GetShape());
  fill(this->kernelsMomentum.GetData(), this->kernelsMomentum.GetData() +
       this->kernelsMomentum.GetSize(), 0.0);
  if (this->bias)
  {
    this->biasesMomentum = Tensor4D(this->biases.GetShape());
    fill(this->biasesMomentum.GetData(), this->biasesMomentum.GetData() +
         this->biasesMomentum.GetSize(), 0.0);
  }
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
