#ifndef CONVOLUTION_HPP
#define CONVOLUTION_HPP

#include "operator.hpp"
#include "operators/im2col.hpp"
#include <Eigen/Dense>
#include <random>
#include <math.h>

typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> RowMajorMap;
typedef Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> ColMajorMap;

class Convolution : public Operator
{
  public:
    Convolution(std::vector<std::vector<std::string>> IO, int _numberOfKernels,
                int _kernelSize, int _stride, int _pad, bool _bias,
                Tensor4D _kernels, Tensor4D _biases) : Operator("Convolution", IO),
                numberOfKernels(_numberOfKernels), kernelSize(_kernelSize),
                stride(_stride), pad(_pad), bias(_bias), kernels(_kernels),
                biases(_biases), im2col(_kernelSize, _stride, _pad){}
    void Forward(std::vector<Tensor4D> bottoms, std::vector<Tensor4D> tops);
    void Backward(std::vector<Tensor4D> bottoms, std::vector<Tensor4D> tops);
    void ComputeTopShape();
    void UpdateWeights(float learningRate, float momentum, float weightDecay);
    void InitWeights();
    void InitMomentums();
    int GetNumberOfKernels() const { return numberOfKernels; }
    int GetKernelSize() const { return kernelSize; }
    int GetStride() const { return stride; }
    int GetPad() const { return pad; }
    int GetBias() const { return bias; }
    Tensor4D GetKernels() const { return kernels; }
    Tensor4D GetBiases() const { return biases; }
    ~Convolution(){}

  private:
    const int numberOfKernels;
    const int kernelSize;
    const int stride;
    const int pad;
    const int bias;
    Tensor4D kernels;
    Tensor4D kernelsMomentum;
    Tensor4D biases;
    Tensor4D biasesMomentum;
    Im2Col im2col;
    Tensor4D col;
};

#endif
