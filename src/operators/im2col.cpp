#include "operators/im2col.hpp"

using namespace std;

vector<Tensor4D> Im2Col::Forward(vector<Tensor4D> input)
{
  int inputN = input[0].GetN();
  int inputH = input[0].GetH() + 2 * this->pad;
  int inputW = input[0].GetW() + 2 * this->pad;
  int inputC = input[0].GetC();

  int outputN = 1;
  int outputH = ((inputH - this->kernelSize) / this->stride + 1) *
                ((inputW - this->kernelSize) / this->stride + 1) *
                inputN;
  int outputW = this->kernelSize * this->kernelSize * inputC;
  int outputC = 1;
  vector<Tensor4D> output(1, Tensor4D(outputN, outputH, outputW, outputC));
  float * outputData = output[0].GetData();
  int outputOffset = 0;

  for (int n = 0; n < inputN; ++n)
  {
    for (int h = 0; h < inputH; h += this->stride)
    {
      for (int w = 0; w < inputW; w += this->stride)
      {
        for (int hK = 0; hK < this->kernelSize; ++hK)
        {
          if (pad)
          {
            for (int wK = 0; wK < this->kernelSize; ++wK)
            {
              bool insidePad = ((h + hK) < this->pad) || ((w + wK) < this->pad) ||
                               ((h + hK) >= inputH - this->pad) ||
                               ((w + wK) >= inputW - this->pad);
              if (insidePad)
              {
                fill(outputData + outputOffset, outputData + outputOffset + inputC, 0.0);
              }
              else
              {
                float * inputPixel = input[0].GetPixel(n, h - this->pad + hK, w - this->pad + wK);
                memcpy(outputData + outputOffset, inputPixel, inputC * sizeof(float));
              }
              outputOffset += inputC;
            }
          }
          else
          {
            float * inputPixel = input[0].GetPixel(n, h - this->pad + hK, w - this->pad);
            memcpy(outputData + outputOffset, inputPixel, this->kernelSize * inputC * sizeof(float));
            outputOffset += this->kernelSize * inputC;
          }
        }
      }
    }
  }

  return output;
}

vector<Tensor4D> Im2Col::Backward(vector<Tensor4D> input)
{
  vector<Tensor4D> output;
  return output;
}
