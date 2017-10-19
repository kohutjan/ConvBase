#include "tests/tensor4d_file_manager.hpp"

using namespace std;

Tensor4D loadTensor4D(string tensorFile, bool gradients)
{
  Tensor4D tensor;
  ifstream tensorStream(tensorFile);
  if (tensorStream.is_open())
  {
    vector<int> tensorParams(4);
    for (auto& param: tensorParams)
    {
      tensorStream >> param;
    }
    tensor = Tensor4D(tensorParams);
    float * tensorVal;
    if (!gradients)
    {
      tensorVal = tensor.GetData();
    }
    else
    {
      tensorVal = tensor.GetGradients();
    }
    for (int i = 0; i < tensorParams[0] * tensorParams[1] * tensorParams[2] *
         tensorParams[3]; ++i)
    {
      tensorStream >> tensorVal[i];
    }
    tensorStream.close();
  }
  else
  {
    cerr << "Unable to open file(" << tensorFile << ") with tensor data." << endl;
  }
  return tensor;
}

void storeTensor4D(Tensor4D tensor, string tensorFile, bool gradients)
{
  ofstream tensorStream(tensorFile);
  if (tensorStream.is_open())
  {
    float * tensorVal;
    if (!gradients)
    {
      tensorVal = tensor.GetData();
    }
    else
    {
      tensorVal = tensor.GetGradients();
    }
    for (int i = 0; i < tensor.GetShape()[Nd] * tensor.GetShape()[Hd] *
         tensor.GetShape()[Wd] * tensor.GetShape()[Cd]; ++i)
    {
      tensorStream << tensorVal[i] << endl;
    }
    tensorStream.close();
  }
  else
  {
    cerr << "Unable to open file(" << tensorFile << ") for storing tensor." << endl;
  }
}
