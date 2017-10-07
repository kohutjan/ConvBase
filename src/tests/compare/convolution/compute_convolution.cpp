#include "net.hpp"

using namespace std;

Tensor4D loadTensor4D(string tensorFile, bool gradients);
void storeTensor4D(Tensor4D tensor, string tensorFile, bool gradients);
Tensor4D testForward(string tensorFile, Net &net);
Tensor4D testBackward(string tensorFile, Net &net);

int main(int argc, char **argv)
{
  Net net;
  net.Load(argv[1]);
  net.Init();

  if (argv[4] == string("forward"))
  {
    storeTensor4D(testForward(argv[2], net), argv[3], false);
  }
  if (argv[4] == string("backward"))
  {
    storeTensor4D(testBackward(argv[2], net), argv[3], true);
  }

  return 0;
}

Tensor4D testForward(string tensorFile, Net &net)
{
  vector<Tensor4D> bottom(1, loadTensor4D(tensorFile, false));
  vector<Tensor4D> top = net.Forward(bottom);
  return top[0];
}

Tensor4D testBackward(string tensorFile, Net &net)
{
  vector<Tensor4D> top(1, loadTensor4D(tensorFile, true));
  vector<Tensor4D> bottom = net.Backward(top);
  return bottom[0];
}

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
    cerr << "Unable to open with tensor data." << endl;
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
    cerr << "Unable to open file for storing tensor." << endl;
  }
}
