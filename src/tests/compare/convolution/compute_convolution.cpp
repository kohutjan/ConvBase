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
  if (argv[4] == string("weights"))
  {
    storeTensor4D(testForward(argv[2] + string("data_input.txt"), net),
                  argv[3] + string("forward_convbase_output.txt"), false);
    storeTensor4D(testBackward(argv[2] + string("gradients_input.txt"), net),
                  argv[3] + string("backward_convbase_output.txt"), true);
    Convolution * conv = dynamic_cast<Convolution*>(net.operators[0].get());
    storeTensor4D(conv->GetKernels(),  argv[3] + string("weights_convbase_output.txt"), true);
  }

  return 0;
}

Tensor4D testForward(string tensorFile, Net &net)
{
  Tensor4D bottom = loadTensor4D(tensorFile, false);
  net.AddTensor4DToContainer("bottom", bottom);
  net.Forward();
  return net.GetTensor4DFromContainer("top");
}

Tensor4D testBackward(string tensorFile, Net &net)
{
  Tensor4D top = loadTensor4D(tensorFile, true);
  net.AddTensor4DToContainer("top", top);
  net.Backward();
  return net.GetTensor4DFromContainer("bottom");
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
