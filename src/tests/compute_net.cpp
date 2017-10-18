#include "tests/tensor4d_file_manager.hpp"
#include "net.hpp"

using namespace std;

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
    storeTensor4D(conv->GetKernels(), argv[3] + string("weights_convbase_output.txt"), true);
    if (conv->GetBias())
    {
      storeTensor4D(conv->GetBiases(), argv[3] + string("biases_convbase_output.txt"), true);
    }
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
