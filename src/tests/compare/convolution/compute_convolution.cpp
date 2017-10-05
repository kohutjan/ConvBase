#include "net.hpp"

using namespace std;


int main(int argc, char **argv)
{
  string inputFile(argv[1]);

  ifstream inputStream(inputFile);
  if (!inputStream.is_open())
  {
    cerr << "Unable to open input file." << endl;
    return 1;
  }
  vector<int> inputParmas(4);
  for (auto& param: inputParmas)
  {
    inputStream >> param;
  }
  vector<Tensor4D> input(1, Tensor4D(inputParmas[0],
                                     inputParmas[1],
                                     inputParmas[2],
                                     inputParmas[3]));
  float * inputVal = input[0].GetData();
  for (int i = 0; i < inputParmas[0] * inputParmas[1] * inputParmas[2] *
       inputParmas[3]; ++i)
  {
    inputStream >> inputVal[i];
  }
  inputStream.close();

  string paramsFile(argv[2]);
  Net net;
  net.Load(paramsFile);
  net.Init(input);
  vector<Tensor4D> output = net.Forward(input);

  ofstream outputStream(argv[3]);
  if (!outputStream.is_open())
  {
    cout << "Unable to open output file." << endl;
    return 2;
  }
  float * outputVal = output[0].GetData();
  for (int i = 0; i < output[0].GetN() * output[0].GetH() * output[0].GetW()
       * output[0].GetC(); ++i)
  {
    outputStream << outputVal[i] << endl;
  }
  outputStream.close();

  return 0;
}
