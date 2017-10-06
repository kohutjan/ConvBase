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
  vector<int> inputParams(4);
  for (auto& param: inputParams)
  {
    inputStream >> param;
  }
  vector<Tensor4D> input(1, Tensor4D(inputParams));
  float * inputVal = input[0].GetData();
  for (int i = 0; i < inputParams[0] * inputParams[1] * inputParams[2] *
       inputParams[3]; ++i)
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
  for (int i = 0; i < output[0].GetShape()[Nd] * output[0].GetShape()[Hd] *
       output[0].GetShape()[Wd] * output[0].GetShape()[Cd]; ++i)
  {
    outputStream << outputVal[i] << endl;
  }
  outputStream.close();

  return 0;
}
