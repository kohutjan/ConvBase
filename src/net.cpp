#include "net.hpp"

using namespace std;

void Net::PrintIO(vector<vector<string>> IO)
{
  cout << "bottoms: ";
  for (size_t i = 0; i < IO[0].size(); ++i)
  {
    cout << IO[0][i];
    if (i != (IO[0].size() - 1))
    {
      cout << ", ";
    }
  }
  cout << " | tops: ";
  for (size_t i = 0; i < IO[1].size(); ++i)
  {
    cout << IO[1][i];
    if (i != (IO[1].size() - 1))
    {
      cout << ",";
    }
  }
}

vector<vector<string>> Net::LoadIO(ifstream& modelStream)
{
  vector<vector<string>> IO(2);
  for (auto& io: IO)
  {
    int number;
    modelStream >> number;
    io.resize(number);
    for (auto& val: io)
    {
      modelStream >> val;
    }
  }
  return IO;
}

unique_ptr<Operator> Net::LoadConvolution(ifstream& modelStream)
{
  vector<vector<string>> IO = this->LoadIO(modelStream);
  vector<int> parameters(5);
  bool bias = true;
  for (auto& parameter: parameters)
  {
    modelStream >> parameter;
  }
  if (parameters[4] == 0)
  {
    bias = false;
  }
  Tensor4D kernels;
  Tensor4D biases;
  int numberOfWeights;
  modelStream >> numberOfWeights;
  if (numberOfWeights > 0)
  {
    int kernelsC;
    modelStream >> kernelsC;
    kernels = Tensor4D(parameters[0], parameters[1], parameters[1], kernelsC);
    float * kernelsVal = kernels.GetData();
    for (int i = 0; i < numberOfWeights; ++i)
    {
      modelStream >> kernelsVal[i];
    }
  }
  int numberOfBiases;
  modelStream >> numberOfBiases;
  if (numberOfBiases > 0)
  {
    biases = Tensor4D(1, 1, 1, numberOfBiases);
    float * biasesVal = biases.GetData();
    for (int i = 0; i < numberOfBiases; ++i)
    {
      modelStream >> biasesVal[i];
    }
  }

  cout << "type: Convolution | ";
  this->PrintIO(IO);
  cout << " | params: " << parameters[0] << "," << parameters[1] << ","
       << parameters[2] << "," << parameters[3] << "," << bias << endl;
  return unique_ptr<Operator>(new Convolution(IO, parameters[0], parameters[1],
                                              parameters[2], parameters[3],
                                              bias, kernels, biases));
}

unique_ptr<Operator> Net::LoadPooling(ifstream &modelStream)
{
  vector<vector<string>> IO = this->LoadIO(modelStream);
  vector<int> parameters(3);
  for (auto& parameter: parameters)
  {
    modelStream >> parameter;
  }
  cout << "type: Pooling | ";
  this->PrintIO(IO);
  cout << " | params: " << parameters[0] << ',' << parameters[1] << ","
       << parameters[2] << endl;
  return unique_ptr<Operator>(new Pooling(IO, parameters[0], parameters[1],
                                          parameters[2]));
}

unique_ptr<Operator> Net::LoadReLU(ifstream &modelStream)
{
  vector<vector<string>> IO = this->LoadIO(modelStream);
  cout << "type: ReLU | ";
  this->PrintIO(IO);
  cout << endl;
  return unique_ptr<Operator>(new ReLU(IO));
}

unique_ptr<Operator> Net::LoadReshape(ifstream &modelStream)
{
  vector<vector<string>> IO = this->LoadIO(modelStream);
  vector<int> parameters(3);
  for (auto& parameter: parameters)
  {
    modelStream >> parameter;
  }
  cout << "type: Reshape | ";
  this->PrintIO(IO);
  cout << " | params: " << parameters[0] << ',' << parameters[1] << ","
       << parameters[2] << endl;
  return unique_ptr<Operator>(new Reshape(IO, parameters[0], parameters[1],
                                          parameters[2]));
}

bool Net::Load(string modelName)
{
  cout << endl;
  cout << endl;
  cout << "Model name: " << modelName << endl;
  ifstream modelStream(modelName);
  if (modelStream.is_open())
  {
    return this->LoadFromStream(modelStream);
  }
  else
  {
    cerr << "Unable to open model file." << endl;
    return false;
  }
}

bool Net::LoadFromStream(ifstream &modelStream)
{
  this->operators.clear();
  cout << "Net structure:" << endl;
  cout << "#############################################################" << endl;
  string operatorName;

  while (getline(modelStream, operatorName))
  {
    if (operatorName == "Convolution")
    {
      this->operators.push_back(this->LoadConvolution(modelStream));
      continue;
    }
    if (operatorName == "Pooling")
    {
      this->operators.push_back(this->LoadPooling(modelStream));
      continue;
    }
    if (operatorName == "ReLU")
    {
      this->operators.push_back(this->LoadReLU(modelStream));
      continue;
    }
    if (operatorName == "Reshape")
    {
      this->operators.push_back(this->LoadReshape(modelStream));
      continue;
    }
  }
  modelStream.close();
  cout << "#############################################################" << endl;
  cout << endl;
  return true;
}

void Net::Init(vector<Tensor4D> input)
{
  //TODO generalize
  for (auto& op: this->operators)
  {
    op->ComputeOutputShape(input[0].GetN(), input[0].GetH(), input[0].GetW(),
                           input[0].GetC());
  }
}

vector<Tensor4D> Net::Forward(vector<Tensor4D> input)
{
   vector<Tensor4D> output = input;
   for (auto& op: this->operators)
   {
     output = op->Forward(output);
   }
   return output;
}
