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

void Net::LoadInput(ifstream& modelStream)
{
  int numberOfInputs;
  modelStream >> numberOfInputs;
  for (int i = 0; i < numberOfInputs; ++i)
  {
    this->inputs = map<string, vector<int>>();
    string inputName;
    modelStream >> inputName;
    vector<int> inputParams(4);
    for (auto& param: inputParams)
    {
      modelStream >> param;
    }
    this->inputs[inputName] = inputParams;
  }

}

shared_ptr<Operator> Net::LoadConvolution(ifstream& modelStream)
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
  int kernelsC;
  modelStream >> kernelsC;
  if (kernelsC > 0)
  {
    kernels = Tensor4D(parameters[0], parameters[1], parameters[1], kernelsC);
    float * kernelsVal = kernels.GetData();
    for (int i = 0; i < parameters[0] * parameters[1] * parameters[1] *
         kernelsC; ++i)
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
  return shared_ptr<Operator>(new Convolution(IO, parameters[0], parameters[1],
                                              parameters[2], parameters[3],
                                              bias, kernels, biases));
}

shared_ptr<Operator> Net::LoadPooling(ifstream &modelStream)
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
  return shared_ptr<Operator>(new Pooling(IO, parameters[0], parameters[1],
                                          parameters[2]));
}

shared_ptr<Operator> Net::LoadReLU(ifstream &modelStream)
{
  vector<vector<string>> IO = this->LoadIO(modelStream);
  cout << "type: ReLU | ";
  this->PrintIO(IO);
  cout << endl;
  return shared_ptr<Operator>(new ReLU(IO));
}

shared_ptr<Operator> Net::LoadReshape(ifstream &modelStream)
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
  return shared_ptr<Operator>(new Reshape(IO, parameters[0], parameters[1],
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

  while (!modelStream.eof())
  {
    modelStream >> operatorName;
    if (operatorName == "Input")
    {
      this->LoadInput(modelStream);
      continue;
    }
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

void Net::Init()
{
  //TODO generalize
  this->operators[0]->SetBottomShape(vector<vector<int>>(1, this->inputs.begin()->second));
  this->operators[0]->ComputeTopShape();
}

vector<Tensor4D> Net::Forward(vector<Tensor4D> bottom)
{
   //TODO generalize
   vector<Tensor4D> top(1, Tensor4D(this->operators[0]->GetTopShape()[0]));
   this->operators[0]->Forward(bottom, top);
   return top;
}

vector<Tensor4D> Net::Backward(vector<Tensor4D> top)
{
  //TODO generalize
  vector<Tensor4D> bottom(1, Tensor4D(this->operators[0]->GetBottomShape()[0]));
  this->operators[0]->Backward(bottom, top);
  return bottom;
}
