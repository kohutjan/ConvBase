#include "net.hpp"

using namespace std;

void Net::PrintIO(vector<vector<string>> IO)
{
  cout << "bottom: ";
  for (size_t i = 0; i < IO[0].size(); ++i)
  {
    cout << IO[0][i];
    if (i != (IO[0].size() - 1))
    {
      cout << ", ";
    }
  }
  cout << " | top: ";
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
  return shared_ptr<Operator>(new Reshape(IO, parameters));
}

shared_ptr<Operator> Net::LoadSoftmaxCrossEntropy(ifstream &modelStream)
{
  vector<vector<string>> IO = this->LoadIO(modelStream);
  cout << "type: SoftmaxCrossEntropy | ";
  this->PrintIO(IO);
  cout << endl;
  return shared_ptr<Operator>(new SoftmaxCrossEntropy(IO));
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
  cout << "Net params:" << endl;
  cout << "#############################################################" << endl;
  while (!modelStream.eof())
  {
    string operatorName;
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
    if (operatorName == "SoftmaxCrossEntropy")
    {
      this->operators.push_back(this->LoadSoftmaxCrossEntropy(modelStream));
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
  int numberOfInputsOperators = 0;
  for (auto& op: this->operators)
  {
    vector<vector<int>> tmpBottomShape;
    for (auto& bottom: op->GetBottomName())
    {
      if (this->inputs.find(bottom) != this->inputs.end())
      {
        tmpBottomShape.push_back(this->inputs[bottom]);
      }
    }
    if (tmpBottomShape.size() == op->GetBottomName().size())
    {
      op->SetBottomShape(tmpBottomShape);
      op->ComputeTopShape();
      ++numberOfInputsOperators;
    }
  }

  for (int bottomIndex = numberOfInputsOperators; bottomIndex < this->operators.size(); ++bottomIndex)
  {
    vector<vector<int>> tmpBottomShape;
    for (int topIndex = 0; topIndex < bottomIndex; ++topIndex)
    {
      for (auto& bottom: this->operators[bottomIndex]->GetBottomName())
      {
        for (int t = 0; t < this->operators[topIndex]->GetTopName().size(); ++t)
        {
          if (bottom.compare(this->operators[topIndex]->GetTopName()[t]) == 0)
          {
            tmpBottomShape.push_back(this->operators[topIndex]->GetTopShape()[t]);
          }
        }
      }
    }
    if (tmpBottomShape.size() == this->operators[bottomIndex]->GetBottomName().size())
    {
      this->operators[bottomIndex]->SetBottomShape(tmpBottomShape);
      this->operators[bottomIndex]->ComputeTopShape();
    }
  }
  this->PrintShapes();
}

void Net::Forward()
{
  for (auto& op: this->operators)
  {
    vector<Tensor4D> tmpBottom;
    for (auto& bottomName: op->GetBottomName())
    {
      if (this->tensor4DContainer.find(bottomName) != this->tensor4DContainer.end())
      {
        tmpBottom.push_back(this->tensor4DContainer[bottomName]);
      }
    }
    vector<Tensor4D> tmpTop;
    for (int topIndex = 0; topIndex < op->GetTopName().size(); ++topIndex)
    {
      if (this->tensor4DContainer.find(op->GetTopName()[topIndex]) == this->tensor4DContainer.end())
      {
        this->tensor4DContainer[op->GetTopName()[topIndex]] = Tensor4D(op->GetTopName()[topIndex],
                                                                       op->GetTopShape()[topIndex]);
      }
      tmpTop.push_back(this->tensor4DContainer[op->GetTopName()[topIndex]]);
    }
    op->Forward(tmpBottom, tmpTop);
  }
}

void Net::Backward()
{
  for (auto op = this->operators.rbegin(); op != this->operators.rend(); ++op)
  {
    vector<Tensor4D> tmpTop;
    for (auto& topName: (*op)->GetTopName())
    {
      if (this->tensor4DContainer.find(topName) != this->tensor4DContainer.end())
      {
        tmpTop.push_back(this->tensor4DContainer[topName]);
      }
    }
    vector<Tensor4D> tmpBottom;
    for (int bottomIndex = 0; bottomIndex < (*op)->GetBottomName().size(); ++bottomIndex)
    {
      if (this->tensor4DContainer.find((*op)->GetBottomName()[bottomIndex]) == this->tensor4DContainer.end())
      {
        this->tensor4DContainer[(*op)->GetBottomName()[bottomIndex]] = Tensor4D((*op)->GetBottomName()[bottomIndex],
                                                                                (*op)->GetBottomShape()[bottomIndex]);
      }
      tmpBottom.push_back(this->tensor4DContainer[(*op)->GetBottomName()[bottomIndex]]);
    }
    (*op)->Backward(tmpBottom, tmpTop);
  }
}

void Net::UpdateWeights(float learningRate)
{
  for (auto& op: this->operators)
  {
    op->UpdateWeights(learningRate);
  }
}

void Net::PrintShapes()
{
  cout << "Net shapes:" << endl;
  cout << "#############################################################" << endl;
  for (auto& op: this->operators)
  {
    cout << "type: " << op->GetType() << " | ";
    cout << "bottom: ";
    for (int bottomIndex = 0; bottomIndex < op->GetBottomName().size(); ++bottomIndex)
    {
      cout << op->GetBottomName()[bottomIndex] << "(";
      for (int i = 0; i < op->GetBottomShape()[bottomIndex].size(); ++i)
      {
        cout << op->GetBottomShape()[bottomIndex][i];
        if (i != op->GetBottomShape()[bottomIndex].size() - 1)
        {
          cout << ",";
        }
        else
        {
          cout << ")";
        }
      }
      if (bottomIndex != op->GetBottomName().size() - 1)
      {
        cout << ", ";
      }
    }
    cout << " | top: ";
    for (int topIndex = 0; topIndex < op->GetBottomName().size(); ++topIndex)
    {
      cout << op->GetTopName()[topIndex] << "(";
      for (int i = 0; i < op->GetTopShape()[topIndex].size(); ++i)
      {
        cout << op->GetTopShape()[topIndex][i];
        if (i != op->GetTopShape()[topIndex].size() - 1)
        {
          cout << ",";
        }
        else
        {
          cout << ")";
        }
      }
      if (topIndex != op->GetTopName().size() - 1)
      {
        cout << ", ";
      }
    }
    cout << endl;
  }
  cout << "#############################################################" << endl;
}
