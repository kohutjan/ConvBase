#include "solver.hpp"

using namespace std;

void Solver::Solve()
{
  int rightGuesses = 0;
  for (int n = 0; n < this->trainIterations; ++n)
  {
    pair<Tensor4D, Tensor4D> batch = this->GetRandomTrainBatch();
    this->net.AddTensor4DToContainer(this->outputTopName, batch.first);
    this->net.AddTensor4DToContainer(this->net.inputs.begin()->first, batch.second);
    this->net.Forward();
    rightGuesses += this->GetRightGuesses(batch, this->net.GetTensor4DFromContainer(this->outputBottomName));
    this->PrintAccuracy("Train", n, this->displayInterval, rightGuesses);
    this->net.Backward();
    this->net.UpdateWeights(this->learningRate);
    this->TestNet(n);
  }
  cout << endl;
  cout << "#############################################################" << endl;
  this->PrintAccuracy("Train", this->trainIterations, this->trainIterations, rightGuesses);
  cout << "#############################################################" << endl;
  cout << endl;
}

void Solver::TestNet(int n)
{
  if (n != 0)
  {
    if (n % this->testInterval == 0)
    {
      int rightGuesses = 0;
      for (int i = 0; i < this->testIterations; ++i)
      {
        vector<int> indexes(this->net.inputs.begin()->second[Nd]);
        for (int j = 0; j < static_cast<int>(indexes.size()); ++j)
        {
          indexes[j] = (i * indexes.size() + j) % this->loader->testDataset.first.size();
        }
        pair<Tensor4D, Tensor4D> batch = this->GetBatch("test", indexes);
        this->net.AddTensor4DToContainer(this->outputTopName, batch.first);
        this->net.AddTensor4DToContainer(this->net.inputs.begin()->first, batch.second);
        this->net.Forward();
        rightGuesses += this->GetRightGuesses(batch, this->net.GetTensor4DFromContainer(this->outputBottomName));
      }
      cout << endl;
      cout << "#############################################################" << endl;
      this->PrintAccuracy("Test", this->testIterations, this->testIterations, rightGuesses);
      cout << "#############################################################" << endl;
      cout << endl;
    }
  }
}

void Solver::PrintAccuracy(string type, int n, int interval, int rightGuesses)
{
  if (n != 0)
  {
    if (n % interval == 0)
    {
      cout << n << " iterations | ";
      cout << type << " accuracy: " << float(rightGuesses)
           / float(n * this->net.inputs.begin()->second[Nd]) << endl;
    }
  }
}

int Solver::GetRightGuesses(pair<Tensor4D, Tensor4D> batch, Tensor4D top)
{
  int rightGuess = 0;
  float * labelVal = top.GetData();
  for (int j = 0; j < top.GetShape()[Nd]; ++j)
  {
    float maxLabelVal = labelVal[j * top.GetShape()[Cd]];
    int topLabel = 0;
    for (int i = 1; i < top.GetShape()[Cd]; ++i)
    {
      if (labelVal[top.GetShape()[Cd] * j + i] > maxLabelVal)
      {
        maxLabelVal = labelVal[top.GetShape()[Cd] * j + i];
        topLabel = i;
      }
    }
    if (topLabel == int(batch.first.GetGradients()[j]))
    {
      rightGuess++;
    }
  }
  return rightGuess;
}

pair<Tensor4D, Tensor4D> Solver::GetRandomTrainBatch()
{
  random_device rd;
  mt19937 mt(rd());
  uniform_int_distribution<int> indexDist(0, this->loader->trainDataset.first.size() - 1);
  vector<int> randomIndexes(this->net.inputs.begin()->second[Nd]);
  for (auto& randomIndex: randomIndexes)
  {
    randomIndex = indexDist(mt);
  }
  return this->GetBatch("train", randomIndexes);
}

pair<Tensor4D, Tensor4D> Solver::GetBatch(string datasetType, vector<int> indexes)
{
  vector<int> inputShape = this->net.inputs.begin()->second;
  Tensor4D labels(inputShape[Nd], 1, 1, 1);
  Tensor4D data(inputShape[Nd], inputShape[Hd], inputShape[Wd], inputShape[Cd]);
  float * labelsVal = labels.GetGradients();
  float * dataVal = data.GetData();
  for (int i = 0; i < static_cast<int>(indexes.size()); ++i)
  {
    vector<float> randomImage;
    if (datasetType == string("train"))
    {
      labelsVal[i] = this->loader->trainDataset.first[indexes[i]];
      randomImage = this->loader->trainDataset.second[indexes[i]];
    }
    else
    {
      labelsVal[i] = this->loader->testDataset.first[indexes[i]];
      randomImage = this->loader->testDataset.second[indexes[i]];
    }
    memcpy(dataVal + i * inputShape[Hd] * inputShape[Wd] * inputShape[Cd],
           &randomImage[0], inputShape[Hd] * inputShape[Wd] * inputShape[Cd]
           * sizeof(float));
  }
  return make_pair(labels, data);
}
