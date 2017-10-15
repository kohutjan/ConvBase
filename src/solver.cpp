#include "solver.hpp"

using namespace std;

void Solver::Solve()
{
  int rightGuess = 0;
  for (int n = 0; n < numberOfIterations; ++n)
  {
    pair<Tensor4D, Tensor4D> batch = this->GetRandomTrainBatch();
    Tensor4D top = this->net.
    this->net.AddTensor4DToContainer(this->net.inputs.begin()->first, batch.second);
    this->net.Forward();
    rightGuesses += this->GetRightGuesses(batch, top);
    this->PrintAccuracy(n, interval, rightGuesses);
    this->net.Backward();
    this->net.UpdateWeights(learningRate);
  }
}

void Solver::PrintAccuracy(int n, int interval, int rightGuesses)
{
  if (n != 0)
  {
    if (n % interval == 0)
    {
      cout << "Iteration " << n << " | train accuracy: " << float(rightGuesses)
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
  uniform_int_distribution<int> indexDist(0, 50000);
  vector<int> randomIndexes(this->net.inputs.begin()->second[Nd]);
  for (int i = 0; i < randomIndexes.size(); ++i)
  {
    randomIndexes[i] = indexDist(mt);
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
  for (int i = 0; i < indexes.size(); ++i)
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
