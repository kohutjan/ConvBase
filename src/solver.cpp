#include "solver.hpp"

using namespace std;

void Solver::Solve(int numberOfIterations, float learningRate)
{
  int rightGuess = 0;
  for (int n = 0; n < numberOfIterations; ++n)
  {
    pair<Tensor4D, Tensor4D> batch = this->GetRandomTrainBatch();
    this->net.AddTensor4DToContainer(this->net.inputs.begin()->first, batch.second);
    this->net.AddTensor4DToContainer("loss", batch.first);
    this->net.Forward();
    Tensor4D top = this->net.GetTensor4DFromContainer("top");
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
    if (n != 0)
    {
      if (n % 50 == 0)
      {
        cout << "Ieration " << n << " | train accuracy: " << float(rightGuess) / float(n * this->net.inputs.begin()->second[Nd]) << endl;;
      }
    }
    this->net.Backward();
    this->net.UpdateWeights(learningRate);
  }
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

pair<Tensor4D, Tensor4D> Solver::GetBatch(string datasetType, vector<int> randomIndexes)
{
  vector<int> inputShape = this->net.inputs.begin()->second;
  Tensor4D labels(inputShape[Nd], 1, 1, 1);
  Tensor4D data(inputShape[Nd], inputShape[Hd], inputShape[Wd], inputShape[Cd]);
  float * labelsVal = labels.GetGradients();
  float * dataVal = data.GetData();
  for (int i = 0; i < randomIndexes.size(); ++i)
  {
    vector<float> randomImage;
    if (datasetType == string("train"))
    {
      labelsVal[i] = this->loader->trainDataset.first[randomIndexes[i]];
      randomImage = this->loader->trainDataset.second[randomIndexes[i]];
    }
    else
    {
      labelsVal[i] = this->loader->testDataset.first[randomIndexes[i]];
      randomImage = this->loader->testDataset.second[randomIndexes[i]];
    }
    memcpy(dataVal + i * inputShape[Hd] * inputShape[Wd] * inputShape[Cd],
           &randomImage[0], inputShape[Hd] * inputShape[Wd] * inputShape[Cd]
           * sizeof(float));
  }
  return make_pair(labels, data);
}
