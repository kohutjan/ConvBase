#include "loaders/cifar10_loader.hpp"

using namespace std;

void CIFAR10Loader::Load()
{
  cout << "Loading CIFAR10..." << endl;
  cout << "#############################################################" << endl;
  this->LoadTrain();
  cout << endl;
  this->LoadTest();
  cout << "#############################################################" << endl;
  cout << endl;
  cout << endl;
}

void CIFAR10Loader::LoadTrain()
{
  cout << "Loading training datset..." << endl;
  for (int b = 0; b < 5; ++b)
  {
    vector<uint8_t> batch = this->LoadBatch(this->datasetFolder + "/data_batch_"
                                            + to_string(b + 1) + ".bin");
    for (int n = 0; n < 10000; ++n)
    {
      vector<float> image = this->LoadImage(batch, n);
      this->trainDataset.first[b * 10000 + n] = batch[n * (32 * 32 * 3 + 1)];
      this->trainDataset.second[b * 10000 + n] = image;
    }
  }
}

void CIFAR10Loader::LoadTest()
{
  cout << "Loading testing dataset..." << endl;
  vector<uint8_t> batch = this->LoadBatch(this->datasetFolder + "/test_batch.bin");
  for (int n = 0; n < 10000; ++n)
  {
    vector<float> image = this->LoadImage(batch, n);
    this->testDataset.first[n] = batch[n * (32 * 32 * 3 + 1)];
    this->testDataset.second[n] = image;
  }
}

vector<uint8_t> CIFAR10Loader::LoadBatch(string batchName)
{
  ifstream batchStream(batchName, ios::binary|ios::ate);
  ifstream::pos_type pos = batchStream.tellg();
  if (pos < 0)
  {
    pos = 0;
  }
  vector<uint8_t> batch(pos);
  batchStream.seekg(0, ios::beg);
  cout << "Reading " << batchName << endl;
  batchStream.read((char *) &batch[0], pos);
  return batch;
}

vector<float> CIFAR10Loader::LoadImage(vector<uint8_t> &batch, int imageIndex)
{
  vector<float> image(3072);
  int pixelIndex = 0;
  for (int hw = 0; hw < 1024; ++hw)
  {
    image[pixelIndex] = batch[(imageIndex * (32 * 32 * 3 + 1) + 1 + hw)];
    image[pixelIndex + 1] = batch[(imageIndex * (32 * 32 * 3 + 1) + 1 + hw + 32 * 32)];
    image[pixelIndex + 2] = batch[(imageIndex * (32 * 32 * 3 + 1) + 1 + hw + 2 * 32 * 32)];
    pixelIndex += 3;
  }
  if (this->mean != 0.0)
  {
    this->ApplyMean(image);
  }
  if (this->scale != 0.0)
  {
    this->ApplyScale(image);
  }
  return image;
}

void CIFAR10Loader::ApplyMean(vector<float> &image)
{
  for (int i = 0; i < 32 * 32 * 3; ++i)
  {
    image[i] -= this->mean;
  }
}


void CIFAR10Loader::ApplyScale(vector<float> &image)
{
  for (int i = 0; i < 32 * 32 * 3; ++i)
  {
    image[i] *= this->scale;
  }
}
