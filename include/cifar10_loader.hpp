#ifndef CIFAR10_LOADER_HPP
#define CIFAR10_LOADER_HPP

#include <vector>
#include <iostream>
#include <utility>
#include <fstream>
#include <stdint.h>

class CIFAR10Loader
{
  public:
    CIFAR10Loader(std::string _datasetFolder, float _mean, float _scale) :
                  datasetFolder(_datasetFolder), mean(_mean), scale(_scale)
    {
      trainDataset = std::make_pair(std::vector<float>(50000), std::vector<std::vector<float>>(50000));
      testDataset = std::make_pair(std::vector<float>(10000), std::vector<std::vector<float>>(10000));
    }
    void Load();
    void LoadTrain();
    void LoadTest();
    std::vector<uint8_t> LoadBatch(std::string batchName);
    std::vector<float> LoadImage(std::vector<uint8_t> &batch, int imageIndex);
    void ApplyMean(std::vector<float> &image);
    void ApplyScale(std::vector<float> &image);
    std::pair<std::vector<float>, std::vector<std::vector<float>>> trainDataset;
    std::pair<std::vector<float>, std::vector<std::vector<float>>> testDataset;
    ~CIFAR10Loader(){};

  private:
    const std::string datasetFolder;
    const float mean;
    const float scale;
};

#endif
