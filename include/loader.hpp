#ifndef LOADER_HPP
#define LOADER_HPP

#include <utility>
#include <vector>

class Loader
{
  public:
    virtual void Load() = 0;
    std::pair<std::vector<float>, std::vector<std::vector<float>>> trainDataset;
    std::pair<std::vector<float>, std::vector<std::vector<float>>> testDataset;
    virtual ~Loader(){};
};

#endif
