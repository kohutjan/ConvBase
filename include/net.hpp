#ifndef NET_HPP
#define NET_HPP

#include "operators/convolution.hpp"
#include "operators/pooling.hpp"
#include "operators/relu.hpp"
#include "operators/reshape.hpp"
#include <iostream>
#include <fstream>
#include <map>

class Net
{
  public:
    bool Load(std::string modelName);
    void Init();
    std::vector<Tensor4D> Forward(std::vector<Tensor4D> bottom);
    std::vector<Tensor4D> Backward(std::vector<Tensor4D> top);
    bool LoadFromStream(std::ifstream &modelStream);
    std::unique_ptr<Operator> LoadConvolution(std::ifstream &modelStream);
    std::unique_ptr<Operator> LoadPooling(std::ifstream &modelStream);
    std::unique_ptr<Operator> LoadReLU(std::ifstream &modelStream);
    std::unique_ptr<Operator> LoadReshape(std::ifstream &modelStream);
    std::vector<std::unique_ptr<Operator>> operators;
    ~Net(){};
  private:
    std::vector<std::vector<std::string>> LoadIO(std::ifstream &modelStream);
    void PrintIO(std::vector<std::vector<std::string>> IO);
    void LoadInput(std::ifstream &modelStream);
    std::map<std::string, std::vector<int>> inputs;
};

#endif
