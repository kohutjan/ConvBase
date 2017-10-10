#ifndef NET_HPP
#define NET_HPP

#include "operators/convolution.hpp"
#include "operators/pooling.hpp"
#include "operators/relu.hpp"
#include "operators/reshape.hpp"
#include "operators/softmax_cross_entropy.hpp"
#include <iostream>
#include <fstream>
#include <map>

class Net
{
  public:
    Net()
    {
      inputs = std::map<std::string, std::vector<int>>();
      tensor4DContainer = std::map<std::string, Tensor4D>();
    }
    bool Load(std::string modelName);
    void Init();
    void PrintShapes();
    void Forward();
    void Backward();
    void UpdateWeights(float learningRate);
    std::map<std::string, Tensor4D> GetTensor4DContainer() const { return tensor4DContainer; }
    void AddTensor4DToContainer(std::string name, Tensor4D tensor)
    {
      tensor.SetName(name);
      tensor4DContainer[name] = tensor;
    }
    Tensor4D GetTensor4DFromContainer(std::string name)
    {
      if (tensor4DContainer.find(name) != tensor4DContainer.end())
      {
        return tensor4DContainer[name];
      }
    }
    bool LoadFromStream(std::ifstream &modelStream);
    std::shared_ptr<Operator> LoadConvolution(std::ifstream &modelStream);
    std::shared_ptr<Operator> LoadPooling(std::ifstream &modelStream);
    std::shared_ptr<Operator> LoadReLU(std::ifstream &modelStream);
    std::shared_ptr<Operator> LoadReshape(std::ifstream &modelStream);
    std::shared_ptr<Operator> LoadSoftmaxCrossEntropy(std::ifstream &modelStream);
    std::vector<std::shared_ptr<Operator>> operators;
    ~Net(){};
  private:
    std::vector<std::vector<std::string>> LoadIO(std::ifstream &modelStream);
    void PrintIO(std::vector<std::vector<std::string>> IO);
    void LoadInput(std::ifstream &modelStream);
    std::map<std::string, std::vector<int>> inputs;
    std::map<std::string, Tensor4D> tensor4DContainer;
};

#endif
