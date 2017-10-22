#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "loader.hpp"
#include "net.hpp"

class Solver
{
  public:
    Solver(Loader * _loader, Net _net) : loader(_loader), net(_net)
    {
      setOutputNames();
    }
    Solver(Loader * _loader, Net _net, float _learningRate, float _momentum,
           int _trainIterations, int _testInterval, int _testIterations,
           int _displayInterval) : loader(_loader), net(_net),
           learningRate(_learningRate), momentum(_momentum),
           trainIterations(_trainIterations), testInterval(_testInterval),
           testIterations(_testIterations), displayInterval(_displayInterval)
    {
      setOutputNames();
    }
    bool Load(std::string modelName);
    bool LoadFromStream(std::ifstream &modelStream);
    void setOutputNames()
    {
      for (auto& op: this->net.operators)
      {
        if (op->GetType() == std::string("SoftmaxCrossEntropy"))
        {
          outputBottomName = op->GetBottomName()[0];
          outputTopName = op->GetTopName()[0];
        }
      }
    }
    void TestNet(int n);
    void PrintAccuracy(std::string type, int n, int interval, int * rightGuesses);
    int GetRightGuesses(std::pair<Tensor4D, Tensor4D> batch, Tensor4D top);
    std::pair<Tensor4D, Tensor4D> GetRandomTrainBatch();
    std::pair<Tensor4D, Tensor4D> GetBatch(std::string datasetType,
                                           std::vector<int> randomIndexes);
    void Solve();
    ~Solver(){}
  private:
    Loader * loader;
    Net net;
    float learningRate;
    float momentum;
    int trainIterations;
    int testInterval;
    int testIterations;
    int displayInterval;
    std::string outputBottomName;
    std::string outputTopName;
};

#endif
