#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "loader.hpp"
#include "net.hpp"

class Solver
{
  public:
    Solver(Loader * _loader, Net _net, float _learningRate, int _trainIterations,
           int _testInterval, int _testIterations ) : loader(_loader), net(_net),
           learningRate(_learningRate), trainIterations(_trainIterations),
           testInterval(_testInterval), testIterations(_testIterations)
    {
      for (auto& op: this->net.operators)
      {
        if (op->GetType() == string("SoftmaxCrossEntropy"))
        {
          outputBottomName = op->GetBottomName()[0];
          outputTopName = op->GetTopName()[0];
        }
      }
    }
    std::pair<Tensor4D, Tensor4D> GetRandomTrainBatch();
    std::pair<Tensor4D, Tensor4D> GetBatch(std::string datasetType,
                                           std::vector<int> randomIndexes);
    void Solve();
    ~Solver(){}
  private:
    Loader * loader;
    Net net;
    const float learningRate;
    const int trainIterations;
    const int testInterval;
    const int testIterations;
    string outputBottomName;
    string outputTopName;
};

#endif
