#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "loader.hpp"
#include "net.hpp"

class Solver
{
  public:
    Solver(Loader * _loader, Net _net) : loader(_loader), net(_net){}
    std::pair<Tensor4D, Tensor4D> GetRandomTrainBatch();
    std::pair<Tensor4D, Tensor4D> GetBatch(std::string datasetType,
                                           std::vector<int> randomIndexes);
    void Solve(int numberOfIterations, float learningRate);
    ~Solver(){}
  private:
    Loader * loader;
    Net net;
};

#endif
