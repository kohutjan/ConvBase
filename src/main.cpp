#include "solver.hpp"
#include "loaders/cifar10_loader.hpp"

using namespace std;


int main(int argc, char **argv)
{
  CIFAR10Loader loader(argv[1], 127.0, 0.007874016);
  loader.Load();
  Net net;
  net.Load(argv[2]);
  net.Init();
  //net.InitWeights();
  Solver solver(&loader, net, 0.1, 2700, 100, 100, 5);
  //solver.TestNet(100);
  solver.Solve();
  //net.Save(argv[3]);
  return 0;
}
