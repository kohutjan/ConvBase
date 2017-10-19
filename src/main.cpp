#include "solver.hpp"
#include "loaders/cifar10_loader.hpp"
#include <getopt.h>

using namespace std;


int main(int argc, char **argv)
{
  static struct option long_options[] = {
  {"net", required_argument, 0, 'n'},
  {"solver", required_argument, 0, 's'},
  {"dataset", required_argument, 0, 'd'},
  {"init-weights", no_argument, 0, 'i'},
  {"save", required_argument, 0, 'a'},
  {0, 0, 0, 0}};

  string netFileName;
  string solverFileName;
  string datasetFileName;
  int initWeights = 0;
  string saveFileName;

  cout << endl;
  cout << "Params" << endl;
  cout << "#############################################################" << endl;
  int option_index = 0;
  int opt = 0;
  while ((opt = getopt_long(argc, argv, "n:s:d:ia:", long_options, &option_index)) != -1)
  {
    switch (opt)
    {
      case 'n':
        netFileName = optarg;
        cout << "Net file: " << optarg << endl;
        break;

      case 's':
        solverFileName = optarg;
        cout << "Solver file: " << optarg << endl;
        break;

      case 'd':
        datasetFileName = optarg;
        cout << "Dataset folder: " << optarg << endl;
        break;

      case 'i':
        initWeights = 1;
        cout << "Init weights set." << endl;
        break;

      case 'a':
        saveFileName = optarg;
        cout << "Save: " << optarg << endl;
        break;

      default:
        break;
    }
  }
  cout << "#############################################################" << endl;
  cout << endl;
  cout << endl;

  if (netFileName.empty() || solverFileName.empty() || datasetFileName.empty())
  {
    cout << "Net, Solver and Dataset paths have to be set." << endl;
    return -1;
  }

  CIFAR10Loader loader(datasetFileName, 127.0, 0.007874016);
  loader.Load();

  Net net;
  net.Load(netFileName);
  net.Init();
  if (initWeights)
  {
    net.InitWeights();
  }

  /*
  Solver solver(&loader, net);
  solver.Solve();
  */

  if (!saveFileName.empty())
  {
    net.Save(saveFileName);
  }

  return 0;
}
