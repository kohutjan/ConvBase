#include "net.hpp"
#include "cifar10_loader.hpp"

using namespace std;


int main(int argc, char **argv)
{
  CIFAR10Loader loader(argv[1], 127, 0.007874016);
  loader.Load();
  uint8_t u = 5;
  cout << float(u) << endl;
  return 0;
}
