#include "net.hpp"

using namespace std;

int main(int argc, char **argv)
{
  string modelName(argv[1]);
  Net net;
  net.Load(modelName);
  return 0;
}
