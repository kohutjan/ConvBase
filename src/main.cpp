#include "net.hpp"

using namespace std;


int main(int argc, char **argv)
{
  Net net;
  net.Load(argv[1]);
  net.Save(argv[2]);
  return 0;
}
