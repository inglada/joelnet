#include <cstdlib>
#include "Train.h"


void Xor()
{
  auto inputs = Tensor({
      {0, 0},
      {1, 0},
      {0, 1},
      {1, 1},
    });

  auto targets = Tensor({
      {1, 0},
      {0, 1},
      {0, 1},
      {1, 0},
    });

  Linear l1(2,2);
  Tanh t;
  Linear l2(2,2);
  NeuralNet net({&l1, &t, &l2});
  train(net, inputs, targets);
  auto predicted = net.forward(inputs);
  std::cout << predicted;
}


int main()
{
   Xor();
   return EXIT_SUCCESS;
}
