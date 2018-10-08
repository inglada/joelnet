#include <cstdlib>
#include "Tensor.h"
#include "Layers.h"
#include "Loss.h"
#include "NN.h"
#include "Optim.h"
#include "Data.h"
#include "Train.h"


void Xor()
{
  std::cout << "Tensors\n";
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

  std::cout << "Layers\n";
  Linear l1(2,2);
  Tanh t;
  Linear l2(2,2);
  std::cout << "Net\n";
  NeuralNet net({&l1, &t, &l2});
  std::cout << "Train\n";
  train(net, inputs, targets);
  auto predicted = net.forward(inputs);
  std::cout << predicted;
}


int main()
{
   Xor();
   return EXIT_SUCCESS;
}
