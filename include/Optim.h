#ifndef OPTIM_H
#define OPTIM_H

#include "NN.h"

namespace joelnet
{
/**
We use an optimizer to adjust the parameters
of our network based on the gradients computed
during backpropagation
*/
struct Optimizer
{
  virtual void step(NeuralNet net) = 0;
};

struct SGD : Optimizer
{
  SGD(double lr = 0.01) : lr{lr} {};

  void step(NeuralNet net)
  {
    for(auto& p : net.params_and_grads())
      *(p.first) -= lr* (*(p.second));
  }
  double lr;

};
}
#endif
