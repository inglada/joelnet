#ifndef LOSS_H
#define LOSS_H

#include "Tensor.h"

namespace joelnet
{
/**
A loss function measures how good our predictions are, we can use this
to adjust the parameters of our network
*/
struct Loss
{
  virtual double loss(const Tensor& predicted, const Tensor& actual) = 0;
  virtual Tensor grad(const Tensor& predicted, const Tensor& actual) = 0;
};

/**
MSE is mean squared error, although we're just going to do total
squared error
*/
struct MSE : Loss
{
  double loss(const Tensor& predicted, const Tensor& actual) override
  {
    return xt::sum(xt::square(predicted - actual))[0];
  }
  Tensor grad(const Tensor& predicted, const Tensor& actual) override
  {
    return 2 * (predicted - actual);
  }
  };
}
#endif
