#ifndef LOSS_H
#define LOSS_H

#include "Tensor.h"

struct Loss
{
  virtual double loss(const Tensor& predicted, const Tensor& actual) = 0;
  virtual Tensor grad(const Tensor& predicted, const Tensor& actual) = 0;
};

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
#endif
