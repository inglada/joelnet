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
    return xt::eval(xt::sum(xt::square(predicted - actual)))[0];
  }
  Tensor grad(const Tensor& predicted, const Tensor& actual) override
  {
    return xt::eval(2 * (predicted - actual));
  }
  };
#endif
