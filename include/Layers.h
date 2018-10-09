#ifndef LAYERS_H
#define LAYERS_H
#include "Tensor.h"
#include <map>

struct Layer
{
  virtual Tensor forward(const Tensor& inputs) = 0;
  virtual Tensor backward(const Tensor& grad) = 0;
  std::map<std::string, Tensor> params;
  std::map<std::string, Tensor> grads;
  Tensor inputs;
};

struct Linear : Layer
{
  Linear(size_t input_size, size_t output_size) : input_size{input_size},
                                                  output_size{output_size} 
  {
    params["w"] = xt::random::randn<double>({input_size, output_size});
    params["b"] = xt::random::randn<double>({output_size});
  }

  Tensor forward(const Tensor& i)
  {
    inputs = i;
    return xt::linalg::dot(inputs, params["w"])+params["b"];
  }

  Tensor backward(const Tensor& grad)
  {
    grads["b"] = xt::sum(grad, {0});
    grads["w"] = xt::linalg::dot(xt::transpose(inputs), grad);
    return xt::linalg::dot(grad, xt::transpose(params["w"]));
  }

  size_t input_size;
  size_t output_size;
};

template<typename F, typename FPrime>
struct Activation : Layer
{
  Tensor forward(const Tensor& i)
  {
    inputs = i;
    return f(inputs);
  }
  Tensor backward(const Tensor& grad)
  {
    return f_prime(inputs)*grad;
  }
      
  F f;
  FPrime f_prime;
};


struct tanh_f 
{
  Tensor operator()  (const Tensor& x)
  {
    return xt::tanh(x);
  }
}  ;

struct tanh_prime_f
{
  Tensor operator()  (const Tensor& x)
  {
    auto y = xt::tanh(x);
    return 1 - xt::square(y);
  }
}  ;

using Tanh = Activation<tanh_f,tanh_prime_f>;

#endif
