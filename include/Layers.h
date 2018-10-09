#ifndef LAYERS_H
#define LAYERS_H
#include "Tensor.h"
#include <map>

namespace joelnet
{
/**
Our neural nets will be made up of layers.
Each layer needs to pass its inputs forward
and propagate gradients backward. For example,
a neural net might look like

inputs -> Linear -> Tanh -> Linear -> output
*/
struct Layer
{
  //Produce the outputs corresponding to these inputs
  virtual Tensor forward(const Tensor& inputs) = 0;
  //Backpropagate this gradient through the layer
  virtual Tensor backward(const Tensor& grad) = 0;
  std::map<std::string, Tensor> params;
  std::map<std::string, Tensor> grads;
  Tensor inputs;
};

//computes output = inputs @ w + b
struct Linear : Layer
{
  Linear(size_t input_size, size_t output_size) : input_size{input_size},
                                                  output_size{output_size} 
  {
    params["w"] = xt::random::randn<double>({input_size, output_size});
    params["b"] = xt::random::randn<double>({output_size});
  }

  // outputs = inputs @ w + b
  Tensor forward(const Tensor& i)
  {
    inputs = i;
    return xt::linalg::dot(inputs, params["w"])+params["b"];
  }

  /**
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db = a.T @ f'(x)
        and dy/dc = f'(x)
  */
  Tensor backward(const Tensor& grad)
  {
    grads["b"] = xt::sum(grad, {0});
    grads["w"] = xt::linalg::dot(xt::transpose(inputs), grad);
    return xt::linalg::dot(grad, xt::transpose(params["w"]));
  }

  size_t input_size;
  size_t output_size;
};


/** 
    An activation layer just applies a function
    elementwise to its inputs
*/
template<typename F, typename FPrime>
struct Activation : Layer
{
  Tensor forward(const Tensor& i)
  {
    inputs = i;
    return f(inputs);
  }

  /**
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
  */
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
}
#endif
