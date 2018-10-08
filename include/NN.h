#ifndef NN_H
#define NN_H

#include "Tensor.h"
#include "Layers.h"
#include <vector>

struct NeuralNet
{
  NeuralNet(std::initializer_list<Layer*>&& layers) : layers{layers} {};
  Tensor forward(const Tensor& inputs)
  {
    Tensor res = inputs;
    for(auto layer : layers)
      {
      res = xt::eval(layer->forward(res));
      }
    return res;
  }

  Tensor backward(const Tensor& grad)
  {
    Tensor res = grad;
    auto iter = layers.rbegin();
    auto end = layers.rend();
    while(iter != end)
      {
      res = xt::eval((*iter)->backward(res));
      ++iter;
      }

    return res;
  }

  std::vector<std::pair<Tensor*, Tensor*>> params_and_grads()
  {
    std::vector<std::pair<Tensor*, Tensor*>> res;
    for(auto layer : layers)
      for(auto& param : layer->params)
        {
        Tensor* grad = &(layer->grads[param.first]);
        res.push_back({&(param.second), grad});
        }
    return res;
  }
  std::vector<Layer*> layers;

};

#endif
