#include <cstdlib>
#include "Tensor.h"
#include "Layers.h"
#include "Loss.h"
#include "NN.h"
#include "Optim.h"
#include "Data.h"
#include "Train.h"
#include "xtensor/xview.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xsort.hpp"

xt::xarray<int> fizz_buzz_encode(int x)
{
  if(x%15 == 0) return xt::xarray<int>{0, 0, 0, 1};
  if(x%5 == 0) return xt::xarray<int>{0, 0, 1, 0};
  if(x%3 == 0) return xt::xarray<int>{0, 1, 0, 0};
  return xt::xarray<int>{1, 0, 0, 0};
}


xt::xarray<int> binary_encode(int x)
{
  xt::xarray<int> result = xt::arange(0,10);
  for(auto& i : result)
    {
    i = (x >> i & 1);
    }
  return result;
    
}

int main()
{
  constexpr size_t nbSamples = 1024-101;
  std::vector<size_t> shape = {nbSamples, 10};
  xt::xarray<double> inputs = xt::zeros<int>(shape);
  for(size_t i=0; i<nbSamples; ++i)
    xt::view(inputs, xt::range(i,i+1), xt::all()) = binary_encode(101+i);

  shape = {nbSamples, 4};
  xt::xarray<double> targets = xt::zeros<int>(shape);
  for(size_t i=0; i<nbSamples; ++i)
    xt::view(targets, xt::range(i,i+1), xt::all()) = fizz_buzz_encode(101+i);

  Linear l1(10,50);
  Tanh t;
  Linear l2(50, 4);
  std::cout << "Net\n";
  NeuralNet net({&l1, &t, &l2});
  std::cout << "Train\n";
  train(net, inputs, targets, 5000,
           new BatchIterator{}, new MSE{}, 
           new SGD{0.001});

 int nbErrors{0};
 constexpr int nbTests{100};
 for(int x=1; x<=nbTests; ++x)
   {
   auto predicted = net.forward(binary_encode(x));
   auto predicted_idx = xt::argmax(predicted, 0)[0];
   auto actual_idx = xt::argmax(fizz_buzz_encode(x), 0)[0];
   std::vector<std::string> labels{std::to_string(x), "fizz", "buzz", "fizzbuzz"};
   std::cout << x << ' ' << labels[predicted_idx] << ' ' << labels[actual_idx] << '\n';
   if(labels[predicted_idx] != labels[actual_idx]) nbErrors++;
   }

 std::cout << "Accuracy = " << static_cast<double>(nbTests-nbErrors)/nbTests << '\n';


  return EXIT_SUCCESS;
}
