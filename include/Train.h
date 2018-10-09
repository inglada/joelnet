#ifndef TRAIN_H
#define TRAIN_H

#include "Tensor.h"
#include "NN.h"
#include "Loss.h"
#include "Optim.h"
#include "Data.h"

namespace joelnet
{
//Here's a function that can train a neural net
void train(NeuralNet& net, Tensor& inputs, Tensor& targets, int num_epochs = 5000,
           DataIterator* iterator = new BatchIterator{}, Loss* loss = new MSE{}, 
           Optimizer* optimizer = new SGD{})
{
  for(int epoch=0; epoch< num_epochs; ++epoch)
    {
    auto epoch_loss = 0.0;

    iterator->PrepareBatches(inputs, targets);
    while(!iterator->done)
      {
      auto batch = iterator->GetBatch();
      auto predicted = net.forward(batch.input_batch);
      epoch_loss += loss->loss(predicted, batch.output_batch);
      auto grad = loss->grad(predicted, batch.output_batch);
      net.backward(grad);
      optimizer->step(net);
      }
    std::cout << epoch << '\t' << epoch_loss << '\n';
    }
}
}
#endif
