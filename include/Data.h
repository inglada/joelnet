#ifndef DATA_H
#define DATA_H

#include "Tensor.h"
#include <algorithm>

struct Batch
{
  Tensor input_batch;
  Tensor output_batch;
};

struct DataIterator
{
  virtual Batch GetBatch() = 0;
  virtual void PrepareBatches(Tensor inputs, Tensor outputs) = 0;
  bool done = false;
  Tensor inputs;
  Tensor outputs;
};

struct BatchIterator : DataIterator
{
  BatchIterator(size_t batch_size = 32, 
                bool shuffle = true) : batch_size{batch_size}, shuffle{shuffle}
  {
  }
  void PrepareBatches(Tensor i, Tensor o) override
  {
    inputs = i;
    outputs = o;
    auto inputs_length = static_cast<int>(inputs.shape()[0]);
    if(inputs_length < batch_size) batch_size = inputs_length;
    assert(static_cast<int>(inputs.shape()[0]) >=
           static_cast<int>(batch_size));
    Tensor starts = xt::arange(0, inputs_length,
                                      static_cast<int>(batch_size));
    if(shuffle)
         {
         xt::random::shuffle(starts);
         }
    current_start = starts(0);
    done = false;
  }

  Batch GetBatch() override
  {
    Batch batch{};
    batch.input_batch = xt::view(inputs, xt::range(current_start, 
                                                   current_start+batch_size));
    batch.output_batch = xt::view(outputs, xt::range(current_start, 
                                                     current_start+batch_size));
    current_start += batch_size;
    if(current_start >= inputs.shape()[0]) done = true;
    return batch;
  }
  size_t batch_size;
  bool shuffle;
  size_t current_start = 0;
};
#endif
