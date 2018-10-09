#ifndef TENSOR_H
#define TENSOR_H
#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xbuilder.hpp"
#include "xtensor/xsort.hpp"

//A tensor is just a n-dimensional array
using Tensor = xt::xarray<double>;
#endif
