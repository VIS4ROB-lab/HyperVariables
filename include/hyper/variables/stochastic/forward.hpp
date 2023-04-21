/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/forward.hpp"

namespace hyper::variables {

enum class GaussianType { STANDARD, CANONICAL };

template <typename TScalar, int TOrder>
class Uncertainty;

template <typename, int, GaussianType>
class Gaussian;

template <typename, int>
class DualGaussian;

}  // namespace hyper::variables
