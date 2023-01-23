/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#ifdef HYPER_COMPILE_WITH_CERES

#include "hyper/manifolds/ceres/gravity.hpp"

namespace hyper::manifolds::ceres {

Manifold<variables::Gravity<double>>::Manifold(const bool constant) : Manifold<Bearing>{constant} {}

}  // namespace hyper::manifolds::ceres

#endif
