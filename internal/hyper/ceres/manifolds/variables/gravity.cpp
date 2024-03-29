/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#ifdef HYPER_COMPILE_WITH_CERES

#include "hyper/ceres/manifolds/variables/gravity.hpp"

namespace hyper::ceres::manifolds {

Manifold<variables::Gravity<double>>::Manifold(const bool constant) : Manifold<Bearing>{constant} {}

}  // namespace hyper::ceres::manifolds

#endif
