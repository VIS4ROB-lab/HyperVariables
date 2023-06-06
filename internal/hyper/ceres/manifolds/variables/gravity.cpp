/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#if HYPER_COMPILE_WITH_CERES

#include "hyper/ceres/manifolds/variables/gravity.hpp"

namespace hyper::ceres::manifolds {

Manifold<variables::Gravity>::Manifold(const bool constant) : Manifold<Bearing>{constant} {}

}  // namespace hyper::ceres::manifolds

#endif
