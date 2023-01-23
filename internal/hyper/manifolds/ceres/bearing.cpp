/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#ifdef HYPER_COMPILE_WITH_CERES

#include <ceres/sphere_manifold.h>

#include "hyper/manifolds/ceres/bearing.hpp"
#include "hyper/manifolds/ceres/euclidean.hpp"
#include "hyper/variables/bearing.hpp"

namespace hyper::manifolds::ceres {

auto Manifold<variables::Bearing<double>>::CreateManifold(const bool constant) -> std::unique_ptr<::ceres::Manifold> {
  if (constant) {
    return std::make_unique<Manifold<variables::Cartesian<Scalar, Bearing::kNumParameters>>>(true);
  } else {
    return std::make_unique<::ceres::SphereManifold<Bearing::kNumParameters>>();
  }
}

}  // namespace hyper::manifolds::ceres

#endif
