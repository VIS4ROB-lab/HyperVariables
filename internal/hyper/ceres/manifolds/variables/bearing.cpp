/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#ifdef HYPER_COMPILE_WITH_CERES

#include <ceres/sphere_manifold.h>

#include "hyper/ceres/manifolds/variables/bearing.hpp"
#include "hyper/ceres/manifolds/variables/euclidean.hpp"
#include "hyper/variables/bearing.hpp"

namespace hyper::ceres::manifolds {

auto Manifold<variables::Bearing<double>>::CreateManifold(const bool constant) -> std::unique_ptr<::ceres::Manifold> {
  if (constant) {
    return std::make_unique<Manifold<variables::Rn<Scalar, Bearing::kNumParameters>>>(true);
  } else {
    return std::make_unique<::ceres::SphereManifold<Bearing::kNumParameters>>();
  }
}

}  // namespace hyper::ceres::manifolds

#endif
