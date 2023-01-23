/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#ifdef HYPER_COMPILE_WITH_CERES

#include "hyper/ceres/manifolds/variables/groups/su2.hpp"
#include "hyper/ceres/manifolds/variables/euclidean.hpp"
#include "hyper/variables/groups/su2.hpp"

namespace hyper::ceres::manifolds {

auto Manifold<variables::SU2<double>>::CreateManifold(const bool constant) -> std::unique_ptr<::ceres::Manifold> {
  if (constant) {
    return std::make_unique<Manifold<variables::Cartesian<Scalar, SU2::kNumParameters>>>(true);
  } else {
    return std::make_unique<::ceres::EigenQuaternionManifold>();
  }
}

}  // namespace hyper::ceres::manifolds

#endif
