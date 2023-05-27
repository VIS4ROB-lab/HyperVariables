/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#if HYPER_COMPILE_WITH_CERES

#include <ceres/product_manifold.h>

#include "hyper/ceres/manifolds/variables/euclidean.hpp"
#include "hyper/ceres/manifolds/variables/se3.hpp"
#include "hyper/variables/se3.hpp"

namespace hyper::ceres::manifolds {

auto Manifold<variables::SE3<double>>::CreateManifold(const bool rotation_constant, const bool translation_constant) -> std::unique_ptr<::ceres::Manifold> {
  using RotationManifold = Manifold<SE3::Rotation>;
  using TranslationManifold = Manifold<SE3::Translation>;
  using ProductManifold = ::ceres::ProductManifold<RotationManifold, TranslationManifold>;
  return std::make_unique<ProductManifold>(RotationManifold{rotation_constant}, TranslationManifold{translation_constant});
}

}  // namespace hyper::ceres::manifolds

#endif
