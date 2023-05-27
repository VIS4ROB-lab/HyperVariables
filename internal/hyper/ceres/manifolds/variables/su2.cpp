/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#if HYPER_COMPILE_WITH_CERES

#include "hyper/ceres/manifolds/variables/su2.hpp"
#include "hyper/ceres/manifolds/variables/euclidean.hpp"
#include "hyper/variables/su2.hpp"

namespace hyper::ceres::manifolds {

namespace internal {

template <typename TGroup>
class GroupManifoldImpl final : public ::ceres::Manifold {
 public:
  // Definitions.
  using Scalar = double;
  using Group = variables::SU2<Scalar>;
  using Tangent = variables::Tangent<Group>;

  // See Ceres documentation.
  [[nodiscard]] auto AmbientSize() const -> int final { return Group::kNumParameters; }

  // See Ceres documentation.
  [[nodiscard]] auto TangentSize() const -> int final { return Tangent::kNumParameters; }

  // See Ceres documentation.
  auto Plus(const Scalar* x, const Scalar* delta, Scalar* x_plus_delta) const -> bool final {
    const auto x_ = Eigen::Map<const Group>{x};
    const auto delta_ = Eigen::Map<const Tangent>{delta};
    Eigen::Map<Group>{x_plus_delta} = x_.tPlus(delta_);
    return true;
  }

  // See Ceres documentation.
  auto PlusJacobian(const Scalar* x, Scalar* jacobian) const -> bool final {
    const auto x_ = Eigen::Map<const Group>{x};
    Eigen::Map<Eigen::Matrix<Scalar, Group::kNumParameters, Tangent::kNumParameters, Eigen::RowMajor>>{jacobian} = x_.tPlusJacobian();
    return true;
  }

  // See Ceres documentation.
  auto Minus(const Scalar* y, const Scalar* x, Scalar* y_minus_x) const -> bool final {
    const auto y_ = Eigen::Map<const Group>{y};
    const auto x_ = Eigen::Map<const Group>{x};
    Eigen::Map<Tangent>{y_minus_x} = y_.tMinus(x_);
    return true;
  }

  // See Ceres documentation.
  auto MinusJacobian(const Scalar* x, Scalar* jacobian) const -> bool final {
    const auto x_ = Eigen::Map<const Group>{x};
    Eigen::Map<Eigen::Matrix<Scalar, Tangent::kNumParameters, Group::kNumParameters, Eigen::RowMajor>>{jacobian} = x_.tMinusJacobian();
    return true;
  }
};

}  // namespace internal

auto Manifold<variables::SU2<double>>::CreateManifold(const bool constant) -> std::unique_ptr<::ceres::Manifold> {
  if (constant) {
    return std::make_unique<Manifold<variables::Rn<Scalar, SU2::kNumParameters>>>(true);
  } else {
    return std::make_unique<internal::GroupManifoldImpl<SU2>>();
  }
}

}  // namespace hyper::ceres::manifolds

#endif
