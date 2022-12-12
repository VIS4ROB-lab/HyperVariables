/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/metrics/angular.hpp"
#include "hyper/metrics/cartesian.hpp"
#include "hyper/metrics/manifold.hpp"

namespace hyper::tests {

using Scalar = double;

class MetricsTests
    : public testing::Test {
 protected:
  static constexpr auto kNumIterations = 5;
  static constexpr auto kNumericIncrement = 1e-8;
  static constexpr auto kNumericTolerance = 1e-7;

  [[nodiscard]] static auto CheckCartesianMetric() -> bool {
    using Input = Cartesian<Scalar, 3>;
    using Metric = TCartesianMetric<Scalar, 3>;
    using Output = Metric::Output;
    using Jacobian = JacobianNM<Output, Input>;

    Input u = Input::Random();
    Input v = Input::Random();

    Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
    const auto f = Metric::Distance(u, v, J_lhs_a.data(), J_rhs_a.data());
    for (auto i = Eigen::Index{0}; i < Input::kNumParameters; ++i) {
      J_lhs_n.col(i) = (Metric::Distance(u + kNumericIncrement * Input::Unit(i), v) - f) / kNumericIncrement;
      J_rhs_n.col(i) = (Metric::Distance(u, v + kNumericIncrement * Input::Unit(i)) - f) / kNumericIncrement;
    }

    return J_lhs_n.isApprox(J_lhs_a, kNumericTolerance) &&
           J_rhs_n.isApprox(J_rhs_a, kNumericTolerance);
  }

  [[nodiscard]] static auto CheckAngularMetric() -> bool {
    using Input = Cartesian<Scalar, 3>;
    using Metric = TAngularMetric<Scalar, 3>;
    using Output = Metric::Output;
    using Jacobian = JacobianNM<Output, Input>;

    Input u = Input::Random();
    Input v = Input::Random();

    Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
    const auto f = Metric::Distance(u, v, J_lhs_a.data(), J_rhs_a.data());
    for (auto i = Eigen::Index{0}; i < Input::kNumParameters; ++i) {
      J_lhs_n.col(i) = (Metric::Distance(u + kNumericIncrement * Input::Unit(i), v) - f) / kNumericIncrement;
      J_rhs_n.col(i) = (Metric::Distance(u, v + kNumericIncrement * Input::Unit(i)) - f) / kNumericIncrement;
    }

    return std::abs(f[0] - std::acos(u.dot(v) / (u.norm() * v.norm()))) <= kNumericTolerance &&
           J_lhs_n.isApprox(J_lhs_a, kNumericTolerance) &&
           J_rhs_n.isApprox(J_rhs_a, kNumericTolerance);
  }

  [[nodiscard]] static auto CheckManifoldMetric(const bool global, const bool coupled) -> bool {
    using Input = SE3<Scalar>;
    using Metric = TManifoldMetric<Scalar, ManifoldEnum::SE3>;
    using Output = Metric::Output;
    using Jacobian = JacobianNM<Output, Tangent<SE3<Scalar>>>;

    Input u = Input::Random();
    Input v = Input::Random();

    Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
    const auto f = Metric::Distance(u, v, J_lhs_a.data(), J_rhs_a.data(), global, coupled);
    for (auto i = Eigen::Index{0}; i < Tangent<SE3<Scalar>>::kNumParameters; ++i) {
      J_lhs_n.col(i) = (Metric::Distance(SE3NumericGroupPlus(u, global, coupled, i), v) - f) / kNumericIncrement;
      J_rhs_n.col(i) = (Metric::Distance(u, SE3NumericGroupPlus(v, global, coupled, i)) - f) / kNumericIncrement;
    }

    return J_lhs_n.isApprox(J_lhs_a, kNumericTolerance) &&
           J_rhs_n.isApprox(J_rhs_a, kNumericTolerance);
  }

 private:
  static auto SE3NumericGroupPlus(const Eigen::Ref<const SE3<Scalar>>& se3, const bool global, const bool coupled, const Eigen::Index i) -> SE3<Scalar> {
    const auto tau = Tangent<SE3<Scalar>>{kNumericIncrement * Tangent<SE3<Scalar>>::Unit(i)};

    if (coupled) {
      if (global) {
        return tau.toManifold().groupPlus(se3);
      } else {
        return se3.groupPlus(tau.toManifold());
      }
    } else {
      if (global) {
        return {tau.angular().toManifold().groupPlus(se3.rotation()), se3.translation() + tau.linear()};
      } else {
        return {se3.rotation().groupPlus(tau.angular().toManifold()), se3.translation() + tau.linear()};
      }
    }
  }
};

TEST_F(MetricsTests, Cartesian) {
  for (auto k = 0; k < kNumIterations; ++k) {
    EXPECT_TRUE(CheckCartesianMetric());
  }
}

TEST_F(MetricsTests, Angular) {
  for (auto k = 0; k < kNumIterations; ++k) {
    EXPECT_TRUE(CheckAngularMetric());
  }
}

TEST_F(MetricsTests, Manifold) {
  for (auto k = 0; k < kNumIterations; ++k) {
    EXPECT_TRUE(CheckManifoldMetric(false, true));
    EXPECT_TRUE(CheckManifoldMetric(false, false));
    EXPECT_TRUE(CheckManifoldMetric(true, true));
    EXPECT_TRUE(CheckManifoldMetric(true, false));
  }
}

} // namespace hyper::tests
