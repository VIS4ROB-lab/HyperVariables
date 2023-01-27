/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/metrics/metrics.hpp"

namespace hyper::metrics::tests {

class MetricsTests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kItr = 5;
  static constexpr auto kInc = 1e-8;
  static constexpr auto kTol = 1e-7;

  // Definitions.
  using Scalar = double;
  using Index = Eigen::Index;
  using SE3 = variables::SE3<Scalar>;
  using SE3Tangent = variables::Tangent<SE3>;

  [[nodiscard]] static auto CheckCartesianMetric() -> bool {
    using Metric = metrics::CartesianMetric<Scalar, 3>;
    using Input = Metric::Input;
    // using Output = Metric::Output;
    using Jacobian = Metric::Jacobian;

    Input u = Input::Random();
    Input v = Input::Random();

    Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
    const auto f = Metric::Distance(u, v, J_lhs_a.data(), J_rhs_a.data());
    for (auto i = Eigen::Index{0}; i < Input::kNumParameters; ++i) {
      J_lhs_n.col(i) = (Metric::Distance(u + kInc * Input::Unit(i), v) - f) / kInc;
      J_rhs_n.col(i) = (Metric::Distance(u, v + kInc * Input::Unit(i)) - f) / kInc;
    }

    return J_lhs_n.isApprox(J_lhs_a, kTol) && J_rhs_n.isApprox(J_rhs_a, kTol);
  }

  [[nodiscard]] static auto CheckAngularMetric() -> bool {
    using Metric = metrics::AngularMetric<Scalar, 3>;
    using Input = Metric::Input;
    // using Output = Metric::Output;
    using Jacobian = Metric::Jacobian;

    Input u = Input::Random();
    Input v = Input::Random();

    Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
    const auto f = Metric::Distance(u, v, J_lhs_a.data(), J_rhs_a.data());
    for (auto i = Eigen::Index{0}; i < Input::kNumParameters; ++i) {
      J_lhs_n.col(i) = (Metric::Distance(u + kInc * Input::Unit(i), v) - f) / kInc;
      J_rhs_n.col(i) = (Metric::Distance(u, v + kInc * Input::Unit(i)) - f) / kInc;
    }

    return std::abs(f[0] - std::acos(u.dot(v) / (u.norm() * v.norm()))) <= kTol && J_lhs_n.isApprox(J_lhs_a, kTol) && J_rhs_n.isApprox(J_rhs_a, kTol);
  }

  [[nodiscard]] static auto CheckManifoldMetric(const bool global, const bool coupled) -> bool {
    using Metric = metrics::ManifoldMetric<SE3>;
    using Input = Metric::Input;
    using Output = Metric::Output;
    using Jacobian = Metric::Jacobian;

    Input u = Input::Random();
    Input v = Input::Random();

    Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
    const auto f = Metric::Distance(u, v, J_lhs_a.data(), J_rhs_a.data(), global, coupled);
    for (auto i = Eigen::Index{0}; i < Output::kNumParameters; ++i) {
      J_lhs_n.col(i) = (Metric::Distance(SE3NumericGroupPlus(u, global, coupled, i), v) - f) / kInc;
      J_rhs_n.col(i) = (Metric::Distance(u, SE3NumericGroupPlus(v, global, coupled, i)) - f) / kInc;
    }

    return J_lhs_n.isApprox(J_lhs_a, kTol) && J_rhs_n.isApprox(J_rhs_a, kTol);
  }

 private:
  static auto SE3NumericGroupPlus(const Eigen::Ref<const SE3>& se3, const bool global, const bool coupled, const Index i) -> SE3 {
    const auto tau = SE3Tangent{kInc * SE3Tangent::Unit(i)};

    if (coupled) {
      if (global) {
        return tau.gExp().gPlus(se3);
      } else {
        return se3.gPlus(tau.gExp());
      }
    } else {
      if (global) {
        return {tau.angular().gExp().gPlus(se3.rotation()), se3.translation() + tau.linear()};
      } else {
        return {se3.rotation().gPlus(tau.angular().gExp()), se3.translation() + tau.linear()};
      }
    }
  }
};

TEST_F(MetricsTests, Cartesian) {
  for (auto k = 0; k < kItr; ++k) {
    EXPECT_TRUE(CheckCartesianMetric());
  }
}

TEST_F(MetricsTests, Angular) {
  for (auto k = 0; k < kItr; ++k) {
    EXPECT_TRUE(CheckAngularMetric());
  }
}

TEST_F(MetricsTests, Manifold) {
  for (auto k = 0; k < kItr; ++k) {
    EXPECT_TRUE(CheckManifoldMetric(false, true));
    EXPECT_TRUE(CheckManifoldMetric(false, false));
    EXPECT_TRUE(CheckManifoldMetric(true, true));
    EXPECT_TRUE(CheckManifoldMetric(true, false));
  }
}

}  // namespace hyper::metrics::tests
