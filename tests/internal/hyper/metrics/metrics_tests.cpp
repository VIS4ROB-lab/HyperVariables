/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/metrics/metrics.hpp"

namespace hyper::metrics::tests {

static constexpr auto kItr = 5;
static constexpr auto kInc = 1e-8;
static constexpr auto kTol = 1e-7;

template <typename TMetric>
auto CheckMetric() -> void {
  using Input = TMetric::Input;
  using Jacobian = TMetric::Jacobian;
  using Tangent = variables::Tangent<Input>;

  Input u = Input::Random();
  Input v = Input::Random();

  Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
  const auto fx = TMetric::Distance(u, v, J_lhs_a.data(), J_rhs_a.data());
  for (int i = 0; i < Tangent::kNumParameters; ++i) {
    const Tangent inc = kInc * Tangent::Unit(i);
    J_lhs_n.col(i) = (TMetric::Distance(u.tPlus(inc), v) - fx) / kInc;
    J_rhs_n.col(i) = (TMetric::Distance(u, v.tPlus(inc)) - fx) / kInc;
  }

  // Angular distance: std::abs(fx[0] - std::acos(u.dot(v) / (u.norm() * v.norm())));
  EXPECT_TRUE(J_lhs_n.isApprox(J_lhs_a, kTol));
  EXPECT_TRUE(J_rhs_n.isApprox(J_rhs_a, kTol));
}

template <>
auto CheckMetric<metrics::ManifoldMetric<variables::SE3<double>>>() -> void {
  using Metric = metrics::ManifoldMetric<variables::SE3<double>>;
  using Input = Metric::Input;
  using Jacobian = Metric::Jacobian;
  using Tangent = variables::Tangent<Input>;

  Input u = Input::Random();
  Input v = Input::Random();

  Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
  const auto fx = Metric::Distance(u, v, J_lhs_a.data(), J_rhs_a.data());
  for (int i = 0; i < Tangent::kNumParameters; ++i) {
    const Tangent inc = kInc * Tangent::Unit(i);
    J_lhs_n.col(i) = (Metric::Distance(u.tPlus(inc), v) - fx) / kInc;
    J_rhs_n.col(i) = (Metric::Distance(u, v.tPlus(inc)) - fx) / kInc;
  }

  EXPECT_TRUE(J_lhs_n.isApprox(J_lhs_a, kTol));
  EXPECT_TRUE(J_rhs_n.isApprox(J_rhs_a, kTol));
}

TEST(MetricsTests, Cartesian) {
  for (auto k = 0; k < kItr; ++k) {
    CheckMetric<metrics::CartesianMetric<double, 3>>();
  }
}

TEST(MetricsTests, Angular) {
  for (auto k = 0; k < kItr; ++k) {
    CheckMetric<metrics::AngularMetric<double, 3>>();
  }
}

TEST(MetricsTests, Manifold) {
  for (auto k = 0; k < kItr; ++k) {
    CheckMetric<metrics::ManifoldMetric<variables::SE3<double>>>();
  }
}

}  // namespace hyper::metrics::tests
