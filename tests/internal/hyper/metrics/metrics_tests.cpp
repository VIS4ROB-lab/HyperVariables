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
  using InputTangent = variables::Tangent<Input>;
  using Jacobian = TMetric::Jacobian;

  Input u = Input::Random();
  Input v = Input::Random();

  Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
  const auto fx = TMetric::Evaluate(u, v, J_lhs_a.data(), J_rhs_a.data());
  for (int i = 0; i < InputTangent::kNumParameters; ++i) {
    const InputTangent dx = kInc * InputTangent::Unit(i);
    J_lhs_n.col(i) = (TMetric::Evaluate(u.tPlus(dx), v) - fx) / kInc;
    J_rhs_n.col(i) = (TMetric::Evaluate(u, v.tPlus(dx)) - fx) / kInc;
  }

  EXPECT_TRUE(J_lhs_n.isApprox(J_lhs_a, kTol));
  EXPECT_TRUE(J_rhs_n.isApprox(J_rhs_a, kTol));
}

TEST(MetricsTests, Angular) {
  for (auto k = 0; k < kItr; ++k) {
    CheckMetric<AngularMetric<variables::R3>>();
  }
}

TEST(MetricsTests, Euclidean) {
  for (auto k = 0; k < kItr; ++k) {
    CheckMetric<EuclideanMetric<variables::R3>>();
  }
}

TEST(MetricsTests, SU2Manifold) {
  for (auto k = 0; k < kItr; ++k) {
    CheckMetric<GroupMetric<variables::SU2>>();
  }
}

TEST(MetricsTests, SE3Manifold) {
  for (auto k = 0; k < kItr; ++k) {
    CheckMetric<GroupMetric<variables::SE3>>();
  }
}

}  // namespace hyper::metrics::tests
