/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include <glog/logging.h>

#include "metrics/angular.hpp"
#include "metrics/cartesian.hpp"
#include "metrics/manifold.hpp"

namespace hyper::tests {

constexpr auto kNumIterations = 5;
constexpr auto kNumericIncrement = 1e-8;
constexpr auto kNumericTolerance = 1e-7;

using Scalar = double;

TEST(MetricTests, Cartesian) {
  using Input = Cartesian<Scalar, 3>;
  using Metric = CartesianMetric<Input>;
  using Output = Metric::Output;
  using Jacobian = Jacobian<Output, Input>;

  for (auto k = 0; k < kNumIterations; ++k) {
    Input u = Input::Random();
    Input v = Input::Random();

    Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
    const auto f = Metric::Distance(u, v, J_lhs_a.data(), J_rhs_a.data());
    for (auto i = Eigen::Index{0}; i < Traits<Input>::kNumParameters; ++i) {
      J_lhs_n.col(i) = (Metric::Distance(u + kNumericIncrement * Input::Unit(i), v) - f) / kNumericIncrement;
      J_rhs_n.col(i) = (Metric::Distance(u, v + kNumericIncrement * Input::Unit(i)) - f) / kNumericIncrement;
    }

    EXPECT_TRUE(J_lhs_n.isApprox(J_lhs_a, kNumericTolerance));
    EXPECT_TRUE(J_rhs_n.isApprox(J_rhs_a, kNumericTolerance));
  }
}

TEST(MetricTests, Angular) {
  using Input = Cartesian<Scalar, 3>;
  using Metric = AngularMetric<Input>;
  using Output = Metric::Output;
  using Jacobian = Jacobian<Output, Input>;

  for (auto k = 0; k < kNumIterations; ++k) {
    Input u = Input::Random();
    Input v = Input::Random();

    Jacobian J_lhs_a, J_rhs_a, J_lhs_n, J_rhs_n;
    const auto f = Metric::Distance(u, v, J_lhs_a.data(), J_rhs_a.data());
    for (auto i = Eigen::Index{0}; i < Traits<Input>::kNumParameters; ++i) {
      J_lhs_n.col(i) = (Metric::Distance(u + kNumericIncrement * Input::Unit(i), v) - f) / kNumericIncrement;
      J_rhs_n.col(i) = (Metric::Distance(u, v + kNumericIncrement * Input::Unit(i)) - f) / kNumericIncrement;
    }

    EXPECT_TRUE(J_lhs_n.isApprox(J_lhs_a, kNumericTolerance));
    EXPECT_TRUE(J_rhs_n.isApprox(J_rhs_a, kNumericTolerance));
  }
}

} // namespace hyper::tests
