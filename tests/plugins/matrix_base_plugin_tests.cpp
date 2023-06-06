/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "hyper/variables/rn.hpp"

namespace hyper::tests {

TEST(MatrixBasePluginTests, HatVeeDuality) {
  constexpr auto kTol = 1e-12;

  using R3 = variables::R3;
  const R3 u = R3::Random();
  const R3 v = R3::Random();
  const R3 uxv = u.cross(v);

  const auto ux = u.hat();
  const auto vx = v.hat();

  EXPECT_TRUE(ux.isApprox(Scalar{-1} * ux.transpose(), kTol));
  EXPECT_TRUE(ux.vee().isApprox(u, kTol));
  EXPECT_TRUE(uxv.isApprox(ux * v, kTol));
  EXPECT_TRUE(uxv.hat().isApprox(ux * vx - vx * ux, kTol));
}

}  // namespace hyper::tests
