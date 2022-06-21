/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include <Eigen/Dense>

namespace hyper::tests {

TEST(MatrixBasePluginTests, HatVeeDuality) {
  constexpr auto kNumericTolerance = 1e-12;

  using Scalar = double;
  using Vector = Eigen::Matrix<Scalar, 3, 1>;

  const Vector u = Vector::Random();
  const Vector v = Vector::Random();
  const Vector uxv = u.cross(v);

  const auto ux = u.hat();
  const auto vx = v.hat();

  EXPECT_TRUE(ux.isApprox(Scalar{-1} * ux.transpose(), kNumericTolerance));
  EXPECT_TRUE(ux.vee().isApprox(u, kNumericTolerance));
  EXPECT_TRUE(uxv.isApprox(ux * v, kNumericTolerance));
  EXPECT_TRUE(uxv.hat().isApprox(ux * vx - vx * ux, kNumericTolerance));
}

} // namespace hyper::tests
