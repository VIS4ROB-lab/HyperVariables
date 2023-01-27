/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/groups/se3.hpp"

namespace hyper::variables::tests {

class SE3Tests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kNumIterations = 25;
  static constexpr auto kNumericIncrement = 1e-7;
  static constexpr auto kNumericTolerance = 1e-7;

  // Definitions.
  using Scalar = double;
  using SE3 = variables::SE3<Scalar>;
  using SE3Tangent = variables::Tangent<SE3>;
  using SE3Jacobian = variables::JacobianNM<SE3Tangent>;

  [[nodiscard]] auto checkGroupInverse() const -> bool {
    return (se3_.gInv().gInv()).isApprox(se3_, kNumericTolerance) && (se3_.gInv().gPlus(se3_)).isApprox(SE3::Identity(), kNumericTolerance) &&
           (se3_.gPlus(se3_.gInv())).isApprox(SE3::Identity(), kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupInverseJacobian(const bool global, const bool coupled) const -> bool {
    SE3Jacobian J_a, J_n;
    const auto i_se3 = se3_.gInv(J_a.data(), global, coupled);
    for (auto j = 0; j < SE3Tangent::kNumParameters; ++j) {
      const SE3Tangent increment = kNumericIncrement * SE3Tangent::Unit(j);
      J_n.col(j) = se3_.tPlus(increment, global, coupled).gInv().tMinus(i_se3, global, coupled) / kNumericIncrement;
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupPlusJacobian(const bool global, const bool coupled) const -> bool {
    const auto other_se3 = SE3::Random();

    SE3Jacobian J_lhs_a, J_lhs_n, J_rhs_a, J_rhs_n;
    const auto se3 = se3_.gPlus(other_se3, J_lhs_a.data(), J_rhs_a.data(), global, coupled);
    for (auto j = 0; j < SE3Tangent::kNumParameters; ++j) {
      const SE3Tangent increment = kNumericIncrement * SE3Tangent::Unit(j);
      J_lhs_n.col(j) = se3_.tPlus(increment, global, coupled).gPlus(other_se3).tMinus(se3, global, coupled) / kNumericIncrement;
      J_rhs_n.col(j) = se3_.gPlus(other_se3.tPlus(increment, global, coupled)).tMinus(se3, global, coupled) / kNumericIncrement;
    }

    return J_lhs_n.isApprox(J_lhs_a, kNumericTolerance) && J_rhs_n.isApprox(J_rhs_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkVectorPlusJacobian(const bool coupled) const -> bool {
    using Vector = SE3::Translation;
    const Vector input = Vector::Random();

    JacobianNM<Vector, SE3Tangent> J_l_a, J_r_a, J_l_n, J_r_n;
    JacobianNM<Vector> J_l_p_a, J_r_p_a, J_p_n;
    const auto output = se3_.act(input);
    se3_.act(input, J_l_a.data(), J_l_p_a.data(), coupled, true);
    se3_.act(input, J_r_a.data(), J_r_p_a.data(), coupled, false);
    for (auto j = 0; j < SE3Tangent::kNumParameters; ++j) {
      const SE3Tangent increment = kNumericIncrement * SE3Tangent::Unit(j);
      J_l_n.col(j) = (se3_.tPlus(increment, coupled, true).act(input) - output) / kNumericIncrement;
      J_r_n.col(j) = (se3_.tPlus(increment, coupled, false).act(input) - output) / kNumericIncrement;
    }

    for (auto j = 0; j < Vector::kNumParameters; ++j) {
      J_p_n.col(j) = (se3_.act(input + kNumericIncrement * Vector::Unit(j)) - output) / kNumericIncrement;
    }

    return J_l_n.isApprox(J_l_a, kNumericTolerance) && J_r_n.isApprox(J_r_a, kNumericTolerance) && J_p_n.isApprox(J_l_p_a, kNumericTolerance) &&
           J_p_n.isApprox(J_r_p_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupExponentials() const -> bool {
    const auto se3 = se3_.toTangent().toManifold();
    return se3.isApprox(se3_, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupExponentialsJacobians(const bool global, const bool coupled) const -> bool {
    SE3Jacobian J_l_a, J_e_a;
    const auto tangent = se3_.toTangent(J_l_a.data(), global, coupled);
    const auto se3 = tangent.toManifold(J_e_a.data(), global, coupled);

    SE3Jacobian J_l_n, J_e_n;
    for (auto j = 0; j < SE3Tangent::kNumParameters; ++j) {
      const SE3Tangent increment = SE3Tangent{kNumericIncrement * SE3Tangent::Unit(j)};
      const SE3Tangent d_tangent = tangent + increment;
      J_l_n.col(j) = (se3_.tPlus(increment, global, coupled).toTangent() - tangent) / kNumericIncrement;
      J_e_n.col(j) = d_tangent.toManifold().tMinus(se3_, global, coupled) / kNumericIncrement;
    }

    return se3.isApprox(se3_, kNumericTolerance) && (J_l_a * J_e_a).isIdentity(kNumericTolerance) && J_l_n.isApprox(J_l_a, kNumericTolerance) &&
           J_e_n.isApprox(J_e_a, kNumericTolerance);
  }

  SE3 se3_;
};

TEST_F(SE3Tests, GroupInverse) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3::Random();
    EXPECT_TRUE(checkGroupInverse());
    EXPECT_TRUE(checkGroupInverseJacobian(false, true));
    EXPECT_TRUE(checkGroupInverseJacobian(false, false));
    EXPECT_TRUE(checkGroupInverseJacobian(true, true));
    EXPECT_TRUE(checkGroupInverseJacobian(true, false));
  }
}

TEST_F(SE3Tests, GroupPlus) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3::Random();
    EXPECT_TRUE(checkGroupPlusJacobian(false, true));
    EXPECT_TRUE(checkGroupPlusJacobian(false, false));
    EXPECT_TRUE(checkGroupPlusJacobian(true, true));
    EXPECT_TRUE(checkGroupPlusJacobian(true, false));
  }
}

TEST_F(SE3Tests, VectorPlus) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3::Random();
    EXPECT_TRUE(checkVectorPlusJacobian(false));
    EXPECT_TRUE(checkVectorPlusJacobian(true));
  }
}

TEST_F(SE3Tests, GroupExponentials) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3::Random();
    EXPECT_TRUE(checkGroupExponentials());
    EXPECT_TRUE(checkGroupExponentialsJacobians(false, true));
    EXPECT_TRUE(checkGroupExponentialsJacobians(false, false));
    EXPECT_TRUE(checkGroupExponentialsJacobians(true, true));
    EXPECT_TRUE(checkGroupExponentialsJacobians(true, false));
  }
}

}  // namespace hyper::variables::tests
