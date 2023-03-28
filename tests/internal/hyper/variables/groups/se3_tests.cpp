/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/groups/se3.hpp"

namespace hyper::variables::tests {

class SE3Tests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kItr = 25;
  static constexpr auto kInc = 1e-6;
  static constexpr auto kTol = 1e-5;

  // Definitions.
  using Scalar = double;
  using SE3 = variables::SE3<Scalar>;
  using SE3Tangent = variables::Tangent<SE3>;
  using SE3Jacobian = variables::JacobianNM<SE3Tangent>;

  [[nodiscard]] auto checkGroupInverse() const -> bool {
    return (se3_.gInv().gInv()).isApprox(se3_, kTol) && (se3_.gInv().gPlus(se3_)).isApprox(SE3::Identity(), kTol) && (se3_.gPlus(se3_.gInv())).isApprox(SE3::Identity(), kTol);
  }

  [[nodiscard]] auto checkGroupInverseJacobian() const -> bool {
    SE3Jacobian J_a, J_n;
    const auto i_se3 = se3_.gInv(J_a.data());
    for (auto j = 0; j < SE3Tangent::kNumParameters; ++j) {
      const SE3Tangent inc = kInc * SE3Tangent::Unit(j);
      J_n.col(j) = se3_.tPlus(inc).gInv().tMinus(i_se3) / kInc;
    }

    return J_n.isApprox(J_a, kTol);
  }

  [[nodiscard]] auto checkGroupPlusJacobian() const -> bool {
    const auto other_se3 = SE3::Random();

    SE3Jacobian J_lhs_a, J_lhs_n, J_rhs_a, J_rhs_n;
    const auto se3 = se3_.gPlus(other_se3, J_lhs_a.data(), J_rhs_a.data());
    for (auto j = 0; j < SE3Tangent::kNumParameters; ++j) {
      const SE3Tangent inc = kInc * SE3Tangent::Unit(j);
      J_lhs_n.col(j) = se3_.tPlus(inc).gPlus(other_se3).tMinus(se3) / kInc;
      J_rhs_n.col(j) = se3_.gPlus(other_se3.tPlus(inc)).tMinus(se3) / kInc;
    }

    return J_lhs_n.isApprox(J_lhs_a, kTol) && J_rhs_n.isApprox(J_rhs_a, kTol);
  }

  [[nodiscard]] auto checkGroupActionJacobian() const -> bool {
    using Vector = SE3::Translation;
    const Vector input = Vector::Random();

    JacobianNM<Vector, SE3Tangent> J_a, J_n;
    JacobianNM<Vector> J_p_a, J_p_n;
    const auto output = se3_.act(input);
    se3_.act(input, J_a.data(), J_p_a.data());
    for (auto j = 0; j < SE3Tangent::kNumParameters; ++j) {
      const SE3Tangent inc = kInc * SE3Tangent::Unit(j);
      J_n.col(j) = (se3_.tPlus(inc).act(input) - output) / kInc;
    }

    for (auto j = 0; j < Vector::kNumParameters; ++j) {
      J_p_n.col(j) = (se3_.act(input + kInc * Vector::Unit(j)) - output) / kInc;
    }

    return J_n.isApprox(J_a, kTol) && J_p_n.isApprox(J_p_a, kTol);
  }

  [[nodiscard]] auto checkGroupExponentials() const -> bool {
    const auto se3 = se3_.gLog().gExp();
    return se3.isApprox(se3_, kTol);
  }

  [[nodiscard]] auto checkGroupExponentialsJacobians() const -> bool {
    SE3Jacobian J_l_a, J_e_a;
    const auto tangent = se3_.gLog(J_l_a.data());
    const auto se3 = tangent.gExp(J_e_a.data());

    SE3Jacobian J_l_n, J_e_n;
    for (auto j = 0; j < SE3Tangent::kNumParameters; ++j) {
      const SE3Tangent inc = SE3Tangent{kInc * SE3Tangent::Unit(j)};
      const SE3Tangent d_tangent = tangent + inc;
      J_l_n.col(j) = (se3_.tPlus(inc).gLog() - tangent) / kInc;
      J_e_n.col(j) = d_tangent.gExp().tMinus(se3_) / kInc;
    }

    return se3.isApprox(se3_, kTol) && (J_l_a * J_e_a).isIdentity(kTol) && J_l_n.isApprox(J_l_a, kTol) && J_e_n.isApprox(J_e_a, kTol);
  }

  SE3 se3_;
};

TEST_F(SE3Tests, GroupInverse) {
  for (auto i = 0; i < kItr; ++i) {
    se3_ = SE3::Random();
    EXPECT_TRUE(checkGroupInverse());
    EXPECT_TRUE(checkGroupInverseJacobian());
  }
}

TEST_F(SE3Tests, GroupPlus) {
  for (auto i = 0; i < kItr; ++i) {
    se3_ = SE3::Random();
    EXPECT_TRUE(checkGroupPlusJacobian());
  }
}

TEST_F(SE3Tests, GroupAction) {
  for (auto i = 0; i < kItr; ++i) {
    se3_ = SE3::Random();
    EXPECT_TRUE(checkGroupActionJacobian());
  }
}

TEST_F(SE3Tests, GroupExponentials) {
  for (auto i = 0; i < kItr; ++i) {
    se3_ = SE3::Random();
    EXPECT_TRUE(checkGroupExponentials());
    EXPECT_TRUE(checkGroupExponentialsJacobians());
  }
}

}  // namespace hyper::variables::tests
