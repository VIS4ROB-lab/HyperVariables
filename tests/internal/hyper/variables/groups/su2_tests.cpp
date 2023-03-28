/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/groups/su2.hpp"

namespace hyper::variables::tests {

class QuaternionTests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kItr = 10;
  static constexpr auto kTol = 1e-7;

  // Definitions.
  using Scalar = double;
  using Quaternion = variables::Quaternion<Scalar>;

  static auto Random() -> Quaternion { return Quaternion{Eigen::internal::random<Scalar>(0.5, 1.5) * Quaternion::UnitRandom().coeffs()}; }

  [[nodiscard]] auto checkGroupExponentials() const -> bool {
    const auto qle = q_.glog().gexp();
    const auto qel = q_.gexp().glog();
    return qle.isApprox(q_, kTol) && qel.isApprox(q_, kTol);
  }

  Quaternion q_;
};

TEST_F(QuaternionTests, GroupExponentials) {
  for (auto i = 0; i < kItr; ++i) {
    q_ = Random();
    EXPECT_TRUE(checkGroupExponentials());
  }
}

class SU2Tests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kItr = 10;
  static constexpr auto kInc = 1e-6;
  static constexpr auto kTol = 1e-5;

  // Definitions.
  using Scalar = double;
  using SU2 = variables::SU2<Scalar>;
  using SU2Tangent = variables::Tangent<SU2>;
  using Vector = variables::Cartesian<Scalar, 3>;
  using SU2Jacobian = variables::JacobianNM<SU2Tangent>;
  using VectorJacobian = variables::JacobianNM<Vector>;
  using VectorSU2Jacobian = variables::JacobianNM<Vector, SU2Tangent>;

  [[nodiscard]] auto checkGroupInverseJacobian() const -> bool {
    SU2Jacobian J_a, J_n;
    const auto i_q = su2_.gInv(J_a.data());
    for (auto j = 0; j < SU2Tangent::kNumParameters; ++j) {
      const SU2Tangent inc = kInc * SU2Tangent::Unit(j);
      J_n.col(j) = su2_.tPlus(inc).gInv().tMinus(i_q) / kInc;
    }

    return J_n.isApprox(J_a, kTol);
  }

  [[nodiscard]] auto checkGroupPlusJacobian() const -> bool {
    const auto other_su2 = SU2::Random();
    const auto su2 = su2_.gPlus(other_su2);

    SU2Jacobian J_lhs_a, J_lhs_n, J_rhs_a, J_rhs_n;
    su2_.gPlus(other_su2, J_lhs_a.data(), J_rhs_a.data());
    for (auto j = 0; j < SU2Tangent::kNumParameters; ++j) {
      const SU2Tangent inc = kInc * SU2Tangent::Unit(j);
      J_lhs_n.col(j) = su2_.tPlus(inc).gPlus(other_su2).tMinus(su2) / kInc;
      J_rhs_n.col(j) = su2_.gPlus(other_su2.tPlus(inc)).tMinus(su2) / kInc;
    }

    return J_lhs_n.isApprox(J_lhs_a, kTol) && J_rhs_n.isApprox(J_rhs_a, kTol);
  }

  [[nodiscard]] auto checkGroupActionJacobian() const -> bool {
    const Vector input = Vector::Random();

    VectorSU2Jacobian J_a, J_n;
    VectorJacobian J_v_a, J_v_n;
    const auto output = su2_.act(input);
    su2_.act(input, J_a.data(), J_v_a.data());
    for (auto j = 0; j < SU2Tangent::kNumParameters; ++j) {
      const SU2Tangent inc = kInc * SU2Tangent::Unit(j);
      J_n.col(j) = (su2_.tPlus(inc).act(input) - output) / kInc;
    }
    for (auto j = 0; j < Vector::kNumParameters; ++j) {
      J_v_n.col(j) = (su2_.act(input + kInc * Vector::Unit(j)) - output) / kInc;
    }

    return J_n.isApprox(J_a, kTol) && J_v_n.isApprox(J_v_a, kTol);
  }

  [[nodiscard]] auto checkGroupExponentials() const -> bool {
    const auto qle = su2_.gLog().gExp();
    const auto qel = su2_.gexp().glog();
    return qle.isApprox(su2_, kTol) && qel.isApprox(su2_, kTol);
  }

  [[nodiscard]] auto checkGroupExponentialsJacobians() const -> bool {
    SU2Jacobian J_l_a, J_e_a;
    const auto tangent = su2_.gLog(J_l_a.data());
    const auto su2 = tangent.gExp(J_e_a.data());

    SU2Jacobian J_l_n, J_e_n;
    for (auto j = 0; j < SU2Tangent::kNumParameters; ++j) {
      const SU2Tangent inc = kInc * SU2Tangent::Unit(j);
      const SU2Tangent d_tangent = tangent + inc;
      J_l_n.col(j) = (su2_.tPlus(inc).gLog() - tangent) / kInc;
      J_e_n.col(j) = d_tangent.gExp().tMinus(su2_) / kInc;
    }

    return su2.isApprox(su2_, kTol) && (J_l_a * J_e_a).isIdentity(kTol) && J_l_n.isApprox(J_l_a, kTol) && J_e_n.isApprox(J_e_a, kTol);
  }

  SU2 su2_;
};

TEST_F(SU2Tests, GroupInverse) {
  for (auto i = 0; i < kItr; ++i) {
    su2_ = SU2::Random();
    EXPECT_TRUE(checkGroupInverseJacobian());
  }
}

TEST_F(SU2Tests, GroupPlus) {
  for (auto i = 0; i < kItr; ++i) {
    su2_ = SU2::Random();
    EXPECT_TRUE(checkGroupPlusJacobian());
  }
}

TEST_F(SU2Tests, GroupAction) {
  for (auto i = 0; i < kItr; ++i) {
    su2_ = SU2::Random();
    EXPECT_TRUE(checkGroupActionJacobian());
  }
}

TEST_F(SU2Tests, GroupExponentials) {
  for (auto i = 0; i < kItr; ++i) {
    su2_ = SU2::Random();
    EXPECT_TRUE(checkGroupExponentials());
    EXPECT_TRUE(checkGroupExponentialsJacobians());
  }
}

}  // namespace hyper::variables::tests
