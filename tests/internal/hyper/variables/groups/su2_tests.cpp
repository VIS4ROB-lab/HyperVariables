/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/groups/su2.hpp"

namespace hyper::variables::tests {

using Scalar = double;

class QuaternionTests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kNumIterations = 10;
  static constexpr auto kNumericTolerance = 1e-7;

  // Definitions.
  using Quaternion = variables::Quaternion<Scalar>;

  static auto Random() -> Quaternion { return Quaternion{Eigen::internal::random<Scalar>(0.5, 1.5) * Quaternion::UnitRandom().coeffs()}; }

  [[nodiscard]] auto checkGroupExponentials() const -> bool {
    const auto qle = q_.groupLog().groupExp();
    const auto qel = q_.groupExp().groupLog();
    return qle.isApprox(q_, kNumericTolerance) && qel.isApprox(q_, kNumericTolerance);
  }

  Quaternion q_;
};

TEST_F(QuaternionTests, GroupExponentials) {
  for (auto i = 0; i < kNumIterations; ++i) {
    q_ = Random();
    EXPECT_TRUE(checkGroupExponentials());
  }
}

class SU2Tests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kNumIterations = 10;
  static constexpr auto kNumericIncrement = 1e-7;
  static constexpr auto kNumericTolerance = 1e-7;

  // Definitions.
  using SU2 = variables::SU2<Scalar>;
  using SU2Tangent = variables::Tangent<SU2>;
  using Vector = variables::Cartesian<Scalar, 3>;
  using SU2Jacobian = variables::JacobianNM<SU2Tangent>;
  using VectorJacobian = variables::JacobianNM<Vector>;
  using VectorSU2Jacobian = variables::JacobianNM<Vector, SU2Tangent>;

  [[nodiscard]] auto checkGroupInverseJacobian(const bool global) const -> bool {
    SU2Jacobian J_a, J_n;
    const auto i_q = su2_.groupInverse(J_a.data(), global);
    for (auto j = 0; j < SU2Tangent::kNumParameters; ++j) {
      J_n.col(j) = NumericGroupMinus(NumericGroupPlus(su2_, global, j).groupInverse(), i_q, global);
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupPlusJacobian(const bool global) const -> bool {
    const auto other_su2 = SU2::Random();
    const auto su2 = su2_ * other_su2;

    SU2Jacobian J_lhs_a, J_lhs_n, J_rhs_a, J_rhs_n;
    su2_.groupPlus(other_su2, J_lhs_a.data(), J_rhs_a.data(), global);
    for (auto j = 0; j < SU2Tangent::kNumParameters; ++j) {
      J_lhs_n.col(j) = NumericGroupMinus(NumericGroupPlus(su2_, global, j).groupPlus(other_su2), su2, global);
      J_rhs_n.col(j) = NumericGroupMinus(su2_.groupPlus(NumericGroupPlus(other_su2, global, j)), su2, global);
    }

    return J_lhs_n.isApprox(J_lhs_a, kNumericTolerance) && J_rhs_n.isApprox(J_rhs_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkVectorPlusJacobian(const bool global) const -> bool {
    const Vector input = Vector::Random();

    VectorSU2Jacobian J_a, J_n;
    VectorJacobian J_v_a, J_v_n;
    const auto output = su2_.vectorPlus(input);
    su2_.vectorPlus(input, J_a.data(), J_v_a.data(), global);
    for (auto j = 0; j < SU2Tangent::kNumParameters; ++j) {
      J_n.col(j) = (NumericGroupPlus(su2_, global, j).vectorPlus(input) - output) / kNumericIncrement;
    }
    for (auto j = 0; j < Vector::kNumParameters; ++j) {
      J_v_n.col(j) = (su2_.vectorPlus(input + kNumericIncrement * Vector::Unit(j)) - output) / kNumericIncrement;
    }

    return J_n.isApprox(J_a, kNumericTolerance) && J_v_n.isApprox(J_v_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupExponentials() const -> bool {
    const auto qle = su2_.groupLog().groupExp();
    const auto qel = su2_.groupExp().groupLog();
    return qle.isApprox(su2_, kNumericTolerance) && qel.isApprox(su2_, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupExponentialsJacobians(const bool global) const -> bool {
    SU2Jacobian J_l_a, J_e_a;
    const auto tangent = su2_.toTangent(J_l_a.data(), global);
    const auto su2 = tangent.toManifold(J_e_a.data(), global);

    SU2Jacobian J_l_n, J_e_n;
    for (auto j = 0; j < SU2Tangent::kNumParameters; ++j) {
      const auto d_tangent = SU2Tangent{tangent + kNumericIncrement * SU2Tangent::Unit(j)};
      J_l_n.col(j) = (NumericGroupPlus(su2_, global, j).toTangent() - tangent) / kNumericIncrement;
      J_e_n.col(j) = NumericGroupMinus(d_tangent.toManifold(), su2_, global);
    }

    return su2.isApprox(su2_, kNumericTolerance) && (J_l_a * J_e_a).isIdentity(kNumericTolerance) && J_l_n.isApprox(J_l_a, kNumericTolerance) &&
           J_e_n.isApprox(J_e_a, kNumericTolerance);
  }

  SU2 su2_;

 private:
  static auto NumericGroupPlus(const SU2& su2, const bool global, const Eigen::Index i) -> SU2 {
    const auto tau = SU2Tangent{kNumericIncrement * SU2Tangent::Unit(i)};
    return (global) ? tau.toManifold() * su2 : su2 * tau.toManifold();
  }

  static auto NumericGroupMinus(const SU2& d_su2, const SU2& su2, const bool global) -> SU2Tangent {
    return ((global) ? d_su2.groupPlus(su2.groupInverse()) : su2.groupInverse().groupPlus(d_su2)).toTangent() / kNumericIncrement;
  }
};

TEST_F(SU2Tests, GroupInverse) {
  for (auto i = 0; i < kNumIterations; ++i) {
    su2_ = SU2::Random();
    EXPECT_TRUE(checkGroupInverseJacobian(true));
    EXPECT_TRUE(checkGroupInverseJacobian(false));
  }
}

TEST_F(SU2Tests, GroupPlus) {
  for (auto i = 0; i < kNumIterations; ++i) {
    su2_ = SU2::Random();
    EXPECT_TRUE(checkGroupPlusJacobian(true));
    EXPECT_TRUE(checkGroupPlusJacobian(false));
  }
}

TEST_F(SU2Tests, VectorPlus) {
  for (auto i = 0; i < kNumIterations; ++i) {
    su2_ = SU2::Random();
    EXPECT_TRUE(checkVectorPlusJacobian(true));
    EXPECT_TRUE(checkVectorPlusJacobian(false));
  }
}

TEST_F(SU2Tests, GroupExponentials) {
  for (auto i = 0; i < kNumIterations; ++i) {
    su2_ = SU2::Random();
    EXPECT_TRUE(checkGroupExponentials());
    EXPECT_TRUE(checkGroupExponentialsJacobians(true));
    EXPECT_TRUE(checkGroupExponentialsJacobians(false));
  }
}

}  // namespace hyper::variables::tests
