/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "variables/groups/su2.hpp"

namespace hyper::tests {

using Scalar = double;

class QuaternionTests
    : public testing::Test {
 protected:
  static constexpr auto kNumIterations = 10;
  static constexpr auto kNumericTolerance = 1e-7;

  static auto Random() -> Quaternion<Scalar> {
    return Quaternion<Scalar>{Eigen::internal::random<Scalar>(0.5, 1.5) * Quaternion<Scalar>::UnitRandom().coeffs()};
  }

  [[nodiscard]] auto checkGroupExponentials() const -> bool {
    const auto qle = q_.groupLog().groupExp();
    const auto qel = q_.groupExp().groupLog();
    return qle.isApprox(q_, kNumericTolerance) && qel.isApprox(q_, kNumericTolerance);
  }

  Quaternion<Scalar> q_;
};

TEST_F(QuaternionTests, GroupExponentials) {
  for (auto i = 0; i < kNumIterations; ++i) {
    q_ = Random();
    EXPECT_TRUE(checkGroupExponentials());
  }
}

class SU2Tests
    : public testing::Test {
 protected:
  static constexpr auto kNumIterations = 10;
  static constexpr auto kNumericIncrement = 1e-7;
  static constexpr auto kNumericTolerance = 1e-7;

  [[nodiscard]] auto checkGroupInverseJacobian(const Frame frame) const -> bool {
    Jacobian<Tangent<SU2<Scalar>>> J_a, J_n;
    const auto i_q = su2_.groupInverse(J_a.data(), frame);
    for (auto j = 0; j < Traits<Tangent<SU2<Scalar>>>::kNumParameters; ++j) {
      J_n.col(j) = NumericGroupMinus(NumericGroupPlus(su2_, frame, j).groupInverse(), i_q, frame);
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupPlusJacobian(const Frame frame) const -> bool {
    const auto other_su2 = SU2<Scalar>::Random();
    const auto su2 = su2_ * other_su2;

    Jacobian<Tangent<SU2<Scalar>>> J_lhs_a, J_lhs_n, J_rhs_a, J_rhs_n;
    su2_.groupPlus(other_su2, J_lhs_a.data(), J_rhs_a.data(), frame);
    for (auto j = 0; j < Traits<Tangent<SU2<Scalar>>>::kNumParameters; ++j) {
      J_lhs_n.col(j) = NumericGroupMinus(NumericGroupPlus(su2_, frame, j).groupPlus(other_su2), su2, frame);
      J_rhs_n.col(j) = NumericGroupMinus(su2_.groupPlus(NumericGroupPlus(other_su2, frame, j)), su2, frame);
    }

    return J_lhs_n.isApprox(J_lhs_a, kNumericTolerance) && J_rhs_n.isApprox(J_rhs_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkVectorPlusJacobian() const -> bool {
    using Vector = Cartesian<Scalar, 3>;
    const Vector input = Vector::Random();

    Jacobian<Vector, Tangent<SU2<Scalar>>> J_l_a, J_r_a, J_l_n, J_r_n;
    Jacobian<Vector> J_l_p_a, J_r_p_a, J_p_n;
    const auto output = su2_.vectorPlus(input);
    su2_.vectorPlus(input, J_l_a.data(), J_l_p_a.data(), Frame::GLOBAL);
    su2_.vectorPlus(input, J_r_a.data(), J_r_p_a.data(), Frame::LOCAL);
    for (auto j = 0; j < Traits<Tangent<SU2<Scalar>>>::kNumParameters; ++j) {
      J_l_n.col(j) = (NumericGroupPlus(su2_, Frame::GLOBAL, j).vectorPlus(input) - output) / kNumericIncrement;
      J_r_n.col(j) = (NumericGroupPlus(su2_, Frame::LOCAL, j).vectorPlus(input) - output) / kNumericIncrement;
    }

    for (auto j = 0; j < Traits<Vector>::kNumParameters; ++j) {
      J_p_n.col(j) = (su2_.vectorPlus(input + kNumericIncrement * Vector::Unit(j)) - output) / kNumericIncrement;
    }

    return J_l_n.isApprox(J_l_a, kNumericTolerance) &&
           J_r_n.isApprox(J_r_a, kNumericTolerance) &&
           J_p_n.isApprox(J_l_p_a, kNumericTolerance) &&
           J_p_n.isApprox(J_r_p_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupExponentials() const -> bool {
    const auto qle = su2_.groupLog().groupExp();
    const auto qel = su2_.groupExp().groupLog();
    return qle.isApprox(su2_, kNumericTolerance) && qel.isApprox(su2_, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupExponentialsJacobians(const Frame frame) const -> bool {
    using Tangent = Tangent<SU2<Scalar>>;

    Jacobian<Tangent> J_l_a, J_e_a;
    const auto tangent = su2_.toTangent(J_l_a.data(), frame);
    const auto su2 = tangent.toManifold(J_e_a.data(), frame);

    Jacobian<Tangent> J_l_n, J_e_n;
    for (auto j = 0; j < Traits<Tangent>::kNumParameters; ++j) {
      const auto d_tangent = Tangent{tangent + kNumericIncrement * Tangent::Unit(j)};
      J_l_n.col(j) = (NumericGroupPlus(su2_, frame, j).toTangent() - tangent) / kNumericIncrement;
      J_e_n.col(j) = NumericGroupMinus(d_tangent.toManifold(), su2_, frame);
    }

    return su2.isApprox(su2_, kNumericTolerance) &&
           (J_l_a * J_e_a).isIdentity(kNumericTolerance) &&
           J_l_n.isApprox(J_l_a, kNumericTolerance) &&
           J_e_n.isApprox(J_e_a, kNumericTolerance);
  }

  SU2<Scalar> su2_;

 private:
  static auto NumericGroupPlus(const SU2<Scalar>& su2, const Frame frame, const Eigen::Index i) -> SU2<Scalar> {
    const auto tau = Tangent<SU2<Scalar>>{kNumericIncrement * Tangent<SU2<Scalar>>::Unit(i)};
    return (frame == Frame::GLOBAL) ? tau.toManifold() * su2 : su2 * tau.toManifold();
  }

  static auto NumericGroupMinus(const SU2<Scalar>& d_su2, const SU2<Scalar>& su2, const Frame frame) -> Tangent<SU2<Scalar>> {
    return ((frame == Frame::GLOBAL) ? d_su2.groupPlus(su2.groupInverse()) : su2.groupInverse().groupPlus(d_su2)).toTangent() / kNumericIncrement;
  }
};

TEST_F(SU2Tests, GroupInverse) {
  for (auto i = 0; i < kNumIterations; ++i) {
    su2_ = SU2<Scalar>::Random();
    EXPECT_TRUE(checkGroupInverseJacobian(Frame::GLOBAL));
    EXPECT_TRUE(checkGroupInverseJacobian(Frame::LOCAL));
  }
}

TEST_F(SU2Tests, GroupPlus) {
  for (auto i = 0; i < kNumIterations; ++i) {
    su2_ = SU2<Scalar>::Random();
    EXPECT_TRUE(checkGroupPlusJacobian(Frame::GLOBAL));
    EXPECT_TRUE(checkGroupPlusJacobian(Frame::LOCAL));
  }
}

TEST_F(SU2Tests, VectorPlus) {
  for (auto i = 0; i < kNumIterations; ++i) {
    su2_ = SU2<Scalar>::Random();
    EXPECT_TRUE(checkVectorPlusJacobian());
  }
}

TEST_F(SU2Tests, GroupExponentials) {
  for (auto i = 0; i < kNumIterations; ++i) {
    su2_ = SU2<Scalar>::Random();
    EXPECT_TRUE(checkGroupExponentials());
    EXPECT_TRUE(checkGroupExponentialsJacobians(Frame::GLOBAL));
    EXPECT_TRUE(checkGroupExponentialsJacobians(Frame::LOCAL));
  }
}

} // namespace hyper::tests
