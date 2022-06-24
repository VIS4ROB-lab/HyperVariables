/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "variables/groups/se3.hpp"

namespace hyper::tests {

using Scalar = double;

class SE3Tests
    : public testing::Test {
 protected:
  static constexpr auto kNumIterations = 25;
  static constexpr auto kNumericIncrement = 1e-7;
  static constexpr auto kNumericTolerance = 1e-7;

  [[nodiscard]] auto checkGroupInverse() const -> bool {
    return (se3_.groupInverse().groupInverse()).isApprox(se3_, kNumericTolerance) &&
           (se3_.groupInverse().groupPlus(se3_)).isApprox(SE3<Scalar>::Identity(), kNumericTolerance) &&
           (se3_.groupPlus(se3_.groupInverse())).isApprox(SE3<Scalar>::Identity(), kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupInverseJacobian(const Frame frame) const -> bool {
    Jacobian<Tangent<SE3<Scalar>>> J_a, J_n;
    const auto i_se3 = se3_.groupInverse(J_a.data(), frame);
    for (auto j = 0; j < Traits<Tangent<SE3<Scalar>>>::kNumParameters; ++j) {
      J_n.col(j) = NumericGroupMinus(NumericGroupPlus(se3_, frame, j).groupInverse(), i_se3, frame);
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupPlusJacobian(const Frame frame) const -> bool {
    const auto other_se3 = SE3<Scalar>::Random();

    Jacobian<Tangent<SE3<Scalar>>> J_lhs_a, J_lhs_n, J_rhs_a, J_rhs_n;
    const auto se3 = se3_.groupPlus(other_se3, J_lhs_a.data(), J_rhs_a.data(), frame);
    for (auto j = 0; j < Traits<Tangent<SE3<Scalar>>>::kNumParameters; ++j) {
      J_lhs_n.col(j) = NumericGroupMinus(NumericGroupPlus(se3_, frame, j).groupPlus(other_se3), se3, frame);
      J_rhs_n.col(j) = NumericGroupMinus(se3_.groupPlus(NumericGroupPlus(other_se3, frame, j)), se3, frame);
    }

    return J_lhs_n.isApprox(J_lhs_a, kNumericTolerance) && J_rhs_n.isApprox(J_rhs_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkVectorPlusJacobian() const -> bool {
    using Vector = SE3<Scalar>::Translation;
    const Vector input = Vector::Random();

    Jacobian<Vector, Tangent<SE3<Scalar>>> J_l_a, J_r_a, J_l_n, J_r_n;
    Jacobian<Vector> J_l_p_a, J_r_p_a, J_p_n;
    const auto output = se3_.vectorPlus(input);
    se3_.vectorPlus(input, J_l_a.data(), J_l_p_a.data(), Frame::GLOBAL);
    se3_.vectorPlus(input, J_r_a.data(), J_r_p_a.data(), Frame::LOCAL);
    for (auto j = 0; j < Traits<Tangent<SE3<Scalar>>>::kNumParameters; ++j) {
      J_l_n.col(j) = (NumericGroupPlus(se3_, Frame::GLOBAL, j).vectorPlus(input) - output) / kNumericIncrement;
      J_r_n.col(j) = (NumericGroupPlus(se3_, Frame::LOCAL, j).vectorPlus(input) - output) / kNumericIncrement;
    }

    for (auto j = 0; j < Traits<Vector>::kNumParameters; ++j) {
      J_p_n.col(j) = (se3_.vectorPlus(input + kNumericIncrement * Vector::Unit(j)) - output) / kNumericIncrement;
    }

    return J_l_n.isApprox(J_l_a, kNumericTolerance) &&
           J_r_n.isApprox(J_r_a, kNumericTolerance) &&
           J_p_n.isApprox(J_l_p_a, kNumericTolerance) &&
           J_p_n.isApprox(J_r_p_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupExponentials() const -> bool {
    const auto se3 = se3_.toTangent().toManifold();
    return se3.isApprox(se3_, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupExponentialsJacobians(const Frame frame) const -> bool {
    using Tangent = Tangent<SE3<Scalar>>;

    Jacobian<Tangent> J_l_a, J_e_a;
    const auto tangent = se3_.toTangent(J_l_a.data(), frame);
    const auto se3 = tangent.toManifold(J_e_a.data(), frame);

    Jacobian<Tangent> J_l_n, J_e_n;
    for (auto j = 0; j < Traits<Tangent>::kNumParameters; ++j) {
      const auto d_tangent = Tangent{tangent + kNumericIncrement * Tangent::Unit(j)};
      J_l_n.col(j) = (NumericGroupPlus(se3_, frame, j).toTangent() - tangent) / kNumericIncrement;
      J_e_n.col(j) = NumericGroupMinus(d_tangent.toManifold(), se3_, frame);
    }

    return se3.isApprox(se3_, kNumericTolerance) &&
           (J_l_a * J_e_a).isIdentity(kNumericTolerance) &&
           J_l_n.isApprox(J_l_a, kNumericTolerance) &&
           J_e_n.isApprox(J_e_a, kNumericTolerance);
  }

  SE3<Scalar> se3_;

 private:
  static auto NumericGroupPlus(const Eigen::Ref<const SE3<Scalar>>& se3, const Frame frame, const Eigen::Index i) -> SE3<Scalar> {
    const auto delta = Tangent<SE3<Scalar>>{kNumericIncrement * Tangent<SE3<Scalar>>::Unit(i)};
    return (frame == Frame::GLOBAL) ? delta.toManifold().groupPlus(se3) : se3.groupPlus(delta.toManifold());
  }

  static auto NumericGroupMinus(const Eigen::Ref<const SE3<Scalar>>& d_se3, const Eigen::Ref<const SE3<Scalar>>& se3, const Frame frame) -> Tangent<SE3<Scalar>> {
    return ((frame == Frame::GLOBAL) ? d_se3.groupPlus(se3.groupInverse()) : se3.groupInverse().groupPlus(d_se3)).toTangent() / kNumericIncrement;
  }
};

TEST_F(SE3Tests, GroupInverse) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3<Scalar>::Random();
    EXPECT_TRUE(checkGroupInverse());
    EXPECT_TRUE(checkGroupInverseJacobian(Frame::GLOBAL));
    EXPECT_TRUE(checkGroupInverseJacobian(Frame::LOCAL));
  }
}

TEST_F(SE3Tests, GroupPlus) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3<Scalar>::Random();
    EXPECT_TRUE(checkGroupPlusJacobian(Frame::GLOBAL));
    EXPECT_TRUE(checkGroupPlusJacobian(Frame::LOCAL));
  }
}

TEST_F(SE3Tests, VectorPlus) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3<Scalar>::Random();
    EXPECT_TRUE(checkVectorPlusJacobian());
  }
}

TEST_F(SE3Tests, GroupExponentials) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3<Scalar>::Random();
    EXPECT_TRUE(checkGroupExponentials());
    EXPECT_TRUE(checkGroupExponentialsJacobians(Frame::GLOBAL));
    EXPECT_TRUE(checkGroupExponentialsJacobians(Frame::LOCAL));
  }
}

} // namespace hyper::tests
