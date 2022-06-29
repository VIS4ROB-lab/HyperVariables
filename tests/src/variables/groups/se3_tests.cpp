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

  [[nodiscard]] auto checkGroupInverseJacobian(const bool global, const bool coupled) const -> bool {
    Jacobian<Tangent<SE3<Scalar>>> J_a, J_n;
    const auto i_se3 = se3_.groupInverse(J_a.data(), coupled, global);
    for (auto j = 0; j < Traits<Tangent<SE3<Scalar>>>::kNumParameters; ++j) {
      J_n.col(j) = NumericGroupMinus(NumericGroupPlus(se3_, coupled, global, j).groupInverse(), i_se3, coupled, global);
    }

    return J_n.isApprox(J_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkGroupPlusJacobian(const bool global, const bool coupled) const -> bool {
    const auto other_se3 = SE3<Scalar>::Random();

    Jacobian<Tangent<SE3<Scalar>>> J_lhs_a, J_lhs_n, J_rhs_a, J_rhs_n;
    const auto se3 = se3_.groupPlus(other_se3, J_lhs_a.data(), J_rhs_a.data(), coupled, global);
    for (auto j = 0; j < Traits<Tangent<SE3<Scalar>>>::kNumParameters; ++j) {
      J_lhs_n.col(j) = NumericGroupMinus(NumericGroupPlus(se3_, coupled, global, j).groupPlus(other_se3), se3, coupled, global);
      J_rhs_n.col(j) = NumericGroupMinus(se3_.groupPlus(NumericGroupPlus(other_se3, coupled, global, j)), se3, coupled, global);
    }

    return J_lhs_n.isApprox(J_lhs_a, kNumericTolerance) && J_rhs_n.isApprox(J_rhs_a, kNumericTolerance);
  }

  [[nodiscard]] auto checkVectorPlusJacobian(const bool coupled) const -> bool {
    using Vector = SE3<Scalar>::Translation;
    const Vector input = Vector::Random();

    Jacobian<Vector, Tangent<SE3<Scalar>>> J_l_a, J_r_a, J_l_n, J_r_n;
    Jacobian<Vector> J_l_p_a, J_r_p_a, J_p_n;
    const auto output = se3_.vectorPlus(input);
    se3_.vectorPlus(input, J_l_a.data(), J_l_p_a.data(), coupled, true);
    se3_.vectorPlus(input, J_r_a.data(), J_r_p_a.data(), coupled, false);
    for (auto j = 0; j < Traits<Tangent<SE3<Scalar>>>::kNumParameters; ++j) {
      J_l_n.col(j) = (NumericGroupPlus(se3_, coupled, true, j).vectorPlus(input) - output) / kNumericIncrement;
      J_r_n.col(j) = (NumericGroupPlus(se3_, coupled, false, j).vectorPlus(input) - output) / kNumericIncrement;
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

  [[nodiscard]] auto checkGroupExponentialsJacobians(const bool global, const bool coupled) const -> bool {
    Jacobian<Tangent<SE3<Scalar>>> J_l_a, J_e_a;
    const auto tangent = se3_.toTangent(J_l_a.data(), coupled, global);
    const auto se3 = tangent.toManifold(J_e_a.data(), coupled, global);

    Jacobian<Tangent<SE3<Scalar>>> J_l_n, J_e_n;
    for (auto j = 0; j < Traits<Tangent<SE3<Scalar>>>::kNumParameters; ++j) {
      const auto d_tangent = Tangent<SE3<Scalar>>{tangent + kNumericIncrement * Tangent<SE3<Scalar>>::Unit(j)};
      J_l_n.col(j) = (NumericGroupPlus(se3_, coupled, global, j).toTangent() - tangent) / kNumericIncrement;
      J_e_n.col(j) = NumericGroupMinus(d_tangent.toManifold(), se3_, coupled, global);
    }

    return se3.isApprox(se3_, kNumericTolerance) &&
           (J_l_a * J_e_a).isIdentity(kNumericTolerance) &&
           J_l_n.isApprox(J_l_a, kNumericTolerance) &&
           J_e_n.isApprox(J_e_a, kNumericTolerance);
  }

  SE3<Scalar> se3_;

 private:
  static auto NumericGroupPlus(const Eigen::Ref<const SE3<Scalar>>& se3, const bool global, const bool coupled, const Eigen::Index i) -> SE3<Scalar> {
    const auto tau = Tangent<SE3<Scalar>>{kNumericIncrement * Tangent<SE3<Scalar>>::Unit(i)};

    if (coupled) {
      if (global) {
        return tau.toManifold().groupPlus(se3);
      } else {
        return se3.groupPlus(tau.toManifold());
      }
    } else {
      if (global) {
        return {tau.angular().toManifold().groupPlus(se3.rotation()), se3.translation() + tau.linear()};
      } else {
        return {se3.rotation().groupPlus(tau.angular().toManifold()), se3.translation() + tau.linear()};
      }
    }
  }

  static auto NumericGroupMinus(const Eigen::Ref<const SE3<Scalar>>& d_se3, const Eigen::Ref<const SE3<Scalar>>& se3, const bool global, const bool coupled) -> Tangent<SE3<Scalar>> {
    if (coupled) {
      if (global) {
        return d_se3.groupPlus(se3.groupInverse()).toTangent() / kNumericIncrement;
      } else {
        return se3.groupInverse().groupPlus(d_se3).toTangent() / kNumericIncrement;
      }
    } else {
      if (global) {
        Tangent<SE3<Scalar>> tangent;
        tangent.angular() = d_se3.rotation().groupPlus(se3.rotation().groupInverse()).toTangent() / kNumericIncrement;
        tangent.linear() = (d_se3.translation() - se3.translation()) / kNumericIncrement;
        return tangent;
      } else {
        Tangent<SE3<Scalar>> tangent;
        tangent.angular() = se3.rotation().groupInverse().groupPlus(d_se3.rotation()).toTangent() / kNumericIncrement;
        tangent.linear() = (d_se3.translation() - se3.translation()) / kNumericIncrement;
        return tangent;
      }
    }
  }
};

TEST_F(SE3Tests, GroupInverse) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3<Scalar>::Random();
    EXPECT_TRUE(checkGroupInverse());
    EXPECT_TRUE(checkGroupInverseJacobian(false, true));
    EXPECT_TRUE(checkGroupInverseJacobian(false, false));
    EXPECT_TRUE(checkGroupInverseJacobian(true, true));
    EXPECT_TRUE(checkGroupInverseJacobian(true, false));
  }
}

TEST_F(SE3Tests, GroupPlus) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3<Scalar>::Random();
    EXPECT_TRUE(checkGroupPlusJacobian(false, true));
    EXPECT_TRUE(checkGroupPlusJacobian(false, false));
    EXPECT_TRUE(checkGroupPlusJacobian(true, true));
    EXPECT_TRUE(checkGroupPlusJacobian(true, false));
  }
}

TEST_F(SE3Tests, VectorPlus) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3<Scalar>::Random();
    EXPECT_TRUE(checkVectorPlusJacobian(false));
    EXPECT_TRUE(checkVectorPlusJacobian(true));
  }
}

TEST_F(SE3Tests, GroupExponentials) {
  for (auto i = 0; i < kNumIterations; ++i) {
    se3_ = SE3<Scalar>::Random();
    EXPECT_TRUE(checkGroupExponentials());
    EXPECT_TRUE(checkGroupExponentialsJacobians(false, true));
    EXPECT_TRUE(checkGroupExponentialsJacobians(false, false));
    EXPECT_TRUE(checkGroupExponentialsJacobians(true, true));
    EXPECT_TRUE(checkGroupExponentialsJacobians(true, false));
  }
}

} // namespace hyper::tests
