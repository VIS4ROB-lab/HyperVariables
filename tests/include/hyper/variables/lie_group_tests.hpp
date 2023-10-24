/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

namespace hyper::variables::tests {

template <typename TGroup>
class LieGroupTests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kInc = 1e-6;
  static constexpr auto kTol = 1e-5;

  // Definitions.
  using Group = TGroup;
  using Tangent = Group::Tangent;
  using Translation = Group::Translation;
  using Adjoint = Group::Adjoint;

  using TangentJacobian = Group::TangentJacobian;
  using TranslationJacobian = Group::TranslationJacobian;
  using GroupToTangentJacobian = Group::GroupToTangentJacobian;
  using TangentToGroupJacobian = Group::TangentToGroupJacobian;
  using ActionJacobian = Group::ActionJacobian;

  [[nodiscard]] auto checkGroupInverseJacobian() const -> bool {
    TangentJacobian J_a, J_n;
    const auto i_q = group_.gInv(J_a.data());
    for (auto j = 0; j < Tangent::kNumParameters; ++j) {
      const Tangent inc = kInc * Tangent::Unit(j);
      J_n.col(j) = group_.tPlus(inc).gInv().tMinus(i_q) / kInc;
    }

    return J_n.isApprox(J_a, kTol);
  }

  [[nodiscard]] auto checkGroupPlusJacobian() const -> bool {
    const auto other_group = Group::Random();
    const auto group = group_.gPlus(other_group);

    TangentJacobian J_lhs_a, J_lhs_n, J_rhs_a, J_rhs_n;
    group_.gPlus(other_group, J_lhs_a.data(), J_rhs_a.data());
    for (auto j = 0; j < Tangent::kNumParameters; ++j) {
      const Tangent inc = kInc * Tangent::Unit(j);
      J_lhs_n.col(j) = group_.tPlus(inc).gPlus(other_group).tMinus(group) / kInc;
      J_rhs_n.col(j) = group_.gPlus(other_group.tPlus(inc)).tMinus(group) / kInc;
    }

    return J_lhs_n.isApprox(J_lhs_a, kTol) && J_rhs_n.isApprox(J_rhs_a, kTol);
  }

  [[nodiscard]] auto checkGroupActionJacobian() const -> bool {
    const Translation input = Translation::Random();

    ActionJacobian J_a, J_n;
    TranslationJacobian J_v_a, J_v_n;
    const auto output = group_.act(input);
    group_.act(input, J_a.data(), J_v_a.data());
    for (auto j = 0; j < Tangent::kNumParameters; ++j) {
      const Tangent inc = kInc * Tangent::Unit(j);
      J_n.col(j) = (group_.tPlus(inc).act(input) - output) / kInc;
    }
    for (auto j = 0; j < Translation::kNumParameters; ++j) {
      J_v_n.col(j) = (group_.act(input + kInc * Translation::Unit(j)) - output) / kInc;
    }

    return J_n.isApprox(J_a, kTol) && J_v_n.isApprox(J_v_a, kTol);
  }

  [[nodiscard]] auto checkGroupExponentialMap() const -> bool {
    const auto group = group_.gLog().gExp();
    return group.gIsApprox(group_, kTol);
  }

  [[nodiscard]] auto checkGroupExponentialMapJacobians() const -> bool {
    TangentJacobian J_l_a, J_e_a;
    const auto tangent = group_.gLog(J_l_a.data());
    const auto group = tangent.gExp(J_e_a.data());

    TangentJacobian J_l_n, J_e_n;
    for (auto j = 0; j < Tangent::kNumParameters; ++j) {
      const Tangent inc = kInc * Tangent::Unit(j);
      const Tangent d_tangent = tangent + inc;
      J_l_n.col(j) = (group_.tPlus(inc).gLog() - tangent) / kInc;
      J_e_n.col(j) = d_tangent.gExp().tMinus(group_) / kInc;
    }

    return group.gIsApprox(group_, kTol) && (J_l_a * J_e_a).isIdentity(kTol) && J_l_n.isApprox(J_l_a, kTol) && J_e_n.isApprox(J_e_a, kTol);
  }

  [[nodiscard]] auto checkTangentJacobian() const -> bool {
    const auto J_a_plus = group_.tPlusJacobian();
    const auto J_a_minus = group_.tMinusJacobian();

    GroupToTangentJacobian J_n_plus;
    for (auto j = 0; j < Tangent::kNumParameters; ++j) {
      const Tangent d_tangent = kInc * Tangent::Unit(j);
      J_n_plus.col(j) = (group_.tPlus(d_tangent).asVector() - group_.asVector()) / kInc;
    }

    return J_n_plus.isApprox(J_a_plus, kTol) && (J_a_minus * J_a_plus).isIdentity(kTol);
  }

  [[nodiscard]] auto checkGroupAdjoint() const -> bool {
    Adjoint A_a, A_n;

    A_a = group_.gAdj();
    for (auto j = 0; j < Tangent::kNumParameters; ++j) {
      const Tangent inc = kInc * Tangent::Unit(j);
      A_n.col(j) = group_.tPlus(inc).gPlus(group_.gInv()).gLog() / kInc;
    }

    return A_n.isApprox(A_a, kTol);
  }

  auto checkGroupOperators(int iterations) -> void {
    for (auto i = 0; i < iterations; ++i) {
      this->group_ = Group::Random();
      EXPECT_TRUE(this->checkGroupInverseJacobian());
      EXPECT_TRUE(this->checkGroupPlusJacobian());
      EXPECT_TRUE(this->checkGroupActionJacobian());
      EXPECT_TRUE(this->checkGroupExponentialMap());
      EXPECT_TRUE(this->checkGroupExponentialMapJacobians());
      EXPECT_TRUE(this->checkGroupAdjoint());
    }
  }

  auto checkTangentOperators(int iterations) -> void {
    for (auto i = 0; i < iterations; ++i) {
      this->group_ = Group::Random();
      EXPECT_TRUE(this->checkTangentJacobian());
    }
  }

  Group group_;
};

}  // namespace hyper::variables::tests
