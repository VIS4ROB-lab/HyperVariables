/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/orthonormality_alignment.hpp"

namespace hyper::variables::tests {

class OrthonormalityAlignmentTests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kOuterItr = 5;
  static constexpr auto kInnerItr = 25;
  static constexpr auto kInc = 1e-6;
  static constexpr auto kTol = 1e-8;

  // Definitions.
  using Scalar = double;
  using Alignment = variables::OrthonormalityAlignment<Scalar, 3>;

  using Input = Alignment::Input;
  using InputJacobian = Alignment::InputJacobian;
  using ParameterJacobian = Alignment::ParameterJacobian;

  [[nodiscard]] auto checkDuality() const -> bool {
    const auto S = alignment_.scalingMatrix();
    const auto A = alignment_.alignmentMatrix();
    const auto SA = alignment_.asMatrix();
    return (S * A).isApprox(SA, kTol);
  }

  [[nodiscard]] auto checkInputJacobian() const -> bool {
    Input input = Input::Random();

    InputJacobian J_a;
    const auto output = alignment_.act(input, J_a.data(), nullptr);

    InputJacobian J_n;
    for (auto j = 0; j < Alignment::kOrder; ++j) {
      const Input d_input = input + kInc * Input::Unit(j);
      const auto d_output = alignment_.act(d_input, nullptr, nullptr);
      J_n.col(j) = (d_output - output) / kInc;
    }

    return J_n.isApprox(J_a, kTol);
  }

  [[nodiscard]] auto checkParameterJacobian() const -> bool {
    Input input = Input::Random();

    ParameterJacobian J_a;
    const auto output = alignment_.act(input, nullptr, J_a.data());

    ParameterJacobian J_n;
    for (auto j = 0; j < alignment_.size(); ++j) {
      const Alignment d_alignment = alignment_ + kInc * Alignment::Unit(j);
      const auto d_output = d_alignment.act(input, nullptr, nullptr);
      J_n.col(j) = (d_output - output) / kInc;
    }

    return J_n.isApprox(J_a, kTol);
  }

  Alignment alignment_;
};

TEST_F(OrthonormalityAlignmentTests, Duality) {
  for (auto i = 0; i < kOuterItr; ++i) {
    alignment_.setRandom();
    EXPECT_TRUE(checkDuality());
  }
}

TEST_F(OrthonormalityAlignmentTests, InputJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    alignment_.setRandom();
    for (auto j = 0; j < kInnerItr; ++j) {
      EXPECT_TRUE(checkInputJacobian());
    }
  }
}

TEST_F(OrthonormalityAlignmentTests, ParameterJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    alignment_.setRandom();
    for (auto j = 0; j < kInnerItr; ++j) {
      EXPECT_TRUE(checkParameterJacobian());
    }
  }
}

}  // namespace hyper::variables::tests
