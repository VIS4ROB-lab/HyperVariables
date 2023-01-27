/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/sensitivity.hpp"

namespace hyper::variables::tests {

class SensitivityTests : public testing::Test {
 protected:
  // Constants.
  static constexpr auto kOuterItr = 5;
  static constexpr auto kInnerItr = 10;
  static constexpr auto kInc = 1e-6;
  static constexpr auto kTol = 1e-8;

  // Definitions.
  using Scalar = double;
  using Sensitivity = variables::Sensitivity<Scalar, 3>;

  using Input = Sensitivity::Input;
  using InputJacobian = Sensitivity::InputJacobian;
  using ParameterJacobian = Sensitivity::ParameterJacobian;

  [[nodiscard]] auto checkInputJacobian() const -> bool {
    Input input = Input::Random();

    InputJacobian J_a;
    const auto output = sensitivity_.act(input, J_a.data(), nullptr);

    InputJacobian J_n;
    for (auto j = 0; j < Sensitivity::kOrder; ++j) {
      const Input d_input = input + kInc * Input::Unit(j);
      const auto d_output = sensitivity_.act(d_input, nullptr, nullptr);
      J_n.col(j) = (d_output - output) / kInc;
    }

    return J_n.isApprox(J_a, kTol);
  }

  [[nodiscard]] auto checkParameterJacobian() const -> bool {
    Input input = Input::Random();

    ParameterJacobian J_a;
    const auto output = sensitivity_.act(input, nullptr, J_a.data());

    ParameterJacobian J_n;
    for (auto j = 0; j < sensitivity_.size(); ++j) {
      const Sensitivity d_alignment = sensitivity_ + kInc * Sensitivity::Unit(j);
      const auto d_output = d_alignment.act(input, nullptr, nullptr);
      J_n.col(j) = (d_output - output) / kInc;
    }

    return J_n.isApprox(J_a, kTol);
  }

  Sensitivity sensitivity_;
};

TEST_F(SensitivityTests, InputJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    sensitivity_.setRandom();
    for (auto j = 0; j < kInnerItr; ++j) {
      EXPECT_TRUE(checkInputJacobian());
    }
  }
}

TEST_F(SensitivityTests, ParameterJacobian) {
  for (auto i = 0; i < kOuterItr; ++i) {
    sensitivity_.setRandom();
    for (auto j = 0; j < kInnerItr; ++j) {
      EXPECT_TRUE(checkParameterJacobian());
    }
  }
}

}  // namespace hyper::variables::tests
