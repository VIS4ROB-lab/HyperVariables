/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#include <gtest/gtest.h>

#include "hyper/variables/groups/adapters.hpp"

namespace hyper::variables::tests {

template <typename TVariable>
class JacobianAdapterTests : public ::testing::Test {
 public:
  // Constants.
  static constexpr auto kItr = 10;

  // Definitions.
  using Variable = TVariable;
  using Tangent = variables::Tangent<Variable>;
  // using VariableJacobian = variables::JacobianNM<Variable>;
  using TangentJacobian = variables::JacobianNM<Tangent>;

  /// Checks the adapter duality.
  /// \return True on success.
  auto checkDuality() -> bool {
    variable = Variable::Random();
    const auto J_l = JacobianAdapter<Variable>(variable.data());
    const auto J_p = InverseJacobianAdapter<Variable>(variable.data());
    return (J_l * J_p).isApprox(TangentJacobian::Identity());
  }

  Variable variable;
};

using TestTypes = ::testing::Types<SU2<double>, SE3<double>>;
TYPED_TEST_SUITE(JacobianAdapterTests, TestTypes);

TYPED_TEST(JacobianAdapterTests, Duality) {
  for (auto i = 0; i < TestFixture::kItr; ++i) {
    EXPECT_TRUE(this->checkDuality());
  }
}

}  // namespace hyper::variables::tests
