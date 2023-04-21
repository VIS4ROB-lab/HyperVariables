/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/jacobian.hpp"
#include "hyper/variables/rn.hpp"

namespace hyper::variables {

template <typename TDerived>
class SensitivityBase : public RnBase<TDerived> {
 public:
  // Constants.
  static constexpr auto kOrder = Traits<TDerived>::kOrder;

  // Definitions.
  using Base = RnBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  using OrderVector = hyper::Vector<Scalar, kOrder>;
  using OrderMatrix = hyper::Matrix<Scalar, kOrder, kOrder>;

  using Input = OrderVector;
  using InputJacobian = hyper::JacobianNM<OrderVector>;
  using ParameterJacobian = hyper::JacobianNM<OrderVector, Base>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(SensitivityBase)

  /// Returns this in matrix form.
  /// \return Sensitivity as matrix.
  [[nodiscard]] inline auto asMatrix() const { return this->reshaped(kOrder, kOrder); }

  /// Returns this in matrix form.
  /// \return Sensitivity as matrix.
  [[nodiscard]] inline auto asMatrix() { return this->reshaped(kOrder, kOrder); }

  /// Acts on the input.
  /// \param input Input to act on.
  /// \param J_i Input Jacobian.
  /// \param J_p Parameter Jacobian.
  /// \return Output.
  auto act(const Eigen::Ref<const Input>& input, Scalar* J_i = nullptr, Scalar* J_p = nullptr) const -> OrderVector {
    const auto this_as_matrix = asMatrix();

    if (J_i) {
      Eigen::Map<InputJacobian>{J_i}.noalias() = this_as_matrix;
    }

    if (J_p) {
      auto J = Eigen::Map<ParameterJacobian>{J_p};
      J.setZero();

      auto k = 0;
      for (auto i = 0; i < kOrder; ++i) {
        for (auto j = 0; j < kOrder; ++j) {
          J(j, i * kOrder + j) = input[k];
        }
        ++k;
      }
    }

    return this_as_matrix * input;
  }
};

template <typename TScalar, int TOrder>
class Sensitivity final : public SensitivityBase<Sensitivity<TScalar, TOrder>> {
 public:
  using Base = SensitivityBase<Sensitivity<TScalar, TOrder>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(Sensitivity)
};

}  // namespace hyper::variables

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(hyper::variables::Sensitivity, int)
