/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/rn.hpp"

namespace hyper::variables {

template <typename TDerived>
class OrthonormalityAlignmentBase : public RnBase<TDerived> {
 public:
  // Constants.
  static constexpr auto kOrder = Traits<TDerived>::kOrder;
  static constexpr auto kNumDiagonalParameters = kOrder;
  static constexpr auto kNumOffDiagonalParameters = ((kOrder - 1) * kOrder) / 2;

  // Definitions.
  using Base = RnBase<TDerived>;
  using Scalar = typename Base::Scalar;
  using Base::Base;

  using Input = Rn<kOrder>;
  using InputJacobian = hyper::JacobianNM<Input>;
  using ParameterJacobian = hyper::JacobianNM<Input, Base>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(OrthonormalityAlignmentBase)

  /// Creates and identity orthonormality alignment.
  /// \return Identity orthonormality alignment.
  static auto Identity() -> OrthonormalityAlignment<kOrder> {
    OrthonormalityAlignment<kOrder> orthonormality_alignment;
    return orthonormality_alignment.setIdentity();
  }

  /// Accessor to the diagonal parameters.
  /// \return Scaling parameters.
  [[nodiscard]] auto diagonalParameters() const { return this->template head<kNumDiagonalParameters>(); }

  /// Modifier of the diagonal parameters.
  /// \return Scaling parameters.
  auto diagonalParameters() { return this->template head<kNumDiagonalParameters>(); }

  /// Accessor to the off-diagonal parameters.
  /// \return Orthogonality parameters.
  [[nodiscard]] auto offDiagonalParameters() const { return this->template tail<kNumOffDiagonalParameters>(); }

  /// Modifier of the off-diagonal parameters.
  /// \return Orthogonality parameters.
  auto offDiagonalParameters() { return this->template tail<kNumOffDiagonalParameters>(); }

  /// Sets the underlying parameters to
  /// represent the identity alignment matrix.
  auto setIdentity() -> OrthonormalityAlignmentBase& {
    diagonalParameters().setOnes();
    offDiagonalParameters().setZero();
    return *this;
  }

  /// Returns the parameters in their matrix form.
  /// \return Orthonormality alignment as matrix.
  [[nodiscard]] auto asMatrix() const -> Matrix<kOrder> {
    auto matrix = scalingMatrix();

    auto i = 0;
    const auto off_diagonal_parameters = offDiagonalParameters();
    for (auto j = 0; j < kOrder - 1; ++j) {
      for (auto k = j + 1; k < kOrder; ++k) {
        matrix(k, j) = off_diagonal_parameters[i];
        ++i;
      }
    }
    return matrix;
  }

  /// Returns the scaling parameters in their matrix form.
  /// \return Scaling parameters as matrix.
  [[nodiscard]] auto scalingMatrix() const -> Matrix<kOrder> { return diagonalParameters().asDiagonal(); }

  /// Returns the alignment parameters in their matrix form.
  /// \return Alignment parameters as matrix.
  [[nodiscard]] auto alignmentMatrix() const -> Matrix<kOrder> {
    Matrix<kOrder> A = Matrix<kOrder>::Identity();

    auto i = 0;
    const auto i_diagonal_parameters = diagonalParameters().cwiseInverse().eval();
    const auto off_diagonal_parameters = offDiagonalParameters();
    for (auto j = 0; j < kOrder - 1; ++j) {
      for (auto k = j + 1; k < kOrder; ++k) {
        A(k, j) = i_diagonal_parameters[k] * off_diagonal_parameters[i];
        ++i;
      }
    }
    return A;
  }

  /// Acts on the input.
  /// \param input Input to act on.
  /// \param J_i Input Jacobian.
  /// \param J_p Parameter Jacobian.
  /// \return Output.
  auto act(const Eigen::Ref<const Input>& input, Scalar* J_i = nullptr, Scalar* J_p = nullptr) const -> Rn<kOrder> {
    const auto this_as_matrix = asMatrix();

    if (J_i) {
      Eigen::Map<InputJacobian>{J_i}.noalias() = this_as_matrix;
    }

    if (J_p) {
      auto J = Eigen::Map<ParameterJacobian>{J_p};
      J.setZero();

      for (auto i = 0; i < kOrder; ++i) {
        J(i, i) = input[i];
      }

      auto j = kOrder;
      for (auto i = 1; i < kOrder; ++i) {
        const auto bound = kOrder - i;
        for (auto k = 0; k < bound; ++k) {
          J(i + k, j + k) = input[i - 1];
        }
        j += bound;
      }
    }

    return this_as_matrix * input;
  }
};

template <int TOrder>
class OrthonormalityAlignment final : public OrthonormalityAlignmentBase<OrthonormalityAlignment<TOrder>> {
 public:
  using Base = OrthonormalityAlignmentBase<OrthonormalityAlignment<TOrder>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(OrthonormalityAlignment)
};

}  // namespace hyper::variables

HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_INTERFACE(hyper::variables, OrthonormalityAlignment, int)
