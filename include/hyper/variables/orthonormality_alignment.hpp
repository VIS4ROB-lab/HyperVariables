/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/variables/cartesian.hpp"
#include "hyper/variables/jacobian.hpp"

namespace hyper {

template <typename TDerived>
class OrthonormalityAlignmentBase
    : public CartesianBase<TDerived> {
 public:
  using Scalar = typename Traits<TDerived>::Scalar;
  using ScalarWithConstIfNotLvalue = typename Traits<TDerived>::ScalarWithConstIfNotLvalue;
  using Base = CartesianBase<TDerived>;
  using Base::Base;

  using AlignmentMatrix = Eigen::Matrix<Scalar, Traits<TDerived>::kOrder, Traits<TDerived>::kOrder>;
  using Input = Cartesian<Scalar, Traits<TDerived>::kOrder>;
  using Output = Cartesian<Scalar, Traits<TDerived>::kOrder>;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(OrthonormalityAlignmentBase)

  /// Creates and identity orthonormality alignment.
  /// \return Identity orthonormality alignment.
  static auto Identity() -> OrthonormalityAlignment<Scalar, Traits<TDerived>::kOrder> {
    OrthonormalityAlignment<Scalar, Traits<TDerived>::kOrder> orthonormality_alignment;
    return orthonormality_alignment.setIdentity();
  }

  /// Accessor to the diagonal parameters.
  /// \return Scaling parameters.
  [[nodiscard]] auto diagonalParameters() const {
    return this->template head<Traits<TDerived>::kNumDiagonalParameters>();
  }

  /// Modifier of the diagonal parameters.
  /// \return Scaling parameters.
  auto diagonalParameters() {
    return this->template head<Traits<TDerived>::kNumDiagonalParameters>();
  }

  /// Accessor to the off-diagonal parameters.
  /// \return Orthogonality parameters.
  [[nodiscard]] auto offDiagonalParameters() const {
    return this->template tail<Traits<TDerived>::kNumOffDiagonalParameters>();
  }

  /// Modifier of the off-diagonal parameters.
  /// \return Orthogonality parameters.
  auto offDiagonalParameters() {
    return this->template tail<Traits<TDerived>::kNumOffDiagonalParameters>();
  }

  /// Sets the underlying parameters to
  /// represent the identity alignment matrix.
  auto setIdentity() -> OrthonormalityAlignmentBase& {
    diagonalParameters().setOnes();
    offDiagonalParameters().setZero();
    return *this;
  }

  /// Returns the parameters in their matrix form.
  /// \return Orthonormality alignment as matrix.
  [[nodiscard]] auto asMatrix() const -> AlignmentMatrix {
    auto A = scalingMatrix();

    Eigen::Index i = 0;
    const auto off_diagonal_parameters = offDiagonalParameters();
    for (Eigen::Index j = 0; j < Traits<TDerived>::kOrder - 1; ++j) {
      for (Eigen::Index k = j + 1; k < Traits<TDerived>::kOrder; ++k) {
        A(k, j) = off_diagonal_parameters[i];
        ++i;
      }
    }
    return A;
  }

  /// Returns the scaling parameters in their matrix form.
  /// \return Scaling parameters as matrix.
  [[nodiscard]] auto scalingMatrix() const -> AlignmentMatrix {
    return diagonalParameters().asDiagonal();
  }

  /// Returns the alignment parameters in their matrix form.
  /// \return Alignment parameters as matrix.
  [[nodiscard]] auto alignmentMatrix() const -> AlignmentMatrix {
    auto A = AlignmentMatrix::Identity().eval();

    Eigen::Index i = 0;
    const auto i_diagonal_parameters = diagonalParameters().cwiseInverse().eval();
    const auto off_diagonal_parameters = offDiagonalParameters();
    for (Eigen::Index j = 0; j < Traits<TDerived>::kOrder - 1; ++j) {
      for (Eigen::Index k = j + 1; k < Traits<TDerived>::kOrder; ++k) {
        A(k, j) = i_diagonal_parameters[k] * off_diagonal_parameters[i];
        ++i;
      }
    }
    return A;
  }

  /// Aligns an input.
  /// \param input Input vector to align.
  /// \param raw_J_i Input Jacobian.
  /// \param raw_J_p Parameter Jacobian.
  /// \return Aligned input.
  auto align(const Eigen::Ref<const typename Traits<Input>::Base>& input, Scalar* raw_J_i = nullptr, Scalar* raw_J_p = nullptr) const -> Output {
    const auto A = asMatrix();

    if (raw_J_i) {
      Eigen::Map<Jacobian<Output, Input>>{raw_J_i}.noalias() = A;
    }

    if (raw_J_p) {
      auto J = Eigen::Map<Jacobian<Output, TDerived>>{raw_J_p};
      J.setZero();

      for (auto i = 0; i < Traits<TDerived>::kOrder; ++i) {
        J(i, i) = input[i];
      }

      auto j = Traits<TDerived>::kOrder;
      for (auto i = 1; i < Traits<TDerived>::kOrder; ++i) {
        const auto bound = Traits<TDerived>::kOrder - i;
        for (auto k = 0; k < bound; ++k) {
          J(i + k, j + k) = input[i - 1];
        }
        j += bound;
      }
    }

    return A * input;
  }
};

template <typename TScalar, int TOrder>
class OrthonormalityAlignment final
    : public OrthonormalityAlignmentBase<OrthonormalityAlignment<TScalar, TOrder>> {
 public:
  using Base = OrthonormalityAlignmentBase<OrthonormalityAlignment<TScalar, TOrder>>;
  using Base::Base;

  HYPER_INHERIT_ASSIGNMENT_OPERATORS(OrthonormalityAlignment)
};

} // namespace hyper

HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(hyper::OrthonormalityAlignment, int)
