/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "Eigen/Dense"

#include "hyper/matrix.hpp"

namespace hyper {

/// Inverts a positive semi-definite matrix.
/// \tparam TScalar Scalar type.
/// \tparam TSize Size type.
/// \param matrix Input matrix
/// \param full_rank True if matrix has full rank.
/// \return Inverse.
/// See https://github.com/ceres-solver/ceres-solver/blob/master/internal/ceres/invert_psd_matrix.h.
template <typename TScalar, int TSize>
auto invertPSDMatrix(const Eigen::Ref<const Matrix<TScalar, TSize, TSize>>& matrix, const bool full_rank) -> Matrix<TScalar, TSize, TSize> {
  using TMatrix = Matrix<TScalar, TSize, TSize>;
  using TSVDMatrix = Matrix<TScalar, TSize, Eigen::Dynamic>;

  const auto size = matrix.rows();

  if (full_rank) {
    if (size > 0 && size < 5) {
      return matrix.inverse();
    }
    return matrix.template selfadjointView<Eigen::Lower>().llt().solve(TMatrix::Identity(size, size));
  }

  Eigen::JacobiSVD<TSVDMatrix> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.solve(TSVDMatrix::Identity(size, size));
}

} // namespace hyper
