/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

/// Hat-operator (see https://en.wikipedia.org/wiki/Hat_operator).
/// \return Skew-symmetric matrix representation of a vector.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE auto hat() const -> Eigen::Matrix<Scalar, 3, 3> {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
  Eigen::Matrix<Scalar, 3, 3> Wx;
  Wx(0, 0) = static_cast<Scalar>(0);
  Wx(1, 0) = (*this)[2];
  Wx(2, 0) = -(*this)[1];
  Wx(0, 1) = -(*this)[2];
  Wx(1, 1) = static_cast<Scalar>(0);
  Wx(2, 1) = (*this)[0];
  Wx(0, 2) = (*this)[1];
  Wx(1, 2) = -(*this)[0];
  Wx(2, 2) = static_cast<Scalar>(0);
  return Wx;
}

/// Vee-operator (inverse of hat-operator).
/// \return Vector representation of a skew-symmetric matrix.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE auto vee() const -> Eigen::Matrix<Scalar, 3, 1> {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(Derived, 3, 3);
  Eigen::Matrix<Scalar, 3, 1> w;
  w[0] = (*this)(2, 1);
  w[1] = (*this)(0, 2);
  w[2] = (*this)(1, 0);
  return w;
}
