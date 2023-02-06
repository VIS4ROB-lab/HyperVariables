/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/matrix.hpp"

namespace hyper {

template <typename TScalar, int TNumRows, int TOptions = DefaultStorageOption(TNumRows, 1)>
using Vector = Eigen::Matrix<TScalar, TNumRows, 1, TOptions>;

template <typename TDerived, int TOptions = DefaultVectorStorageOption(TDerived::SizeAtCompileTime, 1)>
using VectorN = Vector<typename TDerived::Scalar, TDerived::SizeAtCompileTime, TOptions>;

template <typename TScalar, int TOptions = DefaultStorageOption(Eigen::Dynamic, 1)>
using VectorX = Vector<TScalar, Eigen::Dynamic, TOptions>;

}  // namespace hyper
