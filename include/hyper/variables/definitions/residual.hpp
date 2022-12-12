/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#include "hyper/vector.hpp"

namespace hyper {

template <typename TScalar, int TNumRows, int TOptions = DefaultVectorStorageOption(TNumRows, 1)>
using Residual = Vector<TScalar, TNumRows, TOptions>;

template <typename TDerived, int TOptions = DefaultVectorStorageOption(TDerived::SizeAtCompileTime, 1)>
using ResidualN = VectorN<TDerived, TOptions>;

template <typename TScalar, int TOptions = DefaultVectorStorageOption(Eigen::Dynamic, 1)>
using ResidualX = VectorX<TScalar, TOptions>;

} // namespace hyper