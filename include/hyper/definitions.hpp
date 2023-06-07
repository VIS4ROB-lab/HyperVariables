/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#ifndef HYPER_COMPILE_WITH_GLOBAL_LIE_GROUP_DERIVATIVES
#error "Compile with global Lie group derivatives flag must be set."
#endif

namespace hyper {

using Time = double;
using Scalar = double;

template <typename>
struct Traits;

template <typename>
struct NumTraits;

template <typename TPointer, typename TSize = std::int32_t>
using Partition = std::vector<std::pair<TPointer, TSize>>;

template <typename TPointer, typename TSize = std::int32_t>
using Partitions = std::vector<Partition<TPointer, TSize>>;

template <>
struct NumTraits<float> {
  static constexpr float kSmallAngleTolerance = 1e-4;
};

template <>
struct NumTraits<double> {
  static constexpr double kSmallAngleTolerance = 1e-8;
};

static constexpr auto kGravityNorm = Scalar{9.80741};  // Magnitude of local gravity for Zurich in [m/s²].

}  // namespace hyper
