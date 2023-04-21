/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#ifdef HYPER_COMPILE_WITH_CERES

#include "hyper/ceres/manifolds/wrapper.hpp"
#include "hyper/variables/rn.hpp"

namespace hyper::ceres::manifolds {

template <typename TVariable>
class Manifold final : public ManifoldWrapper {
 public:
  // Definitions.
  using Variable = TVariable;

  /// Constructor from constancy flag.
  /// \param constant Constancy flag.
  explicit Manifold(const bool constant = false) : Manifold{TVariable::kNumParameters, constant} {}

  /// Constructor from number of parameters and constancy flag.
  /// \param num_parameters Number of parameters.
  /// \param constant Constancy flag.
  Manifold(const int num_parameters, const bool constant) : ManifoldWrapper{CreateManifold(num_parameters, constant)} {}

 private:
  /// Creates a constant or non-constant manifold.
  /// \param num_parameters Number of parameters.
  /// \param constant Constancy flag.
  /// \return Manifold.
  static auto CreateManifold(const int num_parameters, const bool constant) -> std::unique_ptr<::ceres::Manifold> {
    if (constant) {
      return std::make_unique<::ceres::SubsetManifold>(num_parameters, ManifoldWrapper::ConstancyMask(num_parameters));
    } else {
      return std::make_unique<::ceres::EuclideanManifold<TVariable::kNumParameters>>(num_parameters);
    }
  }
};

}  // namespace hyper::ceres::manifolds

#endif
