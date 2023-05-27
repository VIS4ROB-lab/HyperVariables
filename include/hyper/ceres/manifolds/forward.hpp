/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

#ifndef HYPER_COMPILE_WITH_CERES
#error "Compile with Ceres flag must be set."
#endif

#if HYPER_COMPILE_WITH_CERES

namespace hyper::ceres::manifolds {

class ManifoldWrapper;

template <typename>
class Manifold;

}  // namespace hyper::ceres::manifolds

#endif
