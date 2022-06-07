/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

namespace hyper {

#define HYPER_INHERIT_ASSIGNMENT_OPERATORS(DERIVED) \
  using Base::operator=;                            \
  DERIVED(const DERIVED&) = default;                \
  auto operator=(const DERIVED& other)->DERIVED& {  \
    Base::operator=(other);                         \
    return *this;                                   \
  }

#define HYPER_DECLARE_EIGEN_MAP_TRAITS(CLASS)                                          \
  template <typename TScalar, int TMapOptions>                                         \
  struct Traits<Eigen::Map<CLASS<TScalar>, TMapOptions>> final                         \
      : public Traits<CLASS<TScalar>> {                                                \
    using Base = Eigen::Map<typename Traits<CLASS<TScalar>>::Base, TMapOptions>;       \
  };                                                                                   \
                                                                                       \
  template <typename TScalar, int TMapOptions>                                         \
  struct Traits<Eigen::Map<const CLASS<TScalar>, TMapOptions>> final                   \
      : public Traits<CLASS<TScalar>> {                                                \
    using ScalarWithConstIfNotLvalue = const typename Traits<CLASS<TScalar>>::Scalar;  \
    using Base = Eigen::Map<const typename Traits<CLASS<TScalar>>::Base, TMapOptions>; \
  };

#define HYPER_DECLARE_TEMPLATED_EIGEN_MAP_TRAITS(CLASS, TYPE)                                \
  template <typename TScalar, TYPE TArg, int TMapOptions>                                    \
  struct Traits<Eigen::Map<CLASS<TScalar, TArg>, TMapOptions>> final                         \
      : public Traits<CLASS<TScalar, TArg>> {                                                \
    using Base = Eigen::Map<typename Traits<CLASS<TScalar, TArg>>::Base, TMapOptions>;       \
  };                                                                                         \
                                                                                             \
  template <typename TScalar, TYPE TArg, int TMapOptions>                                    \
  struct Traits<Eigen::Map<const CLASS<TScalar, TArg>, TMapOptions>> final                   \
      : public Traits<CLASS<TScalar, TArg>> {                                                \
    using ScalarWithConstIfNotLvalue = const typename Traits<CLASS<TScalar, TArg>>::Scalar;  \
    using Base = Eigen::Map<const typename Traits<CLASS<TScalar, TArg>>::Base, TMapOptions>; \
  };

#define HYPER_DECLARE_EIGEN_MAP(CLASS)                                \
  namespace Eigen {                                                   \
  using namespace hyper;                                              \
  template <typename TScalar, int TMapOptions>                        \
  class Map<CLASS<TScalar>, TMapOptions> final                        \
      : public CLASS##Base<Map<CLASS<TScalar>, TMapOptions>> {        \
   public:                                                            \
    using Base = CLASS##Base<Map<CLASS<TScalar>, TMapOptions>>;       \
    using Base::Base;                                                 \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)                           \
  };                                                                  \
                                                                      \
  template <typename TScalar, int TMapOptions>                        \
  class Map<const CLASS<TScalar>, TMapOptions> final                  \
      : public CLASS##Base<Map<const CLASS<TScalar>, TMapOptions>> {  \
   public:                                                            \
    using Base = CLASS##Base<Map<const CLASS<TScalar>, TMapOptions>>; \
    using Base::Base;                                                 \
  };                                                                  \
  }

#define HYPER_DECLARE_TEMPLATED_EIGEN_MAP(CLASS, TYPE)                      \
  namespace Eigen {                                                         \
  using namespace hyper;                                                    \
  template <typename TScalar, TYPE TArg, int TMapOptions>                   \
  class Map<CLASS<TScalar, TArg>, TMapOptions> final                        \
      : public CLASS##Base<Map<CLASS<TScalar, TArg>, TMapOptions>> {        \
   public:                                                                  \
    using Base = CLASS##Base<Map<CLASS<TScalar, TArg>, TMapOptions>>;       \
    using Base::Base;                                                       \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)                                 \
  };                                                                        \
                                                                            \
  template <typename TScalar, TYPE TArg, int TMapOptions>                   \
  class Map<const CLASS<TScalar, TArg>, TMapOptions> final                  \
      : public CLASS##Base<Map<const CLASS<TScalar, TArg>, TMapOptions>> {  \
   public:                                                                  \
    using Base = CLASS##Base<Map<const CLASS<TScalar, TArg>, TMapOptions>>; \
    using Base::Base;                                                       \
  };                                                                        \
  }

#define HYPER_DECLARE_TANGENT_MAP_TRAITS(CLASS)                                                          \
  template <typename TScalar, int TMapOptions>                                                           \
  struct Traits<Eigen::Map<Tangent<CLASS<TScalar>>, TMapOptions>>                                        \
      : Traits<Tangent<CLASS<TScalar>>> {                                                                \
    using Base = typename Eigen::Map<typename Traits<Tangent<CLASS<TScalar>>>::Base, TMapOptions>;       \
  };                                                                                                     \
                                                                                                         \
  template <typename TScalar, int TMapOptions>                                                           \
  struct Traits<Eigen::Map<const Tangent<CLASS<TScalar>>, TMapOptions>>                                  \
      : Traits<Tangent<CLASS<TScalar>>> {                                                                \
    using ScalarWithConstIfNotLvalue = const typename Traits<Tangent<CLASS<TScalar>>>::Scalar;           \
    using Base = typename Eigen::Map<const typename Traits<Tangent<CLASS<TScalar>>>::Base, TMapOptions>; \
  };

#define HYPER_DECLARE_ALGEBRA_MAP(CLASS)                                              \
  namespace Eigen {                                                                   \
  using namespace hyper;                                                              \
  template <typename TScalar, int TMapOptions>                                        \
  class Map<Algebra<CLASS<TScalar>>, TMapOptions> final                               \
      : public CLASS##TangentBase<Map<Algebra<CLASS<TScalar>>, TMapOptions>> {        \
   public:                                                                            \
    using Base = CLASS##TangentBase<Map<Algebra<CLASS<TScalar>>, TMapOptions>>;       \
    using Base::Base;                                                                 \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)                                           \
  };                                                                                  \
                                                                                      \
  template <typename TScalar, int TMapOptions>                                        \
  class Map<const Algebra<CLASS<TScalar>>, TMapOptions> final                         \
      : public CLASS##TangentBase<Map<const Algebra<CLASS<TScalar>>, TMapOptions>> {  \
   public:                                                                            \
    using Base = CLASS##TangentBase<Map<const Algebra<CLASS<TScalar>>, TMapOptions>>; \
    using Base::Base;                                                                 \
  };                                                                                  \
  }

#define HYPER_DECLARE_TANGENT_MAP(CLASS)                                              \
  namespace Eigen {                                                                   \
  using namespace hyper;                                                              \
  template <typename TScalar, int TMapOptions>                                        \
  class Map<Tangent<CLASS<TScalar>>, TMapOptions> final                               \
      : public CLASS##TangentBase<Map<Tangent<CLASS<TScalar>>, TMapOptions>> {        \
   public:                                                                            \
    using Base = CLASS##TangentBase<Map<Tangent<CLASS<TScalar>>, TMapOptions>>;       \
    using Base::Base;                                                                 \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)                                           \
  };                                                                                  \
                                                                                      \
  template <typename TScalar, int TMapOptions>                                        \
  class Map<const Tangent<CLASS<TScalar>>, TMapOptions> final                         \
      : public CLASS##TangentBase<Map<const Tangent<CLASS<TScalar>>, TMapOptions>> {  \
   public:                                                                            \
    using Base = CLASS##TangentBase<Map<const Tangent<CLASS<TScalar>>, TMapOptions>>; \
    using Base::Base;                                                                 \
  };                                                                                  \
  }

} // namespace hyper
