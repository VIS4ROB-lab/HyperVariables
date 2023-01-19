/// This file is subject to the terms and conditions defined in
/// the 'LICENSE' file, which is part of this repository.

#pragma once

namespace hyper::variables {

#define HYPER_INHERIT_ASSIGNMENT_OPERATORS(DERIVED) \
  using Base::operator=;                            \
  DERIVED(const DERIVED&) = default;                \
  auto operator=(const DERIVED& other)->DERIVED& {  \
    Base::operator=(other);                         \
    return *this;                                   \
  }

#define HYPER_DECLARE_EIGEN_CLASS_TRAITS(BASE, NAME)                                                  \
  template <typename TScalar, int TMapOptions>                                                        \
  struct Traits<Eigen::BASE<NAME<TScalar>, TMapOptions>> final : public Traits<NAME<TScalar>> {       \
    using Base = Eigen::BASE<typename Traits<NAME<TScalar>>::Base, TMapOptions>;                      \
  };                                                                                                  \
                                                                                                      \
  template <typename TScalar, int TMapOptions>                                                        \
  struct Traits<Eigen::BASE<const NAME<TScalar>, TMapOptions>> final : public Traits<NAME<TScalar>> { \
    using Base = Eigen::BASE<const typename Traits<NAME<TScalar>>::Base, TMapOptions>;                \
  };

#define HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(NAME) \
  HYPER_DECLARE_EIGEN_CLASS_TRAITS(Map, NAME)      \
  HYPER_DECLARE_EIGEN_CLASS_TRAITS(Ref, NAME)

#define HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_TRAITS(BASE, NAME, TYPE)                                              \
  template <typename TScalar, TYPE TArg, int TMapOptions>                                                         \
  struct Traits<Eigen::BASE<NAME<TScalar, TArg>, TMapOptions>> final : public Traits<NAME<TScalar, TArg>> {       \
    using Base = Eigen::BASE<typename Traits<NAME<TScalar, TArg>>::Base, TMapOptions>;                            \
  };                                                                                                              \
                                                                                                                  \
  template <typename TScalar, TYPE TArg, int TMapOptions>                                                         \
  struct Traits<Eigen::BASE<const NAME<TScalar, TArg>, TMapOptions>> final : public Traits<NAME<TScalar, TArg>> { \
    using Base = Eigen::BASE<const typename Traits<NAME<TScalar, TArg>>::Base, TMapOptions>;                      \
  };

#define HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(NAME, TYPE) \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_TRAITS(Map, NAME, TYPE)      \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_TRAITS(Ref, NAME, TYPE)

#define HYPER_DECLARE_EIGEN_CLASS(BASE, NAME, PLUGIN)                                                  \
  namespace Eigen {                                                                                    \
  template <typename TScalar, int TMapOptions>                                                         \
  class BASE<NAME<TScalar>, TMapOptions> final : public NAME##Base<BASE<NAME<TScalar>, TMapOptions>> { \
   public:                                                                                             \
    using Base = NAME##Base<BASE<NAME<TScalar>, TMapOptions>>;                                         \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(BASE)                                                           \
    PLUGIN                                                                                             \
  };                                                                                                   \
  }

#define HYPER_DECLARE_CONST_EIGEN_CLASS(BASE, NAME, PLUGIN)                                                        \
  namespace Eigen {                                                                                                \
  template <typename TScalar, int TMapOptions>                                                                     \
  class BASE<const NAME<TScalar>, TMapOptions> final : public NAME##Base<BASE<const NAME<TScalar>, TMapOptions>> { \
   public:                                                                                                         \
    using Base = NAME##Base<BASE<const NAME<TScalar>, TMapOptions>>;                                               \
    PLUGIN                                                                                                         \
  };                                                                                                               \
  }

#define HYPER_DECLARE_EIGEN_INTERFACE(NAME)                     \
  HYPER_DECLARE_EIGEN_CLASS(Map, NAME, using Base::Base;)       \
  HYPER_DECLARE_CONST_EIGEN_CLASS(Map, NAME, using Base::Base;) \
  HYPER_DECLARE_EIGEN_CLASS(Ref, NAME, using Base::Base;)       \
  HYPER_DECLARE_CONST_EIGEN_CLASS(Ref, NAME, using Base::Base;)

#define HYPER_DECLARE_TEMPLATED_EIGEN_CLASS(BASE, NAME, TYPE, PLUGIN)                                              \
  namespace Eigen {                                                                                                \
  template <typename TScalar, TYPE TArg, int TMapOptions>                                                          \
  class BASE<NAME<TScalar, TArg>, TMapOptions> final : public NAME##Base<BASE<NAME<TScalar, TArg>, TMapOptions>> { \
   public:                                                                                                         \
    using Base = NAME##Base<BASE<NAME<TScalar, TArg>, TMapOptions>>;                                               \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(BASE)                                                                       \
    PLUGIN                                                                                                         \
  };                                                                                                               \
  }

#define HYPER_DECLARE_TEMPLATED_CONST_EIGEN_CLASS(BASE, NAME, TYPE, PLUGIN)                                                    \
  namespace Eigen {                                                                                                            \
  template <typename TScalar, TYPE TArg, int TMapOptions>                                                                      \
  class BASE<const NAME<TScalar, TArg>, TMapOptions> final : public NAME##Base<BASE<const NAME<TScalar, TArg>, TMapOptions>> { \
   public:                                                                                                                     \
    using Base = NAME##Base<BASE<const NAME<TScalar, TArg>, TMapOptions>>;                                                     \
    PLUGIN                                                                                                                     \
  };                                                                                                                           \
  }

#define HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(NAME, TYPE)                     \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS(Map, NAME, TYPE, using Base::Base;)       \
  HYPER_DECLARE_TEMPLATED_CONST_EIGEN_CLASS(Map, NAME, TYPE, using Base::Base;) \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS(Ref, NAME, TYPE, using Base::Base;)       \
  HYPER_DECLARE_TEMPLATED_CONST_EIGEN_CLASS(Ref, NAME, TYPE, using Base::Base;)

#define HYPER_DECLARE_TANGENT_MAP_TRAITS(NAME)                                                            \
  template <typename TScalar, int TMapOptions>                                                            \
  struct Traits<Eigen::Map<Tangent<NAME<TScalar>>, TMapOptions>> : Traits<Tangent<NAME<TScalar>>> {       \
    using Base = typename Eigen::Map<typename Traits<Tangent<NAME<TScalar>>>::Base, TMapOptions>;         \
  };                                                                                                      \
                                                                                                          \
  template <typename TScalar, int TMapOptions>                                                            \
  struct Traits<Eigen::Map<const Tangent<NAME<TScalar>>, TMapOptions>> : Traits<Tangent<NAME<TScalar>>> { \
    using Base = typename Eigen::Map<const typename Traits<Tangent<NAME<TScalar>>>::Base, TMapOptions>;   \
  };

#define HYPER_DECLARE_ALGEBRA_MAP(NAME)                                                                                                   \
  namespace Eigen {                                                                                                                       \
  template <typename TScalar, int TMapOptions>                                                                                            \
  class Map<Algebra<NAME<TScalar>>, TMapOptions> final : public NAME##TangentBase<Map<Algebra<NAME<TScalar>>, TMapOptions>> {             \
   public:                                                                                                                                \
    using Base = NAME##TangentBase<Map<Algebra<NAME<TScalar>>, TMapOptions>>;                                                             \
    using Base::Base;                                                                                                                     \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)                                                                                               \
  };                                                                                                                                      \
                                                                                                                                          \
  template <typename TScalar, int TMapOptions>                                                                                            \
  class Map<const Algebra<NAME<TScalar>>, TMapOptions> final : public NAME##TangentBase<Map<const Algebra<NAME<TScalar>>, TMapOptions>> { \
   public:                                                                                                                                \
    using Base = NAME##TangentBase<Map<const Algebra<NAME<TScalar>>, TMapOptions>>;                                                       \
    using Base::Base;                                                                                                                     \
  };                                                                                                                                      \
  }

#define HYPER_DECLARE_TANGENT_MAP(NAME)                                                                                                   \
  namespace Eigen {                                                                                                                       \
  template <typename TScalar, int TMapOptions>                                                                                            \
  class Map<Tangent<NAME<TScalar>>, TMapOptions> final : public NAME##TangentBase<Map<Tangent<NAME<TScalar>>, TMapOptions>> {             \
   public:                                                                                                                                \
    using Base = NAME##TangentBase<Map<Tangent<NAME<TScalar>>, TMapOptions>>;                                                             \
    using Base::Base;                                                                                                                     \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)                                                                                               \
  };                                                                                                                                      \
                                                                                                                                          \
  template <typename TScalar, int TMapOptions>                                                                                            \
  class Map<const Tangent<NAME<TScalar>>, TMapOptions> final : public NAME##TangentBase<Map<const Tangent<NAME<TScalar>>, TMapOptions>> { \
   public:                                                                                                                                \
    using Base = NAME##TangentBase<Map<const Tangent<NAME<TScalar>>, TMapOptions>>;                                                       \
    using Base::Base;                                                                                                                     \
  };                                                                                                                                      \
  }

}  // namespace hyper::variables
