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

#define HYPER_DECLARE_EIGEN_CLASS_TRAITS(BASE, NAME)                                \
  template <int TMapOptions>                                                        \
  struct Traits<Eigen::BASE<NAME, TMapOptions>> final : public Traits<NAME> {       \
    using Base = Eigen::BASE<typename Traits<NAME>::Base, TMapOptions>;             \
  };                                                                                \
                                                                                    \
  template <int TMapOptions>                                                        \
  struct Traits<Eigen::BASE<const NAME, TMapOptions>> final : public Traits<NAME> { \
    using Base = Eigen::BASE<const typename Traits<NAME>::Base, TMapOptions>;       \
  };

#define HYPER_DECLARE_EIGEN_INTERFACE_TRAITS(NAME) \
  HYPER_DECLARE_EIGEN_CLASS_TRAITS(Map, NAME)      \
  HYPER_DECLARE_EIGEN_CLASS_TRAITS(Ref, NAME)

#define HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_TRAITS(BASE, NAME, TYPE)                            \
  template <TYPE TArg, int TMapOptions>                                                         \
  struct Traits<Eigen::BASE<NAME<TArg>, TMapOptions>> final : public Traits<NAME<TArg>> {       \
    using Base = Eigen::BASE<typename Traits<NAME<TArg>>::Base, TMapOptions>;                   \
  };                                                                                            \
                                                                                                \
  template <TYPE TArg, int TMapOptions>                                                         \
  struct Traits<Eigen::BASE<const NAME<TArg>, TMapOptions>> final : public Traits<NAME<TArg>> { \
    using Base = Eigen::BASE<const typename Traits<NAME<TArg>>::Base, TMapOptions>;             \
  };

#define HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE_TRAITS(NAME, TYPE) \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_TRAITS(Map, NAME, TYPE)      \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_TRAITS(Ref, NAME, TYPE)

#define HYPER_DECLARE_EIGEN_CLASS(BASE, NAME, PLUGIN)                                \
  namespace Eigen {                                                                  \
  template <int TMapOptions>                                                         \
  class BASE<NAME, TMapOptions> final : public NAME##Base<BASE<NAME, TMapOptions>> { \
   public:                                                                           \
    using Base = NAME##Base<BASE<NAME, TMapOptions>>;                                \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(BASE)                                         \
    PLUGIN                                                                           \
  };                                                                                 \
  }

#define HYPER_DECLARE_CONST_EIGEN_CLASS(BASE, NAME, PLUGIN)                                      \
  namespace Eigen {                                                                              \
  template <int TMapOptions>                                                                     \
  class BASE<const NAME, TMapOptions> final : public NAME##Base<BASE<const NAME, TMapOptions>> { \
   public:                                                                                       \
    using Base = NAME##Base<BASE<const NAME, TMapOptions>>;                                      \
    PLUGIN                                                                                       \
  };                                                                                             \
  }

#define HYPER_DECLARE_EIGEN_INTERFACE(NAME)                     \
  HYPER_DECLARE_EIGEN_CLASS(Map, NAME, using Base::Base;)       \
  HYPER_DECLARE_CONST_EIGEN_CLASS(Map, NAME, using Base::Base;) \
  HYPER_DECLARE_EIGEN_CLASS(Ref, NAME, using Base::Base;)       \
  HYPER_DECLARE_CONST_EIGEN_CLASS(Ref, NAME, using Base::Base;)

#define HYPER_DECLARE_TEMPLATED_EIGEN_CLASS(BASE, NAME, TYPE, PLUGIN)                            \
  namespace Eigen {                                                                              \
  template <TYPE TArg, int TMapOptions>                                                          \
  class BASE<NAME<TArg>, TMapOptions> final : public NAME##Base<BASE<NAME<TArg>, TMapOptions>> { \
   public:                                                                                       \
    using Base = NAME##Base<BASE<NAME<TArg>, TMapOptions>>;                                      \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(BASE)                                                     \
    PLUGIN                                                                                       \
  };                                                                                             \
  }

#define HYPER_DECLARE_TEMPLATED_CONST_EIGEN_CLASS(BASE, NAME, TYPE, PLUGIN)                                  \
  namespace Eigen {                                                                                          \
  template <TYPE TArg, int TMapOptions>                                                                      \
  class BASE<const NAME<TArg>, TMapOptions> final : public NAME##Base<BASE<const NAME<TArg>, TMapOptions>> { \
   public:                                                                                                   \
    using Base = NAME##Base<BASE<const NAME<TArg>, TMapOptions>>;                                            \
    PLUGIN                                                                                                   \
  };                                                                                                         \
  }

#define HYPER_DECLARE_TEMPLATED_EIGEN_INTERFACE(NAME, TYPE)                     \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS(Map, NAME, TYPE, using Base::Base;)       \
  HYPER_DECLARE_TEMPLATED_CONST_EIGEN_CLASS(Map, NAME, TYPE, using Base::Base;) \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS(Ref, NAME, TYPE, using Base::Base;)       \
  HYPER_DECLARE_TEMPLATED_CONST_EIGEN_CLASS(Ref, NAME, TYPE, using Base::Base;)

#define HYPER_DECLARE_TANGENT_MAP_TRAITS(NAME)                                                 \
  template <int TMapOptions>                                                                   \
  struct Traits<Eigen::Map<Tangent<NAME>, TMapOptions>> : Traits<Tangent<NAME>> {              \
    using Base = typename Eigen::Map<typename Traits<Tangent<NAME>>::Base, TMapOptions>;       \
  };                                                                                           \
                                                                                               \
  template <int TMapOptions>                                                                   \
  struct Traits<Eigen::Map<const Tangent<NAME>, TMapOptions>> : Traits<Tangent<NAME>> {        \
    using Base = typename Eigen::Map<const typename Traits<Tangent<NAME>>::Base, TMapOptions>; \
  };

#define HYPER_DECLARE_TANGENT_MAP(NAME)                                                                                 \
  namespace Eigen {                                                                                                     \
  template <int TMapOptions>                                                                                            \
  class Map<Tangent<NAME>, TMapOptions> final : public NAME##TangentBase<Map<Tangent<NAME>, TMapOptions>> {             \
   public:                                                                                                              \
    using Base = NAME##TangentBase<Map<Tangent<NAME>, TMapOptions>>;                                                    \
    using Base::Base;                                                                                                   \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)                                                                             \
  };                                                                                                                    \
                                                                                                                        \
  template <int TMapOptions>                                                                                            \
  class Map<const Tangent<NAME>, TMapOptions> final : public NAME##TangentBase<Map<const Tangent<NAME>, TMapOptions>> { \
   public:                                                                                                              \
    using Base = NAME##TangentBase<Map<const Tangent<NAME>, TMapOptions>>;                                              \
    using Base::Base;                                                                                                   \
  };                                                                                                                    \
  }

}  // namespace hyper::variables
