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

#define HYPER_DECLARE_EIGEN_CLASS_TRAITS(BASE, CLASS)                                 \
  template <int TMapOptions>                                                          \
  struct Traits<Eigen::BASE<CLASS, TMapOptions>> final : public Traits<CLASS> {       \
    using Base = Eigen::BASE<typename Traits<CLASS>::Base, TMapOptions>;              \
  };                                                                                  \
                                                                                      \
  template <int TMapOptions>                                                          \
  struct Traits<Eigen::BASE<const CLASS, TMapOptions>> final : public Traits<CLASS> { \
    using Base = Eigen::BASE<const typename Traits<CLASS>::Base, TMapOptions>;        \
  };

#define HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_TRAITS(BASE, CLASS, ARGTYPE)                          \
  template <ARGTYPE TArg, int TMapOptions>                                                        \
  struct Traits<Eigen::BASE<CLASS<TArg>, TMapOptions>> final : public Traits<CLASS<TArg>> {       \
    using Base = Eigen::BASE<typename Traits<CLASS<TArg>>::Base, TMapOptions>;                    \
  };                                                                                              \
                                                                                                  \
  template <ARGTYPE TArg, int TMapOptions>                                                        \
  struct Traits<Eigen::BASE<const CLASS<TArg>, TMapOptions>> final : public Traits<CLASS<TArg>> { \
    using Base = Eigen::BASE<const typename Traits<CLASS<TArg>>::Base, TMapOptions>;              \
  };

#define HYPER_DECLARE_EIGEN_CLASS(BASE, CLASS, PLUGIN)                                  \
  template <int TMapOptions>                                                            \
  class BASE<CLASS, TMapOptions> final : public CLASS##Base<BASE<CLASS, TMapOptions>> { \
   public:                                                                              \
    using Base = CLASS##Base<BASE<CLASS, TMapOptions>>;                                 \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(BASE)                                            \
    PLUGIN                                                                              \
  };

#define HYPER_DECLARE_CONST_EIGEN_CLASS(BASE, CLASS, PLUGIN)                                        \
  template <int TMapOptions>                                                                        \
  class BASE<const CLASS, TMapOptions> final : public CLASS##Base<BASE<const CLASS, TMapOptions>> { \
   public:                                                                                          \
    using Base = CLASS##Base<BASE<const CLASS, TMapOptions>>;                                       \
    PLUGIN                                                                                          \
  };

#define HYPER_DECLARE_TEMPLATED_EIGEN_CLASS(BASE, CLASS, ARGTYPE, PLUGIN)                           \
  template <ARGTYPE TArg, int TMapOptions>                                                          \
  class BASE<CLASS<TArg>, TMapOptions> final : public CLASS##Base<BASE<CLASS<TArg>, TMapOptions>> { \
   public:                                                                                          \
    using Base = CLASS##Base<BASE<CLASS<TArg>, TMapOptions>>;                                       \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(BASE)                                                        \
    PLUGIN                                                                                          \
  };

#define HYPER_DECLARE_TEMPLATED_CONST_EIGEN_CLASS(BASE, CLASS, ARGTYPE, PLUGIN)                                 \
  template <ARGTYPE TArg, int TMapOptions>                                                                      \
  class BASE<const CLASS<TArg>, TMapOptions> final : public CLASS##Base<BASE<const CLASS<TArg>, TMapOptions>> { \
   public:                                                                                                      \
    using Base = CLASS##Base<BASE<const CLASS<TArg>, TMapOptions>>;                                             \
    PLUGIN                                                                                                      \
  };

#define HYPER_DECLARE_EIGEN_CLASS_INTERFACE(NS, CLASS)               \
  namespace hyper {                                                  \
  HYPER_DECLARE_EIGEN_CLASS_TRAITS(Map, NS::CLASS)                   \
  HYPER_DECLARE_EIGEN_CLASS_TRAITS(Ref, NS::CLASS)                   \
  }                                                                  \
  namespace Eigen {                                                  \
  HYPER_DECLARE_EIGEN_CLASS(Map, NS::CLASS, using Base::Base;)       \
  HYPER_DECLARE_CONST_EIGEN_CLASS(Map, NS::CLASS, using Base::Base;) \
  HYPER_DECLARE_EIGEN_CLASS(Ref, NS::CLASS, using Base::Base;)       \
  HYPER_DECLARE_CONST_EIGEN_CLASS(Ref, NS::CLASS, using Base::Base;) \
  }

#define HYPER_DECLARE_EIGEN_CLASS_INTERFACE_NO_REF(NS, CLASS)        \
  namespace hyper {                                                  \
  HYPER_DECLARE_EIGEN_CLASS_TRAITS(Map, NS::CLASS)                   \
  }                                                                  \
  namespace Eigen {                                                  \
  HYPER_DECLARE_EIGEN_CLASS(Map, NS::CLASS, using Base::Base;)       \
  HYPER_DECLARE_CONST_EIGEN_CLASS(Map, NS::CLASS, using Base::Base;) \
  }

#define HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_INTERFACE(NS, CLASS, ARGTYPE)               \
  namespace hyper {                                                                     \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_TRAITS(Map, NS::CLASS, ARGTYPE)                   \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_TRAITS(Ref, NS::CLASS, ARGTYPE)                   \
  }                                                                                     \
  namespace Eigen {                                                                     \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS(Map, NS::CLASS, ARGTYPE, using Base::Base;)       \
  HYPER_DECLARE_TEMPLATED_CONST_EIGEN_CLASS(Map, NS::CLASS, ARGTYPE, using Base::Base;) \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS(Ref, NS::CLASS, ARGTYPE, using Base::Base;)       \
  HYPER_DECLARE_TEMPLATED_CONST_EIGEN_CLASS(Ref, NS::CLASS, ARGTYPE, using Base::Base;) \
  }

#define HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_INTERFACE_NO_REF(NS, CLASS, ARGTYPE)        \
  namespace hyper {                                                                     \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS_TRAITS(Map, NS::CLASS, ARGTYPE)                   \
  }                                                                                     \
  namespace Eigen {                                                                     \
  HYPER_DECLARE_TEMPLATED_EIGEN_CLASS(Map, NS::CLASS, ARGTYPE, using Base::Base;)       \
  HYPER_DECLARE_TEMPLATED_CONST_EIGEN_CLASS(Map, NS::CLASS, ARGTYPE, using Base::Base;) \
  }

#define HYPER_DECLARE_EIGEN_TANGENT_INTERFACE(NS, CLASS)                                                                                       \
  namespace hyper {                                                                                                                            \
  template <int TMapOptions>                                                                                                                   \
  struct Traits<Eigen::Map<NS::Tangent<NS::CLASS>, TMapOptions>> : Traits<NS::Tangent<NS::CLASS>> {                                            \
    using Base = typename Eigen::Map<typename Traits<NS::Tangent<NS::CLASS>>::Base, TMapOptions>;                                              \
  };                                                                                                                                           \
                                                                                                                                               \
  template <int TMapOptions>                                                                                                                   \
  struct Traits<Eigen::Map<const NS::Tangent<NS::CLASS>, TMapOptions>> : Traits<NS::Tangent<NS::CLASS>> {                                      \
    using Base = typename Eigen::Map<const typename Traits<NS::Tangent<NS::CLASS>>::Base, TMapOptions>;                                        \
  };                                                                                                                                           \
  }                                                                                                                                            \
  namespace Eigen {                                                                                                                            \
  template <int TMapOptions>                                                                                                                   \
  class Map<NS::Tangent<NS::CLASS>, TMapOptions> final : public NS::CLASS##TangentBase<Map<NS::Tangent<NS::CLASS>, TMapOptions>> {             \
   public:                                                                                                                                     \
    using Base = NS::CLASS##TangentBase<Map<NS::Tangent<NS::CLASS>, TMapOptions>>;                                                             \
    using Base::Base;                                                                                                                          \
    HYPER_INHERIT_ASSIGNMENT_OPERATORS(Map)                                                                                                    \
  };                                                                                                                                           \
                                                                                                                                               \
  template <int TMapOptions>                                                                                                                   \
  class Map<const NS::Tangent<NS::CLASS>, TMapOptions> final : public NS::CLASS##TangentBase<Map<const NS::Tangent<NS::CLASS>, TMapOptions>> { \
   public:                                                                                                                                     \
    using Base = NS::CLASS##TangentBase<Map<const NS::Tangent<NS::CLASS>, TMapOptions>>;                                                       \
    using Base::Base;                                                                                                                          \
  };                                                                                                                                           \
  }

}  // namespace hyper::variables
