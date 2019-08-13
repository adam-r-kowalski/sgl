#pragma once

#include <array>
#include <cstdlib>
#include <type_traits>

namespace sgl {

namespace traits {

template <class T> struct size;

template <class T> struct shape;

template <class T> struct rank;

template <class T> struct value;

template <class T> struct reference { using type = typename value<T>::type &; };

template <class T> struct const_reference {
  using type = typename value<T>::type const &;
};

} // namespace traits

template <class T> constexpr size_t size_v = traits::size<T>::value;

template <class T> constexpr auto shape_v = traits::shape<T>::value;
template <class T> using shape_t = typename traits::shape<T>::type;

template <class T> constexpr auto rank_v = traits::rank<T>::value;

template <class T> using value_t = typename traits::value<T>::type;

template <class T> using reference_t = typename traits::reference<T>::type;

template <class T>
using const_reference_t = typename traits::const_reference<T>::type;

} // namespace sgl
