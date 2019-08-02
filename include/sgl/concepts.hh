#pragma once

#include <type_traits>
#include <iterator>

#include <sgl/traits.hh>

namespace sgl {

// clang-format off
template <class T, class U>
concept same_as = std::is_same_v<T, U> && std::is_same_v<U, T>;
// clang-format on

template <class From, class To>
concept convertible_to = std::is_convertible_v<From, To>;

// clang-format off
template <class I>
concept iterator_type =
  requires(I i) {
    { *i } -> typename std::iterator_traits<I>::reference;
    { ++i } -> I&;
  };
// clang-format on

template <iterator_type I> struct traits::value<I> {
  using type = typename std::iterator_traits<I>::value_type;
};

// clang-format off
template <class R> concept range = requires(R r) {
  { begin(r) } -> iterator_type;
  { end(r) } -> iterator_type;
};
// clang-format on

template <range R> struct traits::value<R> { using type = typename R::value_type; };

// clang-format off
template <class F, class Result, class... Args>
concept function = requires(F f, Args&&... args) {
  { f(std::forward<Args>(args)...) } -> Result;
};
// clang-format on

constexpr size_t dynamic = -1;
template <size_t N> concept dynamic_v = N == dynamic;

// clang-format off
template <class D>
concept dimensions = requires() {
  { size_v<D> } -> size_t;
  { shape_v<D> } -> shape_t<D>;
};
// clang-format on

// clang-format off
template <class L, class Shape = shape_t<L>> concept layout =
  requires(L const cl, Shape shape, Shape cartesian_index) {
    { L{shape} } -> L;
    { cl.linear_index(cartesian_index) } -> size_t;
  };
// clang-format on

// clang-format off
template <class S> concept storage =
  requires(S s, S const cs, size_t index) {
    { s[index] } -> reference_t<S>;
    { cs[index] } -> const_reference_t<S>;
  };
// clang-format on


}
