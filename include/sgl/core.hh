#pragma once

#include <array>
#include <type_traits>
#include <vector>

namespace sgl {

inline namespace v0 {

template <class T> struct dimension_traits;

template <size_t... Ns> struct dimensions {};

template <size_t... Ns> struct dimension_traits<dimensions<Ns...>> {
  static constexpr size_t rank = sizeof...(Ns);
  static constexpr size_t size = (Ns * ...);
  static constexpr std::array<size_t, rank> shape = {Ns...};
};

// clang-format off
template <class D,
	  class traits = dimension_traits<D>>
concept Dimensions = requires() {
  { traits::rank } -> size_t;
  { traits::size } -> size_t;
  { traits::shape } -> std::array<size_t, traits::rank>;
};
// clang-format on

static constexpr size_t dynamic = -1;

template <class T, size_t N, bool = (N == dynamic)> struct storage;

template <class T, size_t N> struct storage<T, N, true> {
  using storage_type = std::vector<T>;

  storage(size_t n) : data_(n) {}

  friend auto size(const storage &s) -> size_t { return s.data_.size(); }

private:
  storage_type data_;
};

template <class T, size_t N> struct storage<T, N, false> {
  using storage_type = std::array<T, N>;

private:
  storage_type data_;
};

template <class T, size_t N>
constexpr auto size(const storage<T, N, false> &) -> size_t {
  return N;
}

} // namespace v0

} // namespace sgl
